"""Tests for LLM-judge-based clustering."""
import pytest

from s_edl.evaluation.clustering import ClusterClassifier, cluster_predictions, cluster_predictions_with_ground_truth
from s_edl.evaluation.judge import _parse_cluster_response


# =============================================================================
# Tests for _parse_cluster_response
# =============================================================================
class TestParseJudgeResponse:

    def test_valid_index(self):
        assert _parse_cluster_response('Matching index: 1') == 1

    def test_valid_index_zero(self):
        assert _parse_cluster_response('Matching index: 0') == 0

    def test_negative_one_returns_none(self):
        assert _parse_cluster_response('Matching index: -1') is None

    def test_negative_other_returns_none(self):
        assert _parse_cluster_response('Matching index: -3') is None

    def test_with_surrounding_text(self):
        text = 'The candidate matches representative 1.\nMatching index: 1'
        assert _parse_cluster_response(text) == 1

    def test_with_extra_whitespace(self):
        assert _parse_cluster_response('Matching index:  2') == 2

    def test_malformed_no_match(self):
        assert _parse_cluster_response('blah blah no index here') is None

    def test_empty_string(self):
        assert _parse_cluster_response('') is None

    def test_large_index(self):
        assert _parse_cluster_response('Matching index: 42') == 42


# =============================================================================
# Tests for clustering with mock classifier
# =============================================================================
def _make_exact_match_classifier() -> ClusterClassifier:
    """Create a classifier that matches by exact string equality."""

    def _classify(prediction: str, representatives: list[str]) -> int | None:
        for i, rep in enumerate(representatives):
            if prediction == rep:
                return i
        return None

    return _classify


class TestClusterPredictionsWithClassifier:

    def test_all_same_prediction(self):
        """All identical predictions should cluster together."""
        classifier = _make_exact_match_classifier()
        predictions = ['42', '42', '42']
        clusters = cluster_predictions(predictions, classifier=classifier)

        assert len(clusters) == 1
        assert len(clusters[0]) == 3
        assert clusters[0].representative == '42'

    def test_two_groups(self):
        """Two groups of predictions should form two clusters."""
        classifier = _make_exact_match_classifier()
        predictions = ['A', 'B', 'A', 'B', 'A']
        clusters = cluster_predictions(predictions, classifier=classifier)

        assert len(clusters) == 2
        sizes = sorted([len(c) for c in clusters], reverse=True)
        assert sizes == [3, 2]

    def test_no_matches_all_singletons(self):
        """When nothing matches, each prediction is its own cluster."""
        classifier = _make_exact_match_classifier()
        predictions = ['a', 'b', 'c']
        clusters = cluster_predictions(predictions, classifier=classifier)

        assert len(clusters) == 3
        for c in clusters:
            assert len(c) == 1

    def test_is_parseable_true_for_all_llm_clusters(self):
        """In LLM mode, all clusters should have is_parseable=True."""
        classifier = _make_exact_match_classifier()
        predictions = ['foo', 'bar']
        clusters = cluster_predictions(predictions, classifier=classifier)

        for c in clusters:
            assert c.is_parseable is True

    def test_indices_are_correct(self):
        """Verify indices map back to original prediction positions."""
        classifier = _make_exact_match_classifier()
        predictions = ['X', 'Y', 'X', 'Z', 'Y']
        clusters = cluster_predictions(predictions, classifier=classifier)

        # Build a map: representative -> indices
        cluster_map = {c.representative: c.indices for c in clusters}
        assert cluster_map['X'] == [0, 2]
        assert cluster_map['Y'] == [1, 4]
        assert cluster_map['Z'] == [3]


class TestClusterWithGroundTruthAndClassifier:

    def test_gt_matches_existing_cluster(self):
        """GT should match an existing cluster when classifier says so."""
        classifier = _make_exact_match_classifier()
        predictions = ['42', '42', '40', '40']
        clusters, correct_idx = cluster_predictions_with_ground_truth(predictions, '42', classifier=classifier)

        assert clusters[correct_idx].representative == '42'
        assert len(clusters[correct_idx]) == 2

    def test_gt_added_when_no_match(self):
        """GT should be added as empty cluster when no prediction matches."""
        classifier = _make_exact_match_classifier()
        predictions = ['40', '41']
        clusters, correct_idx = cluster_predictions_with_ground_truth(predictions, '99', classifier=classifier)

        assert clusters[correct_idx].representative == '99'
        assert clusters[correct_idx].predictions == []
        assert clusters[correct_idx].indices == []
        assert clusters[correct_idx].is_parseable is True


class TestBackwardCompatibility:
    """Ensure calling without classifier still uses math-verify."""

    def test_no_classifier_same_as_before(self):
        predictions = ['42', '\\frac{84}{2}', '0.5']
        clusters_default = cluster_predictions(predictions)
        clusters_none = cluster_predictions(predictions, classifier=None)

        assert len(clusters_default) == len(clusters_none)
        for c1, c2 in zip(clusters_default, clusters_none):
            assert c1.representative == c2.representative
            assert c1.predictions == c2.predictions
            assert c1.indices == c2.indices

    def test_no_classifier_gt_same_as_before(self):
        predictions = ['42', '42', '40']
        gt = '42'
        clusters1, idx1 = cluster_predictions_with_ground_truth(predictions, gt)
        clusters2, idx2 = cluster_predictions_with_ground_truth(predictions, gt, classifier=None)

        assert idx1 == idx2
        assert len(clusters1) == len(clusters2)


# =============================================================================
# GPU integration test: full clustering with real judge LLM
# =============================================================================
@pytest.mark.gpu
class TestLLMJudgeFullClustering:
    """Integration tests with a real judge model.

    Requires GPU.
    """

    @pytest.fixture(scope='class')
    def judge_classifier(self) -> ClusterClassifier:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        from s_edl.evaluation import build_cluster_judge

        model_name = 'openai/gpt-oss-20b'
        llm = LLM(
            model=model_name,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        sampling_params = SamplingParams(temperature=0.6, top_p=1.0, max_tokens=256)
        return build_cluster_judge(
            llm=llm,
            sampling_params=sampling_params,
            tokenizer=tokenizer,
            chat_template_kwargs={'reasoning_effort': 'low'},
        )

    def test_numeric_with_units(self, judge_classifier: ClusterClassifier):
        """Test clustering numeric answers with units and rounding."""
        ground_truth = '1301.8667 J'
        predictions = [
            '1301.8 J',
            '1301.8667 kJ',
            '1301.8667 J',
            '1031.8667 J',
            'about 1302 J',
            '1301.8667 J',
        ]

        clusters, correct_idx = cluster_predictions_with_ground_truth(
            predictions, ground_truth, classifier=judge_classifier)

        # Collect predictions per cluster for verification
        correct_preds = set(clusters[correct_idx].predictions)
        all_preds = []
        for c in clusters:
            all_preds.extend(c.predictions)

        # All predictions must be accounted for
        assert len(all_preds) == len(predictions)

        # Correct cluster should contain the equivalent values
        assert '1301.8 J' in correct_preds
        assert '1301.8667 J' in correct_preds
        assert 'about 1302 J' in correct_preds
        assert len(correct_preds) >= 3

        # Wrong-unit answer should NOT be in the correct cluster
        assert '1301.8667 kJ' not in correct_preds

        # Wrong-value answer should NOT be in the correct cluster
        assert '1031.8667 J' not in correct_preds

    def test_non_numeric_semantic_equivalence(self, judge_classifier: ClusterClassifier):
        """Test clustering text-based science answers."""
        ground_truth = 'Newton\'s third law of motion'
        predictions = [
            'Newton\'s 3rd law',
            'the third law of Newton',
            'conservation of momentum',
            'Newton\'s third law of motion',
            'every action has an equal and opposite reaction',
            'Newton\'s second law',
        ]

        clusters, correct_idx = cluster_predictions_with_ground_truth(
            predictions, ground_truth, classifier=judge_classifier)

        correct_preds = set(clusters[correct_idx].predictions)
        all_preds = []
        for c in clusters:
            all_preds.extend(c.predictions)

        # All predictions must be accounted for
        assert len(all_preds) == len(predictions)

        # Semantically equivalent answers should be in the correct cluster
        assert 'Newton\'s 3rd law' in correct_preds
        assert 'the third law of Newton' in correct_preds
        assert 'Newton\'s third law of motion' in correct_preds

        # Newton's second law is a different law — should NOT be correct
        assert 'Newton\'s second law' not in correct_preds

    def test_gt_not_in_predictions(self, judge_classifier: ClusterClassifier):
        """Test that GT is added as empty cluster when no prediction matches."""
        ground_truth = '9.81 m/s^2'
        predictions = [
            '3.14 rad/s',
            '3.14 rad/s',
            '6.67e-11 N m^2/kg^2',
            '3.0e8 m/s',
            '3.0e8 m/s',
        ]

        clusters, correct_idx = cluster_predictions_with_ground_truth(
            predictions, ground_truth, classifier=judge_classifier)

        all_preds = []
        for c in clusters:
            all_preds.extend(c.predictions)

        # All predictions must be accounted for
        assert len(all_preds) == len(predictions)

        # GT cluster should be empty (no prediction matched)
        assert clusters[correct_idx].representative == ground_truth
        assert clusters[correct_idx].predictions == []
        assert clusters[correct_idx].indices == []

        # None of the predictions should be in the GT cluster
        for pred in predictions:
            assert pred not in clusters[correct_idx].predictions


# =============================================================================
# GPU integration test: SubprocessClusterJudge
# =============================================================================
@pytest.mark.gpu
class TestSubprocessClusterJudge:
    """Integration tests for subprocess-based judge.

    Requires GPU.
    """

    @pytest.fixture(scope='class')
    def subprocess_classifier(self):
        from s_edl.evaluation import SubprocessClusterJudge

        classifier = SubprocessClusterJudge(
            model_name='openai/gpt-oss-20b',
            gpu_id=0,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            max_num_seqs=64,
            sampling_kwargs={
                'temperature': 0.6, 'top_p': 1.0, 'max_tokens': 256
            },
            reasoning_effort='low',
        )
        yield classifier
        classifier.shutdown()

    def test_classify_matching(self, subprocess_classifier):
        """Subprocess classifier should find a matching representative."""
        result = subprocess_classifier('42', ['41', '42', '43'])
        assert result == 1

    def test_classify_no_match(self, subprocess_classifier):
        """Subprocess classifier should return None when no match."""
        result = subprocess_classifier('999', ['1', '2', '3'])
        assert result is None

    def test_full_clustering(self, subprocess_classifier):
        """Full clustering pipeline with subprocess classifier."""
        predictions = ['42', '42', '40', '40', '42']
        clusters, correct_idx = cluster_predictions_with_ground_truth(
            predictions, '42', classifier=subprocess_classifier)

        assert clusters[correct_idx].representative == '42'
        assert len(clusters[correct_idx]) == 3
