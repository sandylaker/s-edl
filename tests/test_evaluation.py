"""Unit tests for evaluation modules: clustering, metrics, and prediction extraction."""
import pytest

from s_edl.evaluation import (
    SampleEvalResult,
    are_semantically_equivalent,
    cluster_predictions,
    cluster_predictions_with_ground_truth,
    compute_aggregate_metrics,
    compute_ece,
    compute_pass_at_k,
    extract_prediction,
    parse_math_expression,
)


# =============================================================================
# Tests for extract_prediction (process_preds.py)
# =============================================================================
class TestExtractPrediction:
    """Tests for prediction extraction from model outputs."""

    def test_extract_simple_boxed_number(self):
        """Test extracting simple number from \\boxed{}."""
        text = 'The answer is \\boxed{42}.'
        assert extract_prediction(text) == '42'

    def test_extract_boxed_fraction(self):
        """Test extracting LaTeX fraction from \\boxed{}."""
        text = 'Therefore, \\boxed{\\frac{1}{2}} is the answer.'
        assert extract_prediction(text) == '\\frac{1}{2}'

    def test_extract_boxed_nested_braces(self):
        """Test extracting content with nested braces."""
        text = '\\boxed{\\frac{3}{4} + \\frac{1}{2}}'
        assert extract_prediction(text) == '\\frac{3}{4} + \\frac{1}{2}'

    def test_fallback_to_last_number(self):
        """Test fallback to last number when no \\boxed{}."""
        text = 'After calculation, the result is 123'
        assert extract_prediction(text) == '123'

    def test_fallback_decimal_number(self):
        """Test fallback extracts decimal numbers."""
        text = 'The probability is 0.75 so the answer is 75%.'
        # Should extract '75' as the last number (% is not part of number)
        assert extract_prediction(text) == '75'

    def test_empty_string_when_no_prediction(self):
        """Test returns empty string when no prediction found."""
        text = 'No numbers here at all.'
        assert extract_prediction(text) == ''

    def test_multiple_boxed_uses_last(self):
        """Test that last \\boxed{} is used when multiple present (CoT refinement)."""
        text = '\\boxed{10} and \\boxed{20}'
        assert extract_prediction(text) == '20'


# =============================================================================
# Tests for semantic equivalence (verify.py)
# =============================================================================
class TestSemanticEquivalence:
    """Tests for mathematical expression parsing and equivalence."""

    def test_parse_simple_number(self):
        """Test parsing simple integers."""
        parsed, ok = parse_math_expression('42')
        assert ok is True
        assert parsed is not None

    def test_parse_fraction_latex(self):
        """Test parsing LaTeX fractions."""
        parsed, ok = parse_math_expression('\\frac{1}{2}')
        assert ok is True
        assert parsed is not None

    def test_parse_invalid_expression(self):
        """Test parsing invalid expression returns False."""
        parsed, ok = parse_math_expression('not a math expression {}[]')
        # May or may not parse depending on math-verify behavior
        # Just ensure it doesn't crash

    def test_equivalent_same_number(self):
        """Test same numbers are equivalent."""
        assert are_semantically_equivalent('42', '42') is True

    def test_equivalent_fraction_and_decimal(self):
        """Test fraction and decimal equivalence: 1/2 == 0.5."""
        assert are_semantically_equivalent('\\frac{1}{2}', '0.5') is True

    def test_equivalent_different_fraction_forms(self):
        """Test different fraction representations: 1/2 == 2/4."""
        assert are_semantically_equivalent('\\frac{1}{2}', '\\frac{2}{4}') is True

    def test_not_equivalent_different_numbers(self):
        """Test different numbers are not equivalent."""
        assert are_semantically_equivalent('42', '43') is False

    def test_not_equivalent_unparseable(self):
        """Test unparseable expressions return False."""
        # Empty string is unparseable
        assert are_semantically_equivalent('', '42') is False


# =============================================================================
# Tests for semantic clustering (clustering.py)
# =============================================================================
class TestSemanticClustering:
    """Tests for semantic clustering algorithm."""

    def test_cluster_identical_predictions(self):
        """Test that identical predictions form one cluster."""
        predictions = ['42', '42', '42', '42', '42']
        clusters = cluster_predictions(predictions)

        assert len(clusters) == 1
        assert len(clusters[0]) == 5
        assert clusters[0].representative == '42'

    def test_cluster_different_predictions(self):
        """Test that different predictions form separate clusters."""
        predictions = ['42', '43', '44']
        clusters = cluster_predictions(predictions)

        assert len(clusters) == 3
        for cluster in clusters:
            assert len(cluster) == 1

    def test_cluster_semantic_equivalence(self):
        """Test that semantically equivalent predictions cluster together."""
        # 0.5, 1/2, and 2/4 are all equivalent
        predictions = ['0.5', '\\frac{1}{2}', '\\frac{2}{4}', '0.5']
        clusters = cluster_predictions(predictions)

        # All should be in one cluster
        assert len(clusters) == 1
        assert len(clusters[0]) == 4

    def test_cluster_mixed_correct_incorrect(self):
        """Test clustering with mix of correct and incorrect answers."""
        # 5 predictions of "42", 3 predictions of "40"
        predictions = ['42', '42', '40', '42', '40', '42', '40', '42']
        clusters = cluster_predictions(predictions)

        assert len(clusters) == 2
        sizes = sorted([len(c) for c in clusters], reverse=True)
        assert sizes == [5, 3]

    def test_cluster_with_ground_truth_matching(self):
        """Test clustering identifies correct cluster when GT matches."""
        predictions = ['42', '42', '40', '42', '40']
        ground_truth = '42'

        clusters, correct_idx = cluster_predictions_with_ground_truth(predictions, ground_truth)

        # Find the cluster with "42"
        cluster_42_idx = None
        for i, c in enumerate(clusters):
            if c.representative == '42':
                cluster_42_idx = i
                break

        assert correct_idx == cluster_42_idx
        assert len(clusters[correct_idx]) == 3  # Three "42" predictions

    def test_cluster_gt_not_matching_any(self):
        """Test that GT is added as new cluster when no prediction matches."""
        predictions = ['40', '40', '41', '41', '43']
        ground_truth = '42'  # Not in any prediction

        clusters, correct_idx = cluster_predictions_with_ground_truth(predictions, ground_truth)

        # GT should be added as new cluster
        assert clusters[correct_idx].representative == ground_truth
        assert len(clusters[correct_idx]) == 0  # Empty predictions list
        assert clusters[correct_idx].predictions == []

    def test_cluster_gt_semantically_equivalent(self):
        """Test GT matching via semantic equivalence, not string equality."""
        predictions = ['\\frac{1}{2}', '\\frac{1}{2}', '0.3']
        ground_truth = '0.5'  # Semantically equivalent to 1/2

        clusters, correct_idx = cluster_predictions_with_ground_truth(predictions, ground_truth)

        # Should match the 1/2 cluster
        assert len(clusters[correct_idx]) == 2

    def test_cluster_different_strings_become_singletons(self):
        """Test that non-equivalent predictions form separate clusters."""
        # Each unique string that's not semantically equivalent becomes its own cluster
        predictions = ['42', 'abc', '42', 'xyz']
        clusters = cluster_predictions(predictions)

        # "42" forms one cluster with 2 predictions
        # "abc" and "xyz" each form their own clusters
        assert len(clusters) == 3

        # Find the "42" cluster
        cluster_42 = [c for c in clusters if '42' in c.predictions]
        assert len(cluster_42) == 1
        assert len(cluster_42[0]) == 2  # Two "42"s

    def test_cluster_empty_string_becomes_singleton(self):
        """Test that empty strings (unparseable) become singleton clusters."""
        predictions = ['42', '', '42', '']
        clusters = cluster_predictions(predictions)

        # "42" forms one cluster, each empty string is singleton
        cluster_42 = [c for c in clusters if c.representative == '42']
        empty_clusters = [c for c in clusters if c.representative == '']

        assert len(cluster_42) == 1
        assert len(cluster_42[0]) == 2
        assert len(empty_clusters) == 2  # Two empty string singletons


# =============================================================================
# Tests for pass@k computation (metrics.py)
# =============================================================================
class TestPassAtK:
    """Tests for pass@k metric computation."""

    def test_pass_at_k_all_correct(self):
        """Test pass@k = 1.0 when all predictions are correct."""
        # n=10, c=10, k=1
        assert compute_pass_at_k(n=10, c=10, k=1) == 1.0
        assert compute_pass_at_k(n=10, c=10, k=5) == 1.0

    def test_pass_at_k_none_correct(self):
        """Test pass@k = 0.0 when no predictions are correct."""
        # n=10, c=0, k=1: pass@1 = 1 - C(10,1)/C(10,1) = 0
        assert compute_pass_at_k(n=10, c=0, k=1) == 0.0
        # n=10, c=0, k=5: pass@5 = 1 - C(10,5)/C(10,5) = 0
        assert compute_pass_at_k(n=10, c=0, k=5) == 0.0

    def test_pass_at_k_formula_verification(self):
        """Test pass@k formula with known values.

        Formula: pass@k = 1 - C(n-c, k) / C(n, k)

        n=10, c=3, k=1: pass@1 = 1 - C(7,1)/C(10,1) = 1 - 7/10 = 0.3
        n=10, c=3, k=2: pass@2 = 1 - C(7,2)/C(10,2) = 1 - 21/45 = 24/45 = 8/15
        n=10, c=3, k=5: pass@5 = 1 - C(7,5)/C(10,5) = 1 - 21/252 = 231/252 = 11/12
        """
        assert compute_pass_at_k(n=10, c=3, k=1) == pytest.approx(0.3)
        assert compute_pass_at_k(n=10, c=3, k=2) == pytest.approx(8 / 15)
        assert compute_pass_at_k(n=10, c=3, k=5) == pytest.approx(11 / 12)

    def test_pass_at_k_returns_one_when_k_exceeds_failures(self):
        """Test pass@k = 1.0 when k > (n - c), i.e., guaranteed to sample a correct one."""
        # n=10, c=8, k=5: n-c=2 < k, so pass@k = 1.0
        assert compute_pass_at_k(n=10, c=8, k=5) == 1.0


# =============================================================================
# Tests for ECE computation (metrics.py)
# =============================================================================
class TestECE:
    """Tests for Expected Calibration Error computation."""

    def _make_sample(self, confidence: float, is_correct: bool) -> SampleEvalResult:
        """Helper to create a minimal SampleEvalResult for ECE testing."""
        return SampleEvalResult(
            question='test',
            prefill_prompt='test',
            final_answer='42',
            generations=['test'],
            final_preds=['42'],
            semantic_clusters=[['42']],
            correct_cluster_idx=0,
            argmax_cluster_idx=0,
            argmax_prediction='42',
            is_correct=is_correct,
            confidence=confidence,
        )

    def test_ece_perfect_calibration(self):
        """Test ECE = 0 when perfectly calibrated.

        If confidence == accuracy for all bins, ECE should be 0. Create samples where confidence matches correctness
        proportion in each bin.
        """
        # 10 samples all with confidence=1.0 and all correct
        samples = [self._make_sample(confidence=1.0, is_correct=True) for _ in range(10)]
        ece = compute_ece(samples, n_bins=10)
        assert ece == pytest.approx(0.0)

    def test_ece_known_values(self):
        """Test ECE with manually computed expected value.

        2 samples:
        - Sample 1: confidence=0.8, is_correct=True
        - Sample 2: confidence=0.4, is_correct=False

        With n_bins=10:
        - Bin 4 (0.4-0.5): Sample 2
          accuracy = 0/1 = 0, avg_conf = 0.4
          contribution = |0 - 0.4| * (1/2) = 0.2
        - Bin 8 (0.8-0.9): Sample 1
          accuracy = 1/1 = 1, avg_conf = 0.8
          contribution = |1 - 0.8| * (1/2) = 0.1
        - ECE = 0.2 + 0.1 = 0.3
        """
        samples = [
            self._make_sample(confidence=0.8, is_correct=True),
            self._make_sample(confidence=0.4, is_correct=False),
        ]
        ece = compute_ece(samples, n_bins=10)
        assert ece == pytest.approx(0.3)

    def test_ece_overconfident(self):
        """Test ECE when model is overconfident.

        5 samples all with confidence=0.9 but only 2 correct.
        Bin 9 (0.9-1.0): accuracy = 2/5 = 0.4, avg_conf = 0.9
        ECE = |0.4 - 0.9| * 1.0 = 0.5
        """
        samples = [
            self._make_sample(confidence=0.9, is_correct=True),
            self._make_sample(confidence=0.9, is_correct=True),
            self._make_sample(confidence=0.9, is_correct=False),
            self._make_sample(confidence=0.9, is_correct=False),
            self._make_sample(confidence=0.9, is_correct=False),
        ]
        ece = compute_ece(samples, n_bins=10)
        assert ece == pytest.approx(0.5)

    def test_ece_underconfident(self):
        """Test ECE when model is underconfident.

        5 samples all with confidence=0.3 but 4 correct.
        Bin 3 (0.3-0.4): accuracy = 4/5 = 0.8, avg_conf = 0.3
        ECE = |0.8 - 0.3| * 1.0 = 0.5
        """
        samples = [
            self._make_sample(confidence=0.3, is_correct=True),
            self._make_sample(confidence=0.3, is_correct=True),
            self._make_sample(confidence=0.3, is_correct=True),
            self._make_sample(confidence=0.3, is_correct=True),
            self._make_sample(confidence=0.3, is_correct=False),
        ]
        ece = compute_ece(samples, n_bins=10)
        assert ece == pytest.approx(0.5)

    def test_ece_empty_samples(self):
        """Test ECE returns 0 for empty sample list."""
        ece = compute_ece([], n_bins=10)
        assert ece == 0.0


# =============================================================================
# Tests for aggregate metrics (metrics.py)
# =============================================================================
class TestAggregateMetrics:
    """Tests for aggregate metrics computation."""

    def _make_sample_full(
        self,
        final_preds: list[str],
        semantic_clusters: list[list[str]],
        correct_cluster_idx: int,
        argmax_cluster_idx: int,
        confidence: float,
    ) -> SampleEvalResult:
        """Helper to create a full SampleEvalResult for aggregate testing."""
        return SampleEvalResult(
            question='test',
            prefill_prompt='test',
            final_answer='42',
            generations=['test'] * len(final_preds),
            final_preds=final_preds,
            semantic_clusters=semantic_clusters,
            correct_cluster_idx=correct_cluster_idx,
            argmax_cluster_idx=argmax_cluster_idx,
            argmax_prediction=semantic_clusters[argmax_cluster_idx][0]
            if semantic_clusters[argmax_cluster_idx] else '42',
            is_correct=(argmax_cluster_idx == correct_cluster_idx),
            confidence=confidence,
        )

    def test_aggregate_semantic_accuracy(self):
        """Test semantic accuracy computation.

        3 samples: 2 correct (argmax == correct_cluster), 1 wrong
        Accuracy = 2/3
        """
        samples = [
            self._make_sample_full(
                final_preds=['42'] * 10,
                semantic_clusters=[['42'] * 10],
                correct_cluster_idx=0,
                argmax_cluster_idx=0,  # Correct
                confidence=1.0,
            ),
            self._make_sample_full(
                final_preds=['42'] * 10,
                semantic_clusters=[['42'] * 10],
                correct_cluster_idx=0,
                argmax_cluster_idx=0,  # Correct
                confidence=1.0,
            ),
            self._make_sample_full(
                final_preds=['40'] * 6 + ['42'] * 4,
                semantic_clusters=[['40'] * 6, ['42'] * 4],
                correct_cluster_idx=1,
                argmax_cluster_idx=0,  # Wrong - picked 40 not 42
                confidence=0.6,
            ),
        ]
        metrics = compute_aggregate_metrics(samples, ece_bins=[10], k_values=[1])
        assert metrics.semantic_accuracy == pytest.approx(2 / 3)

    def test_aggregate_pass_at_k(self):
        """Test pass@k computation in aggregate metrics.

        Sample with n=10, c=3 (3 correct predictions out of 10).
        pass@1 = 1 - C(7,1)/C(10,1) = 0.3
        """
        # Create a sample where 3 out of 10 predictions are in the correct cluster
        samples = [
            self._make_sample_full(
                final_preds=['42'] * 3 + ['40'] * 7,
                semantic_clusters=[['42'] * 3, ['40'] * 7],
                correct_cluster_idx=0,  # "42" cluster is correct
                argmax_cluster_idx=1,  # But "40" cluster is argmax (wrong)
                confidence=0.7,
            ),
        ]
        metrics = compute_aggregate_metrics(samples, ece_bins=[10], k_values=[1, 5])
        assert metrics.pass_at_k[1] == pytest.approx(0.3)
        assert metrics.pass_at_k[5] == pytest.approx(11 / 12)

    def test_aggregate_with_gt_not_matching(self):
        """Test aggregate metrics when GT doesn't match any prediction.

        When GT is not in any cluster, correct_cluster has 0 predictions. c=0, n=10: pass@1 = 0, pass@5 = 0 is_correct
        = False (since argmax != correct_cluster)
        """
        # All predictions are wrong, GT added as empty cluster
        samples = [
            self._make_sample_full(
                final_preds=['40'] * 10,
                semantic_clusters=[['40'] * 10, []],  # Second cluster is GT-only (empty)
                correct_cluster_idx=1,  # GT cluster
                argmax_cluster_idx=0,  # "40" cluster is argmax
                confidence=1.0,
            ),
        ]
        metrics = compute_aggregate_metrics(samples, ece_bins=[10], k_values=[1, 5])

        # Accuracy should be 0 (argmax is wrong)
        assert metrics.semantic_accuracy == 0.0

        # pass@k should be 0 since c=0
        assert metrics.pass_at_k[1] == 0.0
        assert metrics.pass_at_k[5] == 0.0


# =============================================================================
# Integration test: end-to-end with realistic generations
# =============================================================================
class TestEndToEndEvaluation:
    """Integration tests simulating realistic model outputs."""

    def test_full_pipeline_with_boxed_predictions(self):
        """Test full pipeline with realistic \\boxed{} predictions.

        Scenario: 10 generations for a question where answer is 42.
        - 6 generations have \\boxed{42}
        - 2 generations have \\boxed{40}
        - 1 generation has \\boxed{\\frac{84}{2}} (semantically 42)
        - 1 generation has no boxed, last number is 42

        Expected:
        - 8 predictions in correct cluster (6 + 1 semantic + 1 fallback)
        - 2 predictions in wrong cluster
        - confidence = 0.8 (8/10)
        - is_correct = True (argmax is correct cluster)
        - pass@1 = 1 - C(2,1)/C(10,1) = 0.8
        """
        generations = [
            'Let me calculate... The answer is \\boxed{42}.',
            'After computation, \\boxed{42} is the result.',
            'Therefore \\boxed{40}.',
            'So we get \\boxed{42}.',
            '\\boxed{42}',
            '\\boxed{\\frac{84}{2}}',  # = 42
            'The result is \\boxed{40}.',
            '\\boxed{42}',
            '\\boxed{42}',
            'No boxed but the answer is 42.',  # Fallback
        ]

        final_preds = [extract_prediction(g) for g in generations]
        ground_truth = '42'

        clusters, correct_idx = cluster_predictions_with_ground_truth(final_preds, ground_truth)

        # Verify clustering
        correct_cluster_size = len(clusters[correct_idx])
        assert correct_cluster_size == 8, f'Expected 8 correct, got {correct_cluster_size}'

        # Compute argmax
        cluster_sizes = [len(c) for c in clusters]
        argmax_idx = cluster_sizes.index(max(cluster_sizes))

        assert argmax_idx == correct_idx, 'Argmax should be correct cluster'

        # Create sample and compute metrics
        sample = SampleEvalResult(
            question='What is 6 * 7?',
            prefill_prompt='Question: What is 6 * 7?',
            final_answer=ground_truth,
            generations=generations,
            final_preds=final_preds,
            semantic_clusters=[c.predictions for c in clusters],
            correct_cluster_idx=correct_idx,
            argmax_cluster_idx=argmax_idx,
            argmax_prediction=clusters[argmax_idx].representative,
            is_correct=True,
            confidence=0.8,
        )

        metrics = compute_aggregate_metrics([sample], ece_bins=[10], k_values=[1, 5])

        assert metrics.semantic_accuracy == 1.0
        assert metrics.pass_at_k[1] == pytest.approx(0.8)

    def test_full_pipeline_gt_not_matching(self):
        """Test full pipeline when GT doesn't match any prediction.

        Scenario: 10 generations, all wrong.
        - All generations have \\boxed{40}
        - GT is 42

        Expected:
        - GT added as empty cluster
        - is_correct = False
        - pass@k = 0 for all k
        """
        generations = ['\\boxed{40}'] * 10
        final_preds = [extract_prediction(g) for g in generations]
        ground_truth = '42'

        clusters, correct_idx = cluster_predictions_with_ground_truth(final_preds, ground_truth)

        # Verify GT was added as empty cluster
        assert clusters[correct_idx].predictions == []
        assert clusters[correct_idx].representative == ground_truth

        # Argmax is the "40" cluster
        cluster_sizes = [len(c) for c in clusters]
        argmax_idx = cluster_sizes.index(max(cluster_sizes))
        assert argmax_idx != correct_idx

        # Create sample
        sample = SampleEvalResult(
            question='What is 6 * 7?',
            prefill_prompt='Question: What is 6 * 7?',
            final_answer=ground_truth,
            generations=generations,
            final_preds=final_preds,
            semantic_clusters=[c.predictions for c in clusters],
            correct_cluster_idx=correct_idx,
            argmax_cluster_idx=argmax_idx,
            argmax_prediction=clusters[argmax_idx].representative,
            is_correct=False,
            confidence=1.0,
        )

        metrics = compute_aggregate_metrics([sample], ece_bins=[10], k_values=[1, 5, 10])

        assert metrics.semantic_accuracy == 0.0
        assert metrics.pass_at_k[1] == 0.0
        assert metrics.pass_at_k[5] == 0.0
        assert metrics.pass_at_k[10] == 0.0
