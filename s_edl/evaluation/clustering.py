"""Semantic clustering for predictions using find-first algorithm.

Supports two equivalence backends:
- Math-verify (default): SymPy-based symbolic equivalence for pure math.
- LLM judge: vLLM-based listwise classification for general subjects.
"""
from collections.abc import Callable
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from .verify import are_semantically_equivalent, parse_math_expression

# A classifier takes (prediction, list_of_representatives) and returns the index
# of the matching representative, or None if no match.
ClusterClassifier = Callable[[str, list[str]], int | None]


def _sort_clusters_by_similarity(
    query: str,
    clusters: list['SemanticCluster'],
    parseable_only: bool = True,
) -> list[tuple[int, float]]:
    """Sort cluster indices by string similarity to query (descending).

    Args:
        query: String to compare against cluster representatives.
        clusters: List of SemanticCluster objects.
        parseable_only: If True, only include parseable clusters.

    Returns:
        List of (cluster_index, similarity_score) tuples, sorted by descending similarity.
    """
    similarities = [
        (i, fuzz.ratio(query, c.representative)) for i, c in enumerate(clusters)
        if not parseable_only or c.is_parseable
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities


@dataclass
class SemanticCluster:
    """A cluster of semantically equivalent predictions.

    Attributes:
        representative: The cluster representative (first prediction, or GT for GT-only clusters).
        predictions: List of all predictions in this cluster (may be empty for GT-only cluster).
        indices: List of indices into the original predictions list.
        is_parseable: Whether the representative is parseable by math-verify.
    """

    representative: str
    predictions: list[str] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)
    is_parseable: bool = True

    def __len__(self) -> int:
        """Return the number of predictions in this cluster."""
        return len(self.predictions)


def _math_verify_classify(prediction: str, representatives: list[str]) -> int | None:
    """Classify a prediction by pairwise math-verify equivalence.

    Args:
        prediction: The prediction to classify.
        representatives: Ordered list of cluster representatives.

    Returns:
        Index into representatives of the first match, or None.
    """
    for i, rep in enumerate(representatives):
        if are_semantically_equivalent(prediction, rep):
            return i
    return None


def cluster_predictions(
    predictions: list[str],
    classifier: ClusterClassifier | None = None,
) -> list[SemanticCluster]:
    """Cluster predictions by semantic equivalence using find-first algorithm.

    When ``classifier`` is None (default), uses math-verify for pairwise equivalence.
    When a classifier is provided, delegates equivalence checking to it (e.g. LLM judge).

    Algorithm:
    1. For each prediction, sort existing clusters by string similarity
    2. Check equivalence with cluster representatives (via classifier or math-verify)
    3. If match found, append to that cluster; otherwise create new cluster

    Time complexity: O(N x C) where N = number of predictions, C = number of clusters

    Args:
        predictions: List of prediction strings.
        classifier: Optional callable for equivalence checking.

    Returns:
        List of SemanticCluster objects.
    """
    clusters: list[SemanticCluster] = []
    use_llm = classifier is not None

    for idx, pred in enumerate(predictions):
        if not use_llm:
            # Math-verify mode: unparseable predictions become singleton clusters
            _, pred_parseable = parse_math_expression(pred)
            if not pred_parseable:
                clusters.append(
                    SemanticCluster(
                        representative=pred,
                        predictions=[pred],
                        indices=[idx],
                        is_parseable=False,
                    ))
                continue

        # Sort existing clusters by string similarity for better ordering
        # In LLM mode, include all clusters (parseable_only=False)
        sorted_clusters = _sort_clusters_by_similarity(pred, clusters, parseable_only=not use_llm)
        sorted_representatives = [clusters[ci].representative for ci, _ in sorted_clusters]

        if not sorted_representatives:
            # First iteration: no clusters yet, skip classifier call
            result = None
        elif use_llm:
            result = classifier(pred, sorted_representatives)
        else:
            result = _math_verify_classify(pred, sorted_representatives)

        if result is not None:
            matched_cluster_idx = sorted_clusters[result][0]
            clusters[matched_cluster_idx].predictions.append(pred)
            clusters[matched_cluster_idx].indices.append(idx)
        else:
            clusters.append(
                SemanticCluster(
                    representative=pred,
                    predictions=[pred],
                    indices=[idx],
                    is_parseable=True,
                ))

    return clusters


def cluster_predictions_with_ground_truth(
    predictions: list[str],
    ground_truth: str,
    classifier: ClusterClassifier | None = None,
) -> tuple[list[SemanticCluster], int]:
    """Cluster predictions and identify the correct cluster.

    If no cluster matches the ground truth, adds the ground truth as a new
    cluster with zero predictions (empty list).

    Args:
        predictions: List of prediction strings.
        ground_truth: The ground truth answer.
        classifier: Optional callable for equivalence checking.

    Returns:
        Tuple of (clusters, correct_cluster_idx).
        correct_cluster_idx is always valid (GT added as empty cluster if no match).
    """
    clusters = cluster_predictions(predictions, classifier=classifier)
    use_llm = classifier is not None

    # Find which cluster matches the ground truth (sorted by similarity)
    sorted_clusters = _sort_clusters_by_similarity(ground_truth, clusters, parseable_only=not use_llm)
    sorted_representatives = [clusters[ci].representative for ci, _ in sorted_clusters]

    if use_llm:
        result = classifier(ground_truth, sorted_representatives)
    else:
        result = _math_verify_classify(ground_truth, sorted_representatives)

    correct_cluster_idx = sorted_clusters[result][0] if result is not None else None

    # If no cluster matches, add ground truth as a GT-only cluster
    if correct_cluster_idx is None:
        if use_llm:
            gt_parseable = True
        else:
            _, gt_parseable = parse_math_expression(ground_truth)
        clusters.append(
            SemanticCluster(
                representative=ground_truth,
                predictions=[],  # no predictions match GT
                indices=[],
                is_parseable=gt_parseable,
            ))
        correct_cluster_idx = len(clusters) - 1

    return clusters, correct_cluster_idx
