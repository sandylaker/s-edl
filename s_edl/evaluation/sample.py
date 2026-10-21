"""Sample-level evaluation result dataclass and I/O utilities."""
import json
from dataclasses import asdict, dataclass


@dataclass
class SampleEvalResult:
    """Sample-level evaluation result with semantic clustering.

    Attributes:
        question: Raw question text.
        prefill_prompt: Formatted prompt for inference.
        final_answer: Ground truth answer.
        generations: N rollouts from the model (generated text only, no prefill).
        final_preds: Final predictions extracted from \\boxed{} or fallback.
        semantic_clusters: Predictions grouped by semantic equivalence.
            Each inner list contains predictions in that cluster.
            For GT-only cluster (when no prediction matches GT), the list is empty.
        correct_cluster_idx: Index of the cluster matching ground truth.
            Always valid; GT added as empty cluster if no prediction matches.
        argmax_cluster_idx: Index of the largest cluster.
        argmax_prediction: Representative of the largest cluster.
        is_correct: Whether the argmax prediction matches ground truth.
        confidence: Fraction of predictions in the argmax cluster (len(argmax_cluster) / N).
    """

    question: str
    prefill_prompt: str
    final_answer: str
    generations: list[str]
    final_preds: list[str]
    semantic_clusters: list[list[str]]
    correct_cluster_idx: int
    argmax_cluster_idx: int
    argmax_prediction: str
    is_correct: bool
    confidence: float


def save_eval_samples(samples: list[SampleEvalResult], path: str) -> None:
    """Save evaluation samples to a JSONL file.

    Args:
        samples: List of SampleEvalResult objects.
        path: Path to the output JSONL file.
    """
    with open(path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample), ensure_ascii=False) + '\n')


def load_eval_samples(path: str) -> list[SampleEvalResult]:
    """Load evaluation samples from a JSONL file.

    Args:
        path: Path to the input JSONL file.

    Returns:
        List of SampleEvalResult objects.
    """
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            samples.append(SampleEvalResult(**data))
    return samples
