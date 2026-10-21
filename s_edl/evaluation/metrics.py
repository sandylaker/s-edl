"""Aggregate metrics computation for evaluation results."""
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import relplot as rp

from .sample import SampleEvalResult


@dataclass
class AggregateMetrics:
    """Aggregate evaluation metrics across all samples.

    Attributes:
        semantic_accuracy: Accuracy of argmax(semantic_distribution).
        ece: Expected Calibration Error at different bin counts.
        smece: Smooth Expected Calibration Error (kernel-based).
        pass_at_k: pass@k scores for different k values.
        num_samples: Total number of evaluation samples.
        cluster_stats: Statistics about number of semantic clusters per sample.
    """

    semantic_accuracy: float
    ece: dict[int, float]
    smece: float
    pass_at_k: dict[int, float]
    num_samples: int
    cluster_stats: dict[str, dict[str, float]] | None = None


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k using the unbiased estimator.

    Formula: pass@k = 1 - C(n-c, k) / C(n, k)

    Args:
        n: Total number of predictions.
        c: Number of correct predictions.
        k: k value for pass@k.

    Returns:
        pass@k probability.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def compute_ece(samples: list[SampleEvalResult], n_bins: int) -> float:
    """Compute Expected Calibration Error.

    ECE = sum_b |accuracy_b - confidence_b| * (n_b / N)

    Args:
        samples: List of evaluation results.
        n_bins: Number of bins for calibration.

    Returns:
        ECE value.
    """
    if not samples:
        return 0.0

    # Initialize bins
    bin_correct = [0] * n_bins
    bin_confidence = [0.0] * n_bins
    bin_count = [0] * n_bins

    for sample in samples:
        # Determine which bin this sample falls into
        bin_idx = min(int(sample.confidence * n_bins), n_bins - 1)
        bin_count[bin_idx] += 1
        bin_correct[bin_idx] += int(sample.is_correct)
        bin_confidence[bin_idx] += sample.confidence

    # Compute ECE
    ece = 0.0
    total = len(samples)
    for i in range(n_bins):
        if bin_count[i] > 0:
            accuracy = bin_correct[i] / bin_count[i]
            avg_confidence = bin_confidence[i] / bin_count[i]
            ece += abs(accuracy - avg_confidence) * (bin_count[i] / total)

    return ece


def _get_calibration_data(samples: list[SampleEvalResult]) -> tuple[np.ndarray, np.ndarray]:
    """Extract calibration data from samples.

    Args:
        samples: List of evaluation results.

    Returns:
        Tuple of (confidences, correctness) arrays.
    """
    confidences = np.array([s.confidence for s in samples])
    correctness = np.array([int(s.is_correct) for s in samples])
    return confidences, correctness


def plot_cluster_distribution(
    samples: list[SampleEvalResult],
    output_path: str,
) -> dict[str, dict[str, float]]:
    """Plot distribution of semantic cluster counts with histogram and KDE overlay.

    Creates a 1x3 figure with histograms and KDE curves for:
    - All samples
    - Correct samples only
    - Incorrect samples only

    Args:
        samples: List of evaluation results.
        output_path: Path to save the figure.

    Returns:
        Dictionary with statistics (mean, std, min, max for each group).
    """
    from scipy.stats import gaussian_kde

    # Extract cluster counts
    all_counts = [len(s.semantic_clusters) for s in samples]
    correct_counts = [len(s.semantic_clusters) for s in samples if s.is_correct]
    incorrect_counts = [len(s.semantic_clusters) for s in samples if not s.is_correct]

    # Create 1x3 figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # Determine global max for consistent x-axis
    global_max = max(all_counts) if all_counts else 1
    x_range = np.linspace(1, global_max, 200)

    # Plot histograms with KDE overlay
    for ax, counts, title in zip(
        axes,
        [all_counts, correct_counts, incorrect_counts],
        ['All Samples', 'Correct Samples', 'Incorrect Samples'],
    ):
        if counts:
            # Plot histogram with density normalization
            ax.hist(
                counts,
                bins=range(1, global_max + 2),
                density=True,
                align='left',
                edgecolor='black',
                alpha=0.5,
            )
            # Overlay KDE if enough data points
            if len(counts) > 1:
                kde = gaussian_kde(counts, bw_method='scott')
                density = kde(x_range)
                ax.plot(x_range, density, color='red', linewidth=1.5)
            ax.set_xlabel('Number of Semantic Classes', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.set_title(f'{title} (n={len(counts)})', fontsize=10)
            ax.set_xlim(1, global_max)
            # Reduce x-axis tick clutter
            if global_max > 20:
                tick_step = 5
                ticks = list(range(1, global_max + 1, tick_step))
                if global_max not in ticks:
                    ticks.append(global_max)
                ax.set_xticks(ticks)
            ax.tick_params(axis='both', labelsize=8)
        else:
            ax.set_xlabel('Number of Semantic Classes', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.set_title(f'{title} (n=0)', fontsize=10)
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Return statistics
    def stats(counts: list[int]) -> dict[str, float]:
        if not counts:
            return {'mean': 0.0, 'std': 0.0, 'min': 0, 'max': 0}
        arr = np.array(counts)
        return {
            'mean': float(arr.mean()),
            'std': float(arr.std()),
            'min': int(arr.min()),
            'max': int(arr.max()),
        }

    return {
        'all': stats(all_counts),
        'correct': stats(correct_counts),
        'incorrect': stats(incorrect_counts),
    }


def compute_aggregate_metrics(
    samples: list[SampleEvalResult],
    ece_bins: list[int] | None = None,
    k_values: list[int] | None = None,
    calibration_plot_path: str | None = None,
    cluster_dist_plot_path: str | None = None,
) -> AggregateMetrics:
    """Compute aggregate metrics across all samples.

    Args:
        samples: List of evaluation results.
        ece_bins: List of bin counts for ECE computation. Defaults to [10, 15, 20].
        k_values: List of k values for pass@k. Defaults to [1, 5, 10].
        calibration_plot_path: If provided, save calibration curve to this path.
        cluster_dist_plot_path: If provided, save cluster distribution plot to this path.

    Returns:
        AggregateMetrics with overall scores.
    """
    if ece_bins is None:
        ece_bins = [10, 15, 20]
    if k_values is None:
        k_values = [1, 5, 10]

    num_samples = len(samples)
    if num_samples == 0:
        return AggregateMetrics(
            semantic_accuracy=0.0,
            ece={b: 0.0
                 for b in ece_bins},
            smece=0.0,
            pass_at_k={k: 0.0
                       for k in k_values},
            num_samples=0,
        )

    # Semantic accuracy
    semantic_correct = sum(1 for s in samples if s.is_correct)
    semantic_accuracy = semantic_correct / num_samples

    # ECE at different bin counts
    ece = {b: compute_ece(samples, b) for b in ece_bins}

    # Smooth ECE and optional calibration curve plot
    confidences, correctness = _get_calibration_data(samples)
    smece = float(rp.smECE_sigma(confidences, correctness, sigma=0.05))

    if calibration_plot_path is not None:
        # relplot displays smECE in the plot with auto-selected sigma, so it can differ from
        # the smECE above, where we computed via fixed sigma=0.05 in `rp.smECE_sigma`.
        fig, ax = rp.rel_diagram(confidences, correctness, kde_bandwidth=0.05, plot_density=True)
        ax.set_xlabel('Semantic Confidence')
        ax.set_ylabel('Semantic Accuracy')
        fig.savefig(calibration_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Pass@k
    pass_at_k = {}
    for k in k_values:
        pass_at_k_values = []
        for sample in samples:
            n = len(sample.final_preds)
            # Count correct predictions: those in the correct cluster
            c = len(sample.semantic_clusters[sample.correct_cluster_idx])
            pass_at_k_values.append(compute_pass_at_k(n, c, k))
        pass_at_k[k] = sum(pass_at_k_values) / num_samples

    # Cluster distribution statistics and optional plot
    cluster_stats = None
    if cluster_dist_plot_path is not None:
        cluster_stats = plot_cluster_distribution(samples, cluster_dist_plot_path)

    return AggregateMetrics(
        semantic_accuracy=semantic_accuracy,
        ece=ece,
        smece=smece,
        pass_at_k=pass_at_k,
        num_samples=num_samples,
        cluster_stats=cluster_stats,
    )
