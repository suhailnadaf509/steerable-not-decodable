"""
Publication-Quality Figure Generation
======================================
Generates all figures for the paper from saved results.
Can run as a standalone stage without GPU.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning("matplotlib not available -- figure generation disabled")


def _save_fig(fig, path: Path, dpi: int = 300):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s", path)


# Task ordering by computational category (for consistent figure layouts)
TASK_ORDER = [
    "antonym", "synonym", "hypernym",           # Lexical Retrieval
    "country_capital", "english_spanish", "object_color",  # Factual Retrieval
    "past_tense", "plural",                      # Morphological
    "capitalize", "first_letter", "reverse_word", # Character/Surface
    "sentiment_flip",                             # Compositional
]

CATEGORY_LABELS = {
    "antonym": "Lex.", "synonym": "Lex.", "hypernym": "Lex.",
    "country_capital": "Fact.", "english_spanish": "Fact.", "object_color": "Fact.",
    "past_tense": "Morph.", "plural": "Morph.",
    "capitalize": "Char.", "first_letter": "Char.", "reverse_word": "Char.",
    "sentiment_flip": "Comp.",
}

CATEGORY_COLORS = {
    "Lex.": "#2196F3", "Fact.": "#4CAF50", "Morph.": "#FF9800",
    "Char.": "#F44336", "Comp.": "#9C27B0",
}


def _ordered_tasks(available: set) -> List[str]:
    """Return tasks in canonical order, filtering to what's available."""
    return [t for t in TASK_ORDER if t in available]


# ---------------------------------------------------------------------------
# IID Summary Table Figure
# ---------------------------------------------------------------------------
def plot_iid_summary(iid_data: List[Dict], output_dir: Path):
    """Bar chart of IID accuracy by task, with threshold line."""
    if not HAS_MPL:
        return

    by_task = {}
    for entry in iid_data:
        task = entry["task"]
        by_task.setdefault(task, []).append(entry["best_accuracy"])

    tasks = _ordered_tasks(set(by_task.keys()))
    means = [np.mean(by_task[t]) for t in tasks]
    maxes = [max(by_task[t]) for t in tasks]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(tasks))
    width = 0.35

    ax.bar(x - width / 2, means, width, label="Mean IID Acc", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, maxes, width, label="Max IID Acc", color="darkorange", alpha=0.8)
    ax.axhline(0.10, color="red", linestyle="--", alpha=0.6, label="IID Threshold (0.10)")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("IID Steering Accuracy by Task")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    _save_fig(fig, output_dir / "iid_summary.png")


# ---------------------------------------------------------------------------
# Alignment vs Transfer Scatter
# ---------------------------------------------------------------------------
def plot_alignment_vs_transfer(
    alignments: List[Dict],
    steering_results: Dict,
    output_dir: Path,
):
    """Scatter plot of cosine similarity vs transfer accuracy, colored by task."""
    if not HAS_MPL:
        return

    tasks_colors = {}
    color_cycle = plt.cm.tab10.colors

    cosines, accs, colors, task_labels = [], [], [], []

    for ar in alignments:
        task = ar["task"]
        if task not in tasks_colors:
            tasks_colors[task] = color_cycle[len(tasks_colors) % len(color_cycle)]

        try:
            acc_ab = steering_results[task][ar["template_a"]][ar["template_b"]][ar["layer"]]
            acc_ba = steering_results[task][ar["template_b"]][ar["template_a"]][ar["layer"]]
        except (KeyError, TypeError):
            continue

        for acc in [acc_ab, acc_ba]:
            cosines.append(ar["cosine"])
            accs.append(acc)
            colors.append(tasks_colors[task])
            task_labels.append(task)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(cosines, accs, c=colors, alpha=0.3, s=10)

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=c, label=t) for t, c in tasks_colors.items()
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Transfer Accuracy")
    ax.set_title("Alignment vs Transfer (colored by task)")
    ax.axhline(0.40, color="red", linestyle="--", alpha=0.3, label="Low transfer threshold")
    ax.axvline(0.80, color="blue", linestyle="--", alpha=0.3, label="High alignment threshold")
    ax.grid(alpha=0.2)

    _save_fig(fig, output_dir / "alignment_vs_transfer.png")


# ---------------------------------------------------------------------------
# Readability vs Steering (Figure 6 — the paper's key figure)
# ---------------------------------------------------------------------------
def plot_readability_vs_steering(
    readability_results: List[Dict],
    iid_data: List[Dict],
    output_dir: Path,
    sentiment_results: Optional[List[Dict]] = None,
):
    """
    Paired bar chart: logit-lens readability vs FV steering accuracy.

    This is Figure 6 in the paper — the visual representation of the
    readability-steerability gap.  The GAP between the two bars for each
    task IS the paper's central finding.
    """
    if not HAS_MPL:
        return

    # Best readability (top-10 accuracy) per task
    read_by_task: Dict[str, List[float]] = {}
    for r in readability_results:
        read_by_task.setdefault(r["task"], []).append(r["top_10_accuracy"])

    # Incorporate sentiment polarity if available
    if sentiment_results:
        for sr in sentiment_results:
            read_by_task.setdefault(sr["task"], []).append(
                sr["polarity_classification_accuracy"]
            )

    # Best steering per task
    steer_by_task: Dict[str, List[float]] = {}
    for entry in iid_data:
        steer_by_task.setdefault(entry["task"], []).append(entry["best_accuracy"])

    all_tasks = set(read_by_task.keys()) | set(steer_by_task.keys())
    tasks = _ordered_tasks(all_tasks)
    if not tasks:
        return

    read_maxes = [max(read_by_task.get(t, [0.0])) for t in tasks]
    steer_maxes = [max(steer_by_task.get(t, [0.0])) for t in tasks]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(tasks))
    width = 0.35

    bars_read = ax.bar(
        x - width / 2, read_maxes, width,
        label="Readability (logit lens top-10)", color="forestgreen", alpha=0.85,
    )
    bars_steer = ax.bar(
        x + width / 2, steer_maxes, width,
        label="Steerability (FV best config)", color="steelblue", alpha=0.85,
    )

    # Annotate the gap above each pair
    for i in range(len(tasks)):
        gap = read_maxes[i] - steer_maxes[i]
        if abs(gap) > 0.05:
            y_pos = max(read_maxes[i], steer_maxes[i]) + 0.02
            color = "green" if gap > 0 else "red"
            ax.annotate(
                f"{'+'if gap > 0 else ''}{gap:.2f}",
                xy=(x[i], y_pos), ha="center", va="bottom",
                fontsize=7, color=color, fontweight="bold",
            )

    # Category separators
    cat_boundaries = []
    prev_cat = None
    for i, t in enumerate(tasks):
        cat = CATEGORY_LABELS.get(t, "")
        if cat != prev_cat and prev_cat is not None:
            cat_boundaries.append(i - 0.5)
        prev_cat = cat
    for b in cat_boundaries:
        ax.axvline(b, color="gray", linestyle=":", alpha=0.4)

    ax.axhline(0.10, color="red", linestyle="--", alpha=0.4, label="Threshold (0.10)")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_title("Readability (Logit Lens) vs Steerability (FV Steering)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, min(max(max(read_maxes), max(steer_maxes)) + 0.15, 1.05))
    ax.grid(axis="y", alpha=0.3)

    _save_fig(fig, output_dir / "readability_vs_steering.png")


# ---------------------------------------------------------------------------
# Layer-Wise Readability vs Steering Profiles (Figure 7)
# ---------------------------------------------------------------------------
def plot_layer_profiles(
    readability_results: List[Dict],
    iid_data: List[Dict],
    output_dir: Path,
    representative_tasks: Optional[List[str]] = None,
):
    """
    Layer-wise dual curves: logit-lens readability and FV steering at each
    extraction layer, for representative tasks.

    Shows WHERE in the network readability and steerability diverge.
    """
    if not HAS_MPL:
        return

    # Build per-(task, layer) readability: max top-10 across templates
    read_by_task_layer: Dict[str, Dict[int, List[float]]] = {}
    for r in readability_results:
        read_by_task_layer.setdefault(r["task"], {}).setdefault(
            r["layer"], []
        ).append(r["top_10_accuracy"])

    # Build per-(task, layer) steering: max IID accuracy across templates
    # IID data has best_accuracy and best_layer per (task, template)
    # We need per-layer steering. Use iid_data which has best_layer.
    # For layer-wise, we need the steering_results with per-layer data.
    # Fallback: just plot the best-layer steer as a horizontal line.
    steer_by_task: Dict[str, float] = {}
    for entry in iid_data:
        task = entry["task"]
        steer_by_task[task] = max(steer_by_task.get(task, 0.0), entry["best_accuracy"])

    # Select representative tasks: pick from each 2×2 cell if possible
    available = set(read_by_task_layer.keys())
    if representative_tasks:
        tasks_to_plot = [t for t in representative_tasks if t in available]
    else:
        # Default: 6 representative tasks spanning the taxonomy
        candidates = ["antonym", "country_capital", "past_tense",
                       "first_letter", "sentiment_flip", "reverse_word"]
        tasks_to_plot = [t for t in candidates if t in available]

    if not tasks_to_plot:
        return

    n_tasks = len(tasks_to_plot)
    ncols = min(3, n_tasks)
    nrows = (n_tasks + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                              squeeze=False)

    for idx, task in enumerate(tasks_to_plot):
        row, col = idx // ncols, idx % ncols
        ax = axes[row][col]

        layer_data = read_by_task_layer[task]
        layers_sorted = sorted(layer_data.keys())
        read_means = [np.mean(layer_data[l]) for l in layers_sorted]
        read_stds = [np.std(layer_data[l]) for l in layers_sorted]

        ax.plot(layers_sorted, read_means, "g-o", markersize=3,
                label="Readability (logit lens)", linewidth=1.5)
        ax.fill_between(
            layers_sorted,
            [m - s for m, s in zip(read_means, read_stds)],
            [m + s for m, s in zip(read_means, read_stds)],
            alpha=0.15, color="green",
        )

        # Steering as horizontal dashed line (best across all layers/templates)
        best_steer = steer_by_task.get(task, 0.0)
        ax.axhline(best_steer, color="steelblue", linestyle="--", alpha=0.7,
                    label=f"Best FV steer ({best_steer:.2f})")

        ax.set_title(task, fontsize=10, fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.2)

    # Hide unused subplots
    for idx in range(n_tasks, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row][col].set_visible(False)

    fig.suptitle("Layer-Wise Readability vs Steerability", fontsize=12, y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_dir / "layer_profiles.png")


# ---------------------------------------------------------------------------
# FV Vocabulary Projection Summary (Table figure)
# ---------------------------------------------------------------------------
def plot_fv_vocab_summary(
    vocab_results: List[Dict],
    output_dir: Path,
):
    """
    Bar chart showing correct-output fraction and task-relevant fraction
    of FV vocabulary projections, grouped by task.
    """
    if not HAS_MPL:
        return

    if not vocab_results:
        return

    # Best (highest correct_output_fraction) per task
    by_task: Dict[str, Dict] = {}
    for vr in vocab_results:
        task = vr["task"]
        if task not in by_task or vr["correct_output_fraction"] > by_task[task]["correct_output_fraction"]:
            by_task[task] = vr

    tasks = _ordered_tasks(set(by_task.keys()))
    if not tasks:
        return

    correct_fracs = [by_task[t]["correct_output_fraction"] for t in tasks]
    relevant_fracs = [by_task[t]["task_relevant_fraction"] for t in tasks]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(tasks))
    width = 0.35

    ax.bar(x - width / 2, correct_fracs, width,
           label="Correct output tokens in top-50", color="forestgreen", alpha=0.8)
    ax.bar(x + width / 2, relevant_fracs, width,
           label="Task-relevant tokens in top-50", color="goldenrod", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.set_ylabel("Fraction of top-50 tokens")
    ax.set_title("FV Vocabulary Projection: What Do Steering Vectors Encode?")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    _save_fig(fig, output_dir / "fv_vocab_projection.png")


# ---------------------------------------------------------------------------
# Steering Harm Analysis Figure
# ---------------------------------------------------------------------------
def plot_steering_harm(
    harm_data: Dict[str, Any],
    output_dir: Path,
):
    """
    Grouped bar chart: zero-shot vs few-shot vs FV steering per task.

    Highlights destructive cases where steering drops below zero-shot.
    This figure directly addresses the safety question: "Is the FV actively
    making the model worse?"
    """
    if not HAS_MPL:
        return

    per_task = harm_data.get("per_task", {})
    if not per_task:
        return

    tasks = _ordered_tasks(set(per_task.keys()))
    if not tasks:
        return

    zs = [per_task[t]["mean_zero_shot"] for t in tasks]
    fs = [per_task[t]["mean_few_shot"] for t in tasks]
    st = [per_task[t]["mean_steering"] for t in tasks]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(tasks))
    width = 0.25

    ax.bar(x - width, zs, width, label="Zero-shot", color="#90CAF9", alpha=0.9)
    ax.bar(x, fs, width, label="Few-shot ICL", color="#42A5F5", alpha=0.9)
    ax.bar(x + width, st, width, label="FV steering", color="#E57373", alpha=0.9)

    # Mark destructive cases with a red triangle below the steering bar
    for i, t in enumerate(tasks):
        delta = per_task[t]["mean_delta"]
        if delta < -0.01:
            ax.annotate(
                f"{delta:+.2f}",
                xy=(x[i] + width, st[i]),
                xytext=(x[i] + width, max(st[i] - 0.06, -0.04)),
                ha="center", va="top", fontsize=7, color="darkred",
                fontweight="bold",
            )

    # Category separators
    cat_boundaries = []
    prev_cat = None
    for i, t in enumerate(tasks):
        cat = CATEGORY_LABELS.get(t, "")
        if cat != prev_cat and prev_cat is not None:
            cat_boundaries.append(i - 0.5)
        prev_cat = cat
    for b in cat_boundaries:
        ax.axvline(b, color="gray", linestyle=":", alpha=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_title("Steering Harm: Zero-Shot vs Few-Shot ICL vs FV Steering")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(-0.02, min(max(max(zs), max(fs), max(st)) + 0.1, 1.05))
    ax.grid(axis="y", alpha=0.3)

    agg = harm_data.get("aggregate", {})
    n_dest = agg.get("total_destructive", 0)
    rate = agg.get("destructive_rate", 0)
    ax.text(
        0.02, 0.97,
        f"Destructive cases: {n_dest} ({rate:.0%})",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    _save_fig(fig, output_dir / "steering_harm.png")


# ---------------------------------------------------------------------------
# Legacy: Probe vs Steering (kept for backward compatibility with old results)
# ---------------------------------------------------------------------------
def plot_probe_vs_steering(
    probe_results: List[Dict],
    iid_data: List[Dict],
    output_dir: Path,
):
    """Compare probe accuracy vs steering accuracy by task (legacy)."""
    if not HAS_MPL:
        return

    # Aggregate by task
    probe_by_task = {}
    for pr in probe_results:
        probe_by_task.setdefault(pr["task"], []).append(pr["accuracy"])

    steer_by_task = {}
    for entry in iid_data:
        steer_by_task.setdefault(entry["task"], []).append(entry["best_accuracy"])

    tasks = sorted(set(probe_by_task.keys()) & set(steer_by_task.keys()))
    if not tasks:
        return

    probe_maxes = [max(probe_by_task[t]) for t in tasks]
    steer_maxes = [max(steer_by_task[t]) for t in tasks]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(tasks))
    width = 0.35

    ax.bar(x - width / 2, probe_maxes, width, label="Probe (best layer)", color="forestgreen", alpha=0.8)
    ax.bar(x + width / 2, steer_maxes, width, label="Steering (best config)", color="steelblue", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Linear Probing vs Additive Steering by Task")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    _save_fig(fig, output_dir / "probe_vs_steering.png")


# ---------------------------------------------------------------------------
# 3-Way Comparison: Logit Lens vs Tuned Lens vs FV Steering (Figure 8)
# ---------------------------------------------------------------------------
def plot_tuned_lens_comparison(
    logit_lens_results: List[Dict],
    tuned_lens_results: List[Dict],
    iid_data: List[Dict],
    output_dir: Path,
    threshold: float = 0.10,
):
    """
    Grouped bar chart comparing logit lens, tuned lens, and FV steering
    accuracy per task — the visual representation of the expanded 2×3 matrix.

    The gap between tuned-lens and logit-lens bars shows the representational
    dialect correction.  The gap between tuned-lens and steering bars shows
    whether steerability-without-decodability survives the better decoder.
    """
    if not HAS_MPL:
        return

    # Best top-10 per task for logit lens
    ll_by_task: Dict[str, float] = {}
    for r in logit_lens_results:
        task = r["task"]
        ll_by_task[task] = max(ll_by_task.get(task, 0.0), r["top_10_accuracy"])

    # Best top-10 per task for tuned lens
    tl_by_task: Dict[str, float] = {}
    for r in tuned_lens_results:
        task = r["task"]
        tl_by_task[task] = max(tl_by_task.get(task, 0.0), r["top_10_accuracy"])

    # Best steering per task
    steer_by_task: Dict[str, float] = {}
    for entry in iid_data:
        task = entry["task"]
        steer_by_task[task] = max(
            steer_by_task.get(task, 0.0), entry["best_accuracy"],
        )

    all_tasks = set(ll_by_task.keys()) | set(tl_by_task.keys()) | set(steer_by_task.keys())
    tasks = _ordered_tasks(all_tasks)
    if not tasks:
        return

    ll_vals = [ll_by_task.get(t, 0.0) for t in tasks]
    tl_vals = [tl_by_task.get(t, 0.0) for t in tasks]
    steer_vals = [steer_by_task.get(t, 0.0) for t in tasks]

    fig, ax = plt.subplots(figsize=(15, 6))
    x = np.arange(len(tasks))
    width = 0.25

    ax.bar(
        x - width, ll_vals, width,
        label="Logit Lens (top-10)", color="forestgreen", alpha=0.85,
    )
    ax.bar(
        x, tl_vals, width,
        label="Tuned Lens (top-10)", color="darkorange", alpha=0.85,
    )
    ax.bar(
        x + width, steer_vals, width,
        label="FV Steering (best config)", color="steelblue", alpha=0.85,
    )

    # Threshold line
    ax.axhline(
        threshold, color="red", linestyle="--", alpha=0.4,
        label=f"Threshold ({threshold})",
    )

    # Category separators
    cat_boundaries = []
    prev_cat = None
    for i, t in enumerate(tasks):
        cat = CATEGORY_LABELS.get(t, "")
        if cat != prev_cat and prev_cat is not None:
            cat_boundaries.append(i - 0.5)
        prev_cat = cat
    for b in cat_boundaries:
        ax.axvline(b, color="gray", linestyle=":", alpha=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_title("Logit Lens vs Tuned Lens vs FV Steering")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, min(
        max(max(ll_vals), max(tl_vals), max(steer_vals)) + 0.15, 1.05,
    ))
    ax.grid(axis="y", alpha=0.3)

    _save_fig(fig, output_dir / "tuned_lens_comparison.png")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def generate_all_figures(
    results_dir: Path,
    figures_dir: Path,
):
    """Generate all figures from saved results."""
    if not HAS_MPL:
        logger.warning("matplotlib not available -- skipping figures")
        return

    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load IID summary
    iid_path = results_dir / "iid_summary.json"
    iid_data = None
    if iid_path.exists():
        with open(iid_path) as f:
            iid_data = json.load(f)
        plot_iid_summary(iid_data, figures_dir)

    # Load geometric analysis
    geo_path = results_dir / "geometric_analysis.json"
    steering_path = results_dir / "steering_results.json"

    if geo_path.exists() and steering_path.exists():
        with open(geo_path) as f:
            geo = json.load(f)
        with open(steering_path) as f:
            raw_steering = json.load(f)

        # Convert keys
        def _to_int_keys(d, depth=0):
            if isinstance(d, dict):
                new = {}
                for k, v in d.items():
                    try:
                        k = int(k)
                    except (ValueError, TypeError):
                        pass
                    new[k] = _to_int_keys(v, depth + 1)
                return new
            return d

        steering = _to_int_keys(raw_steering)
        alignments = geo.get("alignments", [])
        if alignments:
            plot_alignment_vs_transfer(alignments, steering, figures_dir)

    # Readability vs Steering (new — replaces broken probe figure)
    readability_path = results_dir / "readability_results.json"
    if readability_path.exists() and iid_data is not None:
        with open(readability_path) as f:
            readability_data = json.load(f)

        read_results = readability_data.get("readability", [])
        sentiment_results = readability_data.get("sentiment_polarity", [])
        vocab_results = readability_data.get("fv_vocab_projection", [])

        if read_results:
            plot_readability_vs_steering(
                read_results, iid_data, figures_dir,
                sentiment_results=sentiment_results or None,
            )
            plot_layer_profiles(read_results, iid_data, figures_dir)

        if vocab_results:
            plot_fv_vocab_summary(vocab_results, figures_dir)

        # Steering harm analysis
        harm_data = readability_data.get("steering_harm")
        if harm_data:
            plot_steering_harm(harm_data, figures_dir)

    # Tuned Lens 3-way comparison (if tuned lens results exist)
    tuned_lens_path = results_dir / "tuned_lens_results.json"
    if tuned_lens_path.exists() and iid_data is not None:
        with open(tuned_lens_path) as f:
            tuned_lens_data = json.load(f)
        tl_read_results = tuned_lens_data.get("readability", [])

        if tl_read_results and readability_path.exists():
            with open(readability_path) as f:
                rd = json.load(f)
            ll_read_results = rd.get("readability", [])
            if ll_read_results:
                plot_tuned_lens_comparison(
                    ll_read_results, tl_read_results, iid_data, figures_dir,
                )

    # Legacy: probe vs steering (only if old mechanistic results exist
    # AND new readability results don't)
    if not readability_path.exists():
        mech_path = results_dir / "mechanistic_results.json"
        if mech_path.exists() and iid_data is not None:
            with open(mech_path) as f:
                mech = json.load(f)
            probe_results = mech.get("probing", [])
            if probe_results:
                plot_probe_vs_steering(probe_results, iid_data, figures_dir)

    logger.info("Figure generation complete. Output: %s", figures_dir)
