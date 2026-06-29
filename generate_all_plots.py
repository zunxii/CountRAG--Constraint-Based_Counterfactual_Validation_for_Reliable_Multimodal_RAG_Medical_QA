
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------
def set_publication_style() -> None:
    """Set a clean, paper-friendly Matplotlib style."""
    mpl.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# -----------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------
def _candidate_roots() -> List[Path]:
    cwd = Path.cwd().resolve()
    here = Path(__file__).resolve().parent if "__file__" in globals() else cwd
    roots = [cwd, cwd / "test", here, here / "test", cwd.parent, here.parent]
    out: List[Path] = []
    seen = set()
    for r in roots:
        try:
            r = r.resolve()
        except Exception:
            continue
        if r.exists() and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def locate_file(rel_candidates: Sequence[str]) -> Path:
    for root in _candidate_roots():
        for rel in rel_candidates:
            p = root / rel
            if p.exists():
                return p
    raise FileNotFoundError(
        "Could not locate any of the following paths:\n"
        + "\n".join(f"  - {p}" for p in rel_candidates)
    )


def root_from_any(found_path: Path) -> Path:
    # For .../outputs/evaluation/.../results.json => repo root is parents[3]
    # parents: [folder, subfolder, outputs, repo_root, ...]
    try:
        return found_path.resolve().parents[3]
    except Exception:
        return found_path.resolve().parent


def load_json_auto(rel_candidates: Sequence[str]) -> Tuple[Dict[str, Any], Path]:
    path = locate_file(rel_candidates)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f), path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    ensure_dir(out_dir)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ saved {stem}")


def annotate_bars(ax: plt.Axes, bars: Iterable[mpl.patches.Patch], fmt: str = "{:.3f}", dy: float = 0.01) -> None:
    for bar in bars:
        h = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2
        ax.text(x, h + dy, fmt.format(h), ha="center", va="bottom", fontsize=9)


def heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    *,
    cmap: str = "Blues",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    fmt: str = "{:.2f}",
    annotate: bool = True,
    cbar_label: str = "",
) -> None:
    if vmin is None:
        vmin = float(np.nanmin(data))
    if vmax is None:
        vmax = float(np.nanmax(data))
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, pad=10, fontweight="bold")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=0)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    ax.tick_params(axis="both", which="both", length=0)
    ax.set_xlim(-0.5, len(col_labels) - 0.5)
    ax.set_ylim(len(row_labels) - 0.5, -0.5)

    if annotate:
        threshold = (vmin + vmax) / 2.0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isnan(val):
                    continue
                color = "white" if val > threshold else "black"
                ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=8, color=color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    if cbar_label:
        cbar.set_label(cbar_label, rotation=90)
    cbar.ax.tick_params(labelsize=9)


def _sorted_diag_keys_by_ranking(d: Dict[str, Any], ranking: Optional[List[str]] = None) -> List[str]:
    if ranking:
        return [k for k in ranking if k in d]
    return sorted(d.keys())


# -----------------------------------------------------------------------------
# Data loaders
# -----------------------------------------------------------------------------
def load_training_history() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Path, Path]:
    lora, lora_p = load_json_auto(
        [
            "outputs/models/trained_lora/training_history.json",
            "test/outputs/models/trained_lora/training_history.json",
        ]
    )
    fusion, fusion_p = load_json_auto(
        [
            "outputs/models/trained_fusion/training_history.json",
            "test/outputs/models/trained_fusion/training_history.json",
        ]
    )
    return lora, fusion, lora_p, fusion_p


def load_retrieval_results() -> Tuple[Dict[str, Any], Dict[str, Any], Path]:
    results, path = load_json_auto(
        [
            "outputs/evaluation/retrieval/results.json",
            "test/outputs/evaluation/retrieval/results.json",
        ]
    )
    per_diag, _ = load_json_auto(
        [
            "outputs/evaluation/retrieval/per_diagnosis_metrics.json",
            "test/outputs/evaluation/retrieval/per_diagnosis_metrics.json",
        ]
    )
    return results, per_diag, path


def load_counterfactual_results() -> Tuple[Dict[str, Any], Dict[str, Any], Path]:
    # Prefer the full/final run, then fall back to the first counterfactual run.
    results, path = load_json_auto(
        [
            "outputs/evaluation/counterfactual_final/results.json",
            "test/outputs/evaluation/counterfactual_final/results.json",
            "outputs/evaluation/counterfactual/results.json",
            "test/outputs/evaluation/counterfactual/results.json",
        ]
    )
    per_diag_candidates = [
        "outputs/evaluation/counterfactual_final/per_diagnosis_metrics.json",
        "test/outputs/evaluation/counterfactual_final/per_diagnosis_metrics.json",
        "outputs/evaluation/counterfactual/per_diagnosis_stability.json",
        "test/outputs/evaluation/counterfactual/per_diagnosis_stability.json",
    ]
    per_diag, _ = load_json_auto(per_diag_candidates)
    return results, per_diag, path


# -----------------------------------------------------------------------------
# Figure 1: training curves
# -----------------------------------------------------------------------------
def plot_training_curves(out_dir: Path) -> None:
    try:
        lora_hist, fusion_hist, _, _ = load_training_history()
    except FileNotFoundError:
        print("⚠ training history not found; skipping training curves")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)

    # LoRA
    epochs = [x["epoch"] for x in lora_hist]
    train_loss = [x["train_loss"] for x in lora_hist]
    val_loss = [x["val_loss"] for x in lora_hist]
    train_ctr = [x.get("train_contrastive", np.nan) for x in lora_hist]
    val_ctr = [x.get("val_contrastive", np.nan) for x in lora_hist]

    ax = axes[0]
    ax.plot(epochs, train_loss, marker="o", lw=2.2, color="#4C72B0", label="Train loss")
    ax.plot(epochs, val_loss, marker="s", lw=2.2, color="#C44E52", label="Val loss")
    ax.plot(epochs, train_ctr, marker="^", lw=1.8, color="#55A868", alpha=0.75, linestyle="--", label="Train contrastive")
    ax.plot(epochs, val_ctr, marker="D", lw=1.8, color="#8172B3", alpha=0.75, linestyle="--", label="Val contrastive")
    ax.set_title("LoRA adaptation", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xticks(epochs[:: max(1, len(epochs)//6)])
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, axis="y")

    # Fusion
    epochs = [x["epoch"] for x in fusion_hist]
    train_loss = [x["train_loss"] for x in fusion_hist]
    val_loss = [x["val_loss"] for x in fusion_hist]

    ax = axes[1]
    ax.plot(epochs, train_loss, marker="o", lw=2.2, color="#4C72B0", label="Train loss")
    ax.plot(epochs, val_loss, marker="s", lw=2.2, color="#C44E52", label="Val loss")
    ax.set_title("Adaptive fusion", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xticks(epochs[:: max(1, len(epochs)//6)])
    ax.legend(frameon=False)
    ax.grid(True, axis="y")

    save_figure(fig, out_dir, "fig_training_curves")


# -----------------------------------------------------------------------------
# Figure 2: retrieval heatmap
# -----------------------------------------------------------------------------
def plot_retrieval_summary_heatmap(out_dir: Path) -> None:
    try:
        results, _, _ = load_retrieval_results()
    except FileNotFoundError:
        print("⚠ retrieval results not found; skipping retrieval summary heatmap")
        return

    runs = results["runs"]
    config_order = [
        ("base_flat", "text"),
        ("base_flat", "image"),
        ("base_flat", "fusion"),
        ("base_concept", "text"),
        ("base_concept", "image"),
        ("base_concept", "fusion"),
        ("lora_concept", "text"),
        ("lora_concept", "image"),
        ("lora_concept", "fusion"),
    ]
    row_labels = []
    rows = []
    for run_name, mode_name in config_order:
        r = runs[run_name][mode_name]["metrics"]
        row_labels.append(f"{run_name.replace('_', ' ').title()} / {mode_name.title()}")
        rows.append([r["R@1"], r["R@5"], r["R@10"], r["MRR"], r["MAP"], r["NDCG@10"]])

    data = np.array(rows, dtype=float)
    fig, ax = plt.subplots(figsize=(12.8, 6.2), constrained_layout=True)
    heatmap(
        ax,
        data,
        row_labels,
        ["R@1", "R@5", "R@10", "MRR", "MAP", "NDCG@10"],
        "Retrieval performance across configurations",
        cmap="Blues",
        vmin=0.35,
        vmax=0.90,
        cbar_label="Score",
    )
    save_figure(fig, out_dir, "fig_retrieval_summary_heatmap")


# -----------------------------------------------------------------------------
# Figure 3: recall curves
# -----------------------------------------------------------------------------
def plot_recall_curves(out_dir: Path) -> None:
    try:
        results, _, _ = load_retrieval_results()
    except FileNotFoundError:
        print("⚠ retrieval results not found; skipping recall curves")
        return

    k_values = [1, 5, 10, 20]
    colors = {
        "text": "#4C72B0",
        "image": "#C44E52",
        "fusion": "#55A868",
    }
    labels = {"text": "Text", "image": "Image", "fusion": "Fusion"}

    fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
    for mode in ["text", "image", "fusion"]:
        recalls = [results["runs"]["lora_concept"][mode]["metrics"][f"R@{k}"] for k in k_values]
        ax.plot(
            k_values,
            recalls,
            marker={"text": "o", "image": "s", "fusion": "^"}[mode],
            lw=2.4,
            color=colors[mode],
            label=labels[mode],
        )
        for x, y in zip(k_values, recalls):
            ax.text(x, y + 0.01, f"{y:.3f}", ha="center", va="bottom", fontsize=8, color=colors[mode])

    ax.set_title("Recall curves on the CLIPSyntel holdout", fontweight="bold")
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K")
    ax.set_xticks(k_values)
    ax.set_ylim(0.55, 1.02)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, axis="y")
    save_figure(fig, out_dir, "fig_recall_curves")


# -----------------------------------------------------------------------------
# Figure 4: per-diagnosis retrieval heatmap (final model)
# -----------------------------------------------------------------------------
def plot_per_diagnosis_retrieval_heatmap(out_dir: Path) -> None:
    try:
        _, per_diag, _ = load_retrieval_results()
    except FileNotFoundError:
        print("⚠ retrieval per-diagnosis metrics not found; skipping diagnosis retrieval heatmap")
        return

    key = "lora_concept_fusion" if "lora_concept_fusion" in per_diag else next(iter(per_diag.keys()))
    diag_map = per_diag[key]

    # Sort by R@1 descending, tie-break by count if available elsewhere.
    rows = sorted(diag_map.items(), key=lambda kv: (kv[1].get("R@1", 0.0), kv[1].get("MRR", 0.0)), reverse=True)
    row_labels = [d.replace("_", " ").title() for d, _ in rows]
    data = np.array([[m["R@1"], m["MRR"], m["MAP"], m["NDCG@10"]] for _, m in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(10.6, max(5.5, 0.34 * len(rows) + 1.8)), constrained_layout=True)
    heatmap(
        ax,
        data,
        row_labels,
        ["R@1", "MRR", "MAP", "NDCG@10"],
        f"Per-diagnosis retrieval quality ({key.replace('_', ' ').title()})",
        cmap="Greens",
        vmin=float(np.nanmin(data)),
        vmax=float(np.nanmax(data)),
        cbar_label="Score",
    )
    save_figure(fig, out_dir, "fig_per_diagnosis_retrieval_heatmap")


# -----------------------------------------------------------------------------
# Figure 5: counterfactual summary
# -----------------------------------------------------------------------------
def _ci95(mean: float, std: float, n: int) -> float:
    if n <= 1:
        return 0.0
    return 1.96 * std / math.sqrt(n)


def plot_counterfactual_summary(out_dir: Path) -> None:
    try:
        results, _, _ = load_counterfactual_results()
    except FileNotFoundError:
        print("⚠ counterfactual results not found; skipping counterfactual summary")
        return

    analysis = results["analysis"]
    bm = analysis["basic_metrics"]
    stats = analysis["statistical_tests"]
    ss = stats.get("sample_statistics", {})
    n = int(ss.get("sample_size", results.get("num_samples", 1)))

    means = [
        bm["avg_divergences"]["no_text"],
        bm["avg_divergences"]["no_image"],
        bm["avg_divergences"]["noisy"],
    ]
    stds = [
        ss.get("no_text", {}).get("std", 0.0),
        ss.get("no_image", {}).get("std", 0.0),
        ss.get("noisy", {}).get("std", 0.0),
    ]
    cis = [_ci95(m, s, n) for m, s in zip(means, stds)]
    labels = ["Text removal", "Image removal", "Noise"]
    colors = ["#4C72B0", "#C44E52", "#8172B3"]

    rob = bm["robustness_distribution"]
    rob_labels = ["High", "Medium", "Low"]
    rob_vals = [rob.get("high", 0), rob.get("medium", 0), rob.get("low", 0)]
    total = sum(rob_vals) or 1
    rob_pct = [v / total * 100 for v in rob_vals]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), constrained_layout=True)

    ax = axes[0]
    bars = ax.bar(labels, means, yerr=cis, capsize=4, color=colors, edgecolor="#222222", linewidth=0.8)
    ax.set_title("Modality sensitivity (mean JS ± 95% CI)", fontweight="bold")
    ax.set_ylabel("Jensen–Shannon divergence")
    ax.set_ylim(0, max(means) + max(cis) + 0.08)
    ax.grid(True, axis="y")
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    bottom = 0.0
    stack_colors = ["#55A868", "#DD8452", "#C44E52"]
    for val, lab, col in zip(rob_pct, rob_labels, stack_colors):
        ax.barh(["Robustness"], [val], left=bottom, color=col, edgecolor="white", linewidth=0.8, label=f"{lab} ({val:.1f}%)")
        bottom += val
    ax.set_xlim(0, 100)
    ax.set_title("Robustness distribution", fontweight="bold")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(True, axis="x")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=1)
    ax.set_yticks([])
    ax.set_xlabel("Percent of queries")

    save_figure(fig, out_dir, "fig_counterfactual_summary")


# -----------------------------------------------------------------------------
# Figure 6: perturbation stability curve
# -----------------------------------------------------------------------------
def plot_perturbation_curve(out_dir: Path) -> None:
    try:
        results, _, _ = load_counterfactual_results()
    except FileNotFoundError:
        print("⚠ counterfactual results not found; skipping perturbation curve")
        return

    pa = results["analysis"]["perturbation_analysis"]
    curve = pa.get("degradation_curve", [])
    if not curve:
        print("⚠ degradation curve not found; skipping perturbation curve")
        return

    xs = [float(x["scale"]) for x in curve]
    ys = [float(x["divergence"]) for x in curve]

    scale_analysis = pa.get("scale_analysis", {})
    stable_ratio = [scale_analysis.get(str(s), {}).get("stable_ratio", np.nan) for s in xs]

    fig, ax1 = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
    ax1.plot(xs, ys, marker="o", lw=2.4, color="#C44E52", label="JS divergence")
    for x, y in zip(xs, ys):
        ax1.text(x, y + max(ys) * 0.03, f"{y:.3f}", ha="center", va="bottom", fontsize=9)
    ax1.set_title("Perturbation sensitivity across noise scales", fontweight="bold")
    ax1.set_xlabel("Noise scale (σ)")
    ax1.set_ylabel("Mean JS divergence")
    ax1.grid(True, axis="y")

    ax2 = ax1.twinx()
    ax2.plot(xs, stable_ratio, marker="s", lw=2.0, color="#4C72B0", linestyle="--", alpha=0.95, label="Stable ratio")
    ax2.set_ylabel("Stable ratio")
    ax2.set_ylim(0, 1.05)

    # Build a combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")

    save_figure(fig, out_dir, "fig_perturbation_curve")


# -----------------------------------------------------------------------------
# Figure 7: constraint summary
# -----------------------------------------------------------------------------
def plot_constraint_summary(out_dir: Path) -> None:
    try:
        results, _, _ = load_counterfactual_results()
    except FileNotFoundError:
        print("⚠ counterfactual results not found; skipping constraint summary")
        return

    ca = results["analysis"]["constraints_analysis"]
    axes = ca["axis_summary"]

    axis_order = [
        "evidence_concentration",
        "modality_consistency",
        "decision_boundary_proximity",
        "evidence_diversity",
        "ood_validity",
    ]
    pretty = {
        "evidence_concentration": "Evidence concentration",
        "modality_consistency": "Modality consistency",
        "decision_boundary_proximity": "Decision boundary",
        "evidence_diversity": "Evidence diversity",
        "ood_validity": "OOD validity",
    }

    mean_scores = [axes[k]["mean_score"] for k in axis_order]
    violation_rates = [axes[k]["violation_rate"] for k in axis_order]

    fig, axes_arr = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)

    ax = axes_arr[0]
    bars = ax.barh(
        [pretty[k] for k in axis_order],
        mean_scores,
        color=["#4C72B0", "#55A868", "#8172B3", "#DD8452", "#C44E52"],
        edgecolor="#222222",
        linewidth=0.8,
    )
    ax.set_xlim(0, 1.05)
    ax.set_title("Constraint mean scores", fontweight="bold")
    ax.set_xlabel("Score")
    ax.grid(True, axis="x")
    for b, v in zip(bars, mean_scores):
        ax.text(v + 0.015, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)

    ax = axes_arr[1]
    bars = ax.barh(
        [pretty[k] for k in axis_order],
        np.array(violation_rates) * 100.0,
        color=["#4C72B0", "#55A868", "#8172B3", "#DD8452", "#C44E52"],
        edgecolor="#222222",
        linewidth=0.8,
    )
    ax.set_xlim(0, 100)
    ax.set_title("Constraint violation rates", fontweight="bold")
    ax.set_xlabel("Violation rate (%)")
    ax.grid(True, axis="x")
    for b, v in zip(bars, violation_rates):
        ax.text(v * 100.0 + 1.0, b.get_y() + b.get_height() / 2, f"{v*100:.1f}%", va="center", fontsize=9)

    save_figure(fig, out_dir, "fig_constraint_summary")


# -----------------------------------------------------------------------------
# Figure 8: reliability correlations
# -----------------------------------------------------------------------------
def plot_reliability_correlations(out_dir: Path) -> None:
    try:
        results, _, _ = load_counterfactual_results()
    except FileNotFoundError:
        print("⚠ counterfactual results not found; skipping reliability correlations")
        return

    ca = results["analysis"]["constraints_analysis"]
    diag = ca["diagnosis_breakdown"]

    xs_js = []
    xs_violation = []
    ys_agg = []
    ys_rob = []
    labels = []
    for name, d in diag.items():
        if name.startswith("_"):
            continue
        labels.append(name)
        xs_js.append(d["mean_js"])
        xs_violation.append(d["overall_violation_rate"])
        ys_agg.append(d["aggregate_mean"])
        rob = results["analysis"]["basic_metrics"]["per_diagnosis_stability"].get(name, {})
        total = max(1, rob.get("count", 0))
        ys_rob.append(rob.get("high", 0) / total)

    corrs = ca["reliability_correlations"]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), constrained_layout=True)

    ax = axes[0]
    ax.scatter(xs_js, ys_agg, s=70, color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.8)
    for x, y, lab in zip(xs_js, ys_agg, labels):
        if y < 0.62 or x > 0.20:
            ax.text(x + 0.002, y + 0.005, lab.replace("_", " ").title(), fontsize=8)
    m, b = np.polyfit(xs_js, ys_agg, 1)
    xs = np.linspace(min(xs_js), max(xs_js), 100)
    ax.plot(xs, m * xs + b, color="#C44E52", lw=2.2)
    ax.set_xlabel("Mean JS divergence")
    ax.set_ylabel("Aggregate reliability score")
    ax.set_title("Reliability vs. modality sensitivity", fontweight="bold")
    ax.grid(True)
    txt = f"Spearman ρ = {corrs['aggregate_vs_mean_js']['rho']:.3f}\np = {corrs['aggregate_vs_mean_js']['p_value']:.1e}"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc"))

    ax = axes[1]
    ax.scatter(xs_violation, ys_agg, s=70, color="#55A868", alpha=0.85, edgecolor="white", linewidth=0.8)
    for x, y, lab in zip(xs_violation, ys_agg, labels):
        if y < 0.62 or x > 0.4:
            ax.text(x + 0.006, y + 0.005, lab.replace("_", " ").title(), fontsize=8)
    m, b = np.polyfit(xs_violation, ys_agg, 1)
    xs = np.linspace(min(xs_violation), max(xs_violation), 100)
    ax.plot(xs, m * xs + b, color="#C44E52", lw=2.2)
    ax.set_xlabel("Overall violation rate")
    ax.set_ylabel("Aggregate reliability score")
    ax.set_title("Reliability vs. violation rate", fontweight="bold")
    ax.grid(True)
    txt = f"Spearman ρ = {corrs['aggregate_vs_overall_violation']['rho']:.3f}\np = {corrs['aggregate_vs_overall_violation']['p_value']:.1e}"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc"))

    save_figure(fig, out_dir, "fig_reliability_correlations")


# -----------------------------------------------------------------------------
# Figure 9: diagnosis profile heatmap
# -----------------------------------------------------------------------------
def plot_diagnosis_profile_heatmap(out_dir: Path) -> None:
    try:
        results, _, _ = load_counterfactual_results()
    except FileNotFoundError:
        print("⚠ counterfactual results not found; skipping diagnosis profile heatmap")
        return

    ca = results["analysis"]["constraints_analysis"]
    diag_breakdown = {k: v for k, v in ca["diagnosis_breakdown"].items() if not k.startswith("_")}
    ranking = results["analysis"]["diagnostic_profiles"].get("_comparative", {}).get("stability_ranking")

    ordered = _sorted_diag_keys_by_ranking(diag_breakdown, ranking)
    rows = []
    labels = []
    for name in ordered:
        d = diag_breakdown[name]
        labels.append(name.replace("_", " ").title() + f" (n={d['count']})")
        rob = results["analysis"]["basic_metrics"]["per_diagnosis_stability"].get(name, {})
        count = max(1, d["count"])
        high = rob.get("high", 0) / count
        med = rob.get("medium", 0) / count
        low = rob.get("low", 0) / count
        rows.append(
            [
                d["mean_js"],
                d["aggregate_mean"],
                d["overall_violation_rate"],
                d["ood_validity_mean"],
                d["decision_boundary_proximity_mean"],
                d["modality_consistency_mean"],
                high,
                med,
                low,
            ]
        )

    data = np.array(rows, dtype=float)

    # Split into two heatmaps to avoid mixing count-like and rate-like signals.
    left_cols = [
        "Mean JS",
        "Aggregate score",
        "Violation rate",
        "OOD validity",
        "Boundary",
        "Consistency",
    ]
    left_data = data[:, :6]

    right_cols = ["High rob.", "Med. rob.", "Low rob."]
    right_data = data[:, 6:]

    fig = plt.figure(figsize=(13.2, max(6.0, 0.42 * len(labels) + 1.8)), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.25, 1.0])

    ax1 = fig.add_subplot(gs[0, 0])
    heatmap(
        ax1,
        left_data,
        labels,
        left_cols,
        "Diagnosis-level reliability profiles",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        cbar_label="Score",
    )

    ax2 = fig.add_subplot(gs[0, 1])
    heatmap(
        ax2,
        right_data,
        labels,
        right_cols,
        "Robustness composition",
        cmap="Greens",
        vmin=0.0,
        vmax=1.0,
        cbar_label="Fraction",
    )

    save_figure(fig, out_dir, "fig_diagnosis_profile_heatmap")


# -----------------------------------------------------------------------------
# Optional: overall robustness distribution
# -----------------------------------------------------------------------------
def plot_overall_robustness_distribution(out_dir: Path) -> None:
    try:
        results, _, _ = load_counterfactual_results()
    except FileNotFoundError:
        print("⚠ counterfactual results not found; skipping robustness distribution")
        return

    rob = results["analysis"]["basic_metrics"]["robustness_distribution"]
    vals = [rob.get("high", 0), rob.get("medium", 0), rob.get("low", 0)]
    labels = ["High", "Medium", "Low"]
    colors = ["#55A868", "#DD8452", "#C44E52"]
    total = sum(vals) or 1
    pct = [v / total * 100 for v in vals]

    fig, ax = plt.subplots(figsize=(8.5, 3.6), constrained_layout=True)
    left = 0
    for p, lab, col in zip(pct, labels, colors):
        ax.barh(["Queries"], [p], left=left, color=col, edgecolor="white", linewidth=0.8, label=f"{lab} ({p:.1f}%)")
        left += p
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percent of queries")
    ax.set_title("Overall robustness distribution", fontweight="bold")
    ax.set_yticks([])
    ax.grid(True, axis="x")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    save_figure(fig, out_dir, "fig_overall_robustness_distribution")


# -----------------------------------------------------------------------------
# Optional: retrieval per-diagnosis bar chart from final model
# -----------------------------------------------------------------------------
def plot_top_diagnosis_retrieval_bars(out_dir: Path) -> None:
    try:
        _, per_diag, _ = load_retrieval_results()
    except FileNotFoundError:
        print("⚠ retrieval per-diagnosis metrics not found; skipping retrieval diagnosis bars")
        return

    key = "lora_concept_fusion" if "lora_concept_fusion" in per_diag else next(iter(per_diag.keys()))
    diag_map = per_diag[key]
    rows = sorted(diag_map.items(), key=lambda kv: (kv[1].get("R@1", 0.0), kv[1].get("MRR", 0.0)), reverse=True)[:15]
    labels = [k.replace("_", " ").title() for k, _ in rows]
    scores = [v["R@1"] for _, v in rows]

    fig, ax = plt.subplots(figsize=(10.8, 5.8), constrained_layout=True)
    colors = ["#4C72B0" if s >= 0.95 else "#55A868" if s >= 0.85 else "#DD8452" for s in scores]
    bars = ax.barh(labels[::-1], scores[::-1], color=colors[::-1], edgecolor="#222222", linewidth=0.8)
    ax.set_xlim(0, 1.05)
    ax.set_title("Per-diagnosis retrieval performance (R@1)", fontweight="bold")
    ax.set_xlabel("Recall@1")
    ax.grid(True, axis="x")
    for b in bars:
        v = b.get_width()
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8)
    save_figure(fig, out_dir, "fig_per_diagnosis_retrieval_bars")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def build_output_dir() -> Path:
    for rel in [
        "outputs/paper_figures",
        "test/outputs/paper_figures",
    ]:
        for root in _candidate_roots():
            candidate = root / rel
            # If parent exists, use it; otherwise create later.
            if candidate.parent.exists():
                return ensure_dir(candidate)
    # Fallback to cwd/outputs/paper_figures
    return ensure_dir(Path.cwd() / "outputs" / "paper_figures")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-quality figures from evaluation JSON files.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for figures. If omitted, auto-detected outputs/paper_figures is used.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training curves even if history files are present.",
    )
    parser.add_argument(
        "--skip-extra",
        action="store_true",
        help="Skip extra optional figures such as overall robustness and per-diagnosis bars.",
    )
    args = parser.parse_args()

    set_publication_style()

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else build_output_dir()
    ensure_dir(out_dir)

    print("=" * 78)
    print("Generating paper figures")
    print(f"Output dir: {out_dir}")
    print("=" * 78)

    if not args.skip_training:
        plot_training_curves(out_dir)

    plot_retrieval_summary_heatmap(out_dir)
    plot_recall_curves(out_dir)
    plot_per_diagnosis_retrieval_heatmap(out_dir)

    plot_counterfactual_summary(out_dir)
    plot_perturbation_curve(out_dir)
    plot_constraint_summary(out_dir)
    plot_reliability_correlations(out_dir)
    plot_diagnosis_profile_heatmap(out_dir)

    if not args.skip_extra:
        plot_overall_robustness_distribution(out_dir)
        plot_top_diagnosis_retrieval_bars(out_dir)

    print("=" * 78)
    print("Done.")
    for p in sorted(out_dir.glob("*")):
        print(" -", p.name)
    print("=" * 78)


if __name__ == "__main__":
    main()