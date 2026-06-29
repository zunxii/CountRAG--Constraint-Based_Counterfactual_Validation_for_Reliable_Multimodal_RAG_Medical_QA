import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "figure.dpi": 300
})

metrics = [
    "Evidence\nConcentration",
    "Modality\nConsistency",
    "Decision\nBoundary",
    "Evidence\nDiversity",
    "OOD\nValidity"
]

mean_scores = [
    0.8269,
    0.6623,
    0.9389,
    0.8405,
    0.5471
]

violation_rates = [
    0.020,
    0.015,
    0.045,
    0.075,
    0.295
]

N = len(metrics)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

score_values = mean_scores + mean_scores[:1]
viol_values = violation_rates + violation_rates[:1]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(11,5.5),
    subplot_kw=dict(polar=True)
)

plots = [
    (
        axes[0],
        score_values,
        "Constraint Satisfaction",
        (0,1.0),
        "#2F5D8C"
    ),
    (
        axes[1],
        viol_values,
        "Constraint Violation Rate",
        (0,0.35),
        "#8B3A3A"
    )
]

for ax, values, title, ylim, color in plots:

    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)

    ax.set_ylim(*ylim)

    ax.grid(alpha=0.35)

    ax.spines["polar"].set_alpha(0.35)

    ax.plot(
        angles,
        values,
        linewidth=2.5,
        color=color
    )

    ax.fill(
        angles,
        values,
        color=color,
        alpha=0.18
    )

    ax.set_title(
        title,
        pad=22,
        weight="bold"
    )

plt.tight_layout()

output_dir = Path("outputs/figures")
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(
    output_dir / "constraint_radar.pdf",
    bbox_inches="tight",
    transparent=True
)

plt.savefig(
    output_dir / "constraint_radar.png",
    bbox_inches="tight",
    transparent=True,
    dpi=600
)

plt.show()