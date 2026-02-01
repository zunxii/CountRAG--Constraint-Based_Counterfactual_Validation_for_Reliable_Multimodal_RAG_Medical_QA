import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("experiments/retrieval")
OUTPUT_DIR = Path("outputs/paper_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_metrics(mode):
    with open(RESULTS_DIR / f"{mode}.json") as f:
        return json.load(f)["metrics"]

modes = ["image", "text", "fusion"]
labels = ["Image Only", "Text Only", "CountRAG (Adaptive)"]
colors = ["#e74c3c", "#3498db", "#2ecc71"]

metrics_to_plot = ["R@1", "R@5", "R@10"]
x = np.arange(len(metrics_to_plot))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))

for i, (mode, label) in enumerate(zip(modes, labels)):
    data = load_metrics(mode)
    values = [data[m] for m in metrics_to_plot]
    
    # Offset bars
    offset = (i - 1) * width
    bars = ax.bar(x + offset, values, width, label=label, color=colors[i], edgecolor="black", alpha=0.9)
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Recall Score', fontsize=12, fontweight='bold')
ax.set_title('Retrieval Performance by Modality', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_to_plot, fontsize=11)
ax.legend(fontsize=10, loc='lower right')
ax.set_ylim(0, 1.15)
ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
out_path = OUTPUT_DIR / "retrieval_performance.png"
plt.savefig(out_path, dpi=300)
print(f"Saved plot to {out_path}")
