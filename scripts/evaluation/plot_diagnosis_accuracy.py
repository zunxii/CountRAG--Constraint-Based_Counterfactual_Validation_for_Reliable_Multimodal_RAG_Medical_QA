import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

RESULTS_FILE = Path("experiments/retrieval/fusion.json")
OUTPUT_DIR = Path("outputs/paper_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_per_label_accuracy():
    if not RESULTS_FILE.exists():
        print(f"File not found: {RESULTS_FILE}")
        return {}
    with open(RESULTS_FILE) as f:
        data = json.load(f)
        return data["metrics"].get("per_label_accuracy", {})

def plot_accuracy():
    full_data = load_per_label_accuracy()
    if not full_data:
        return

    # Sort by accuracy
    sorted_items = sorted(full_data.items(), key=lambda x: x[1], reverse=True)
    labels = [x[0].title() for x in sorted_items]
    accuracies = [x[1] for x in sorted_items]

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color map from high to low
    colors = sns.color_palette("viridis", len(labels))

    bars = ax.bar(labels, accuracies, color=colors, edgecolor="black", alpha=0.9)

    # Styling
    ax.set_ylabel('Recall@1 Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Diagnostic Accuracy by Condition (Fusion Model)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    
    # Rotate x labels nicely
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # Value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    out_path = OUTPUT_DIR / "diagnosis_accuracy.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    plot_accuracy()
