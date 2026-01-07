"""
Generate All Research-Grade Plots for Paper
Creates publication-ready figures from evaluation results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import pandas as pd

# Set style for publication
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

OUTPUT_DIR = Path("outputs/paper_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. TRAINING CURVES COMPARISON
# ============================================================================
def plot_training_comparison():
    """Compare LoRA and Fusion training curves side-by-side"""
    
    # Load histories
    with open("outputs/models/trained_lora/training_history.json") as f:
        lora_hist = json.load(f)
    with open("outputs/models/trained_fusion/training_history.json") as f:
        fusion_hist = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # LoRA curves
    epochs_lora = [h['epoch'] for h in lora_hist]
    train_lora = [h['train_loss'] for h in lora_hist]
    val_lora = [h['val_loss'] for h in lora_hist]
    
    axes[0].plot(epochs_lora, train_lora, 'b-o', label='Train', linewidth=2, markersize=6)
    axes[0].plot(epochs_lora, val_lora, 'r-s', label='Validation', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Contrastive Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('(a) LoRA Fine-tuning', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11, loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Fusion curves
    epochs_fusion = [h['epoch'] for h in fusion_hist]
    train_fusion = [h['train_loss'] for h in fusion_hist]
    val_fusion = [h['val_loss'] for h in fusion_hist]
    
    axes[1].plot(epochs_fusion, train_fusion, 'b-o', label='Train', linewidth=2, markersize=6)
    axes[1].plot(epochs_fusion, val_fusion, 'r-s', label='Validation', linewidth=2, markersize=6)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Contrastive Loss', fontsize=12, fontweight='bold')
    axes[1].set_title('(b) Fusion Module Training', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11, loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "training_curves_comparison.pdf", bbox_inches='tight')
    plt.close()
    
    print("✓ Training comparison plot saved")

# ============================================================================
# 2. RETRIEVAL PERFORMANCE COMPARISON
# ============================================================================
def plot_retrieval_performance():
    """Bar chart comparing text, image, and fusion retrieval"""
    
    with open("outputs/evaluation/retrieval/results.json") as f:
        results = json.load(f)
    
    modes = ['Text', 'Image', 'Fusion']
    metrics_names = ['R@1', 'R@5', 'R@10', 'MRR', 'MAP']
    
    data = {metric: [] for metric in metrics_names}
    
    for mode_key, mode_name in zip(['text', 'image', 'fusion'], modes):
        mode_results = results['modes'][mode_key]['metrics']
        for metric in metrics_names:
            data[metric].append(mode_results[metric])
    
    # Create grouped bar chart
    x = np.arange(len(metrics_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, [data[m][0] for m in metrics_names], width, 
                   label='Text', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x, [data[m][1] for m in metrics_names], width,
                   label='Image', color='#e74c3c', edgecolor='black')
    bars3 = ax.bar(x + width, [data[m][2] for m in metrics_names], width,
                   label='Fusion', color='#2ecc71', edgecolor='black')
    
    ax.set_xlabel('Metric', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Retrieval Performance: Modality Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "retrieval_performance.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "retrieval_performance.pdf", bbox_inches='tight')
    plt.close()
    
    print("✓ Retrieval performance plot saved")

# ============================================================================
# 3. STABILITY DISTRIBUTION HEATMAP
# ============================================================================
def plot_stability_heatmap():
    """Heatmap of JS divergence per diagnosis"""
    
    with open("outputs/evaluation/counterfactual/results.json") as f:
        results = json.load(f)
    
    # Collect data
    stability_data = []
    for test in results['stability_tests']:
        diag = test['diagnosis']
        js_divs = test['stability']['js_divergence']
        stability_data.append({
            'diagnosis': diag,
            'no_text': js_divs['no_text'],
            'no_image': js_divs['no_image'],
            'noisy': js_divs['noisy']
        })
    
    # Group by diagnosis
    df = pd.DataFrame(stability_data)
    grouped = df.groupby('diagnosis').mean()
    
    # Select top 15 diagnoses by frequency
    diag_counts = Counter([t['diagnosis'] for t in results['stability_tests']])
    top_diags = [d for d, _ in diag_counts.most_common(15)]
    
    plot_data = grouped.loc[top_diags][['no_text', 'no_image', 'noisy']]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 10))
    
    sns.heatmap(plot_data, annot=True, fmt='.3f', cmap='RdYlGn_r',
                cbar_kws={'label': 'JS Divergence'}, vmin=0, vmax=0.4,
                linewidths=0.5, ax=ax)
    
    ax.set_xlabel('Perturbation Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Diagnosis', fontsize=12, fontweight='bold')
    ax.set_title('Counterfactual Stability: JS Divergence Heatmap', 
                 fontsize=13, fontweight='bold')
    ax.set_xticklabels(['No Text', 'No Image', 'Noisy'], fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "stability_heatmap.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "stability_heatmap.pdf", bbox_inches='tight')
    plt.close()
    
    print("✓ Stability heatmap saved")

# ============================================================================
# 4. MODALITY DEPENDENCY PIE CHART
# ============================================================================
def plot_modality_dependency():
    """Pie chart showing modality dependency distribution"""
    
    with open("outputs/evaluation/counterfactual/results.json") as f:
        results = json.load(f)
    
    # Count modality dependencies
    profiles = results['analysis']['diagnostic_profiles']
    
    # This data needs to be collected from per-diagnosis analysis
    # For now, create aggregate view
    
    dependency_counts = Counter()
    
    for test in results['stability_tests']:
        baseline = test['baseline_distribution']['distribution']
        
        # Simple heuristic: if only one diagnosis with high confidence
        if len(baseline) == 1 and list(baseline.values())[0] > 0.8:
            js_div = test['stability']['js_divergence']
            
            if js_div['no_text'] < 0.1 and js_div['no_image'] < 0.1:
                dependency_counts['Multimodal'] += 1
            elif js_div['no_text'] < 0.1:
                dependency_counts['Image-Dominant'] += 1
            elif js_div['no_image'] < 0.1:
                dependency_counts['Text-Dominant'] += 1
            else:
                dependency_counts['Unstable'] += 1
    
    # Create pie chart
    labels = list(dependency_counts.keys())
    sizes = list(dependency_counts.values())
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']
    explode = (0.05, 0.05, 0.05, 0.05)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels,
                                       colors=colors, autopct='%1.1f%%',
                                       shadow=True, startangle=90,
                                       textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    # Make percentage text bold and white
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(13)
    
    ax.set_title('Modality Dependency Distribution', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "modality_dependency.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "modality_dependency.pdf", bbox_inches='tight')
    plt.close()
    
    print("✓ Modality dependency plot saved")

# ============================================================================
# 5. ROBUSTNESS LEVEL DISTRIBUTION
# ============================================================================
def plot_robustness_distribution():
    """Bar chart of robustness levels across diagnoses"""
    
    with open("outputs/evaluation/counterfactual/results.json") as f:
        results = json.load(f)
    
    per_diag = results['analysis']['basic_metrics']['per_diagnosis_stability']
    
    # Collect data
    diagnoses = []
    high_pcts = []
    medium_pcts = []
    low_pcts = []
    
    # Select top 12 diagnoses by count
    sorted_diags = sorted(per_diag.items(), key=lambda x: x[1]['count'], reverse=True)[:12]
    
    for diag, stats in sorted_diags:
        diagnoses.append(diag.replace('_', ' ').title())
        total = stats['count']
        high_pcts.append(stats.get('high', 0) / total * 100)
        medium_pcts.append(stats.get('medium', 0) / total * 100)
        low_pcts.append(stats.get('low', 0) / total * 100)
    
    # Create stacked bar chart
    x = np.arange(len(diagnoses))
    width = 0.6
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    p1 = ax.bar(x, high_pcts, width, label='High Stability', color='#2ecc71')
    p2 = ax.bar(x, medium_pcts, width, bottom=high_pcts, 
                label='Medium Stability', color='#f39c12')
    p3 = ax.bar(x, low_pcts, width, 
                bottom=np.array(high_pcts) + np.array(medium_pcts),
                label='Low Stability', color='#e74c3c')
    
    ax.set_xlabel('Diagnosis', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Robustness Level Distribution by Diagnosis', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(diagnoses, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "robustness_distribution.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "robustness_distribution.pdf", bbox_inches='tight')
    plt.close()
    
    print("✓ Robustness distribution plot saved")

# ============================================================================
# 6. RECALL CURVES (R@K)
# ============================================================================
def plot_recall_curves():
    """Line plot showing recall at different K values"""
    
    with open("outputs/evaluation/retrieval/results.json") as f:
        results = json.load(f)
    
    k_values = [1, 5, 10, 20]
    
    recalls_text = [results['modes']['text']['metrics'][f'R@{k}'] for k in k_values]
    recalls_image = [results['modes']['image']['metrics'][f'R@{k}'] for k in k_values]
    recalls_fusion = [results['modes']['fusion']['metrics'][f'R@{k}'] for k in k_values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_values, recalls_text, 'o-', label='Text', linewidth=2.5, markersize=8)
    ax.plot(k_values, recalls_image, 's-', label='Image', linewidth=2.5, markersize=8)
    ax.plot(k_values, recalls_fusion, '^-', label='Fusion', linewidth=2.5, markersize=8)
    
    ax.set_xlabel('K (Number of Retrieved Cases)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Recall@K', fontsize=12, fontweight='bold')
    ax.set_title('Recall Curves: Modality Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(k_values)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.95, 1.01])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "recall_curves.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "recall_curves.pdf", bbox_inches='tight')
    plt.close()
    
    print("✓ Recall curves saved")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def generate_all_plots():
    """Generate all research plots"""
    
    print("\n" + "="*70)
    print("GENERATING RESEARCH-GRADE FIGURES")
    print("="*70 + "\n")
    
    try:
        plot_training_comparison()
    except Exception as e:
        print(f"⚠ Training comparison failed: {e}")
    
    try:
        plot_retrieval_performance()
    except Exception as e:
        print(f"⚠ Retrieval performance failed: {e}")
    
    try:
        plot_stability_heatmap()
    except Exception as e:
        print(f"⚠ Stability heatmap failed: {e}")
    
    try:
        plot_modality_dependency()
    except Exception as e:
        print(f"⚠ Modality dependency failed: {e}")
    
    try:
        plot_robustness_distribution()
    except Exception as e:
        print(f"⚠ Robustness distribution failed: {e}")
    
    try:
        plot_recall_curves()
    except Exception as e:
        print(f"⚠ Recall curves failed: {e}")
    
    print("\n" + "="*70)
    print(f"✓ ALL PLOTS SAVED TO: {OUTPUT_DIR}")
    print("="*70)
    
    # Print file list
    print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {file.name}")

if __name__ == "__main__":
    generate_all_plots()