"""
Generate All Research-Grade Plots for Paper - FIXED VERSION
Creates publication-ready figures from evaluation results with proper error handling
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
    
    try:
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
    except Exception as e:
        print(f"⚠ Training comparison failed: {e}")

# ============================================================================
# 2. RETRIEVAL PERFORMANCE COMPARISON
# ============================================================================
def plot_retrieval_performance():
    """Bar chart comparing text, image, and fusion retrieval"""
    
    try:
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
    except Exception as e:
        print(f"⚠ Retrieval performance failed: {e}")

# ============================================================================
# 3. STABILITY DISTRIBUTION HEATMAP
# ============================================================================
def plot_stability_heatmap():
    """Heatmap of JS divergence per diagnosis"""
    
    try:
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
    except Exception as e:
        print(f"⚠ Stability heatmap failed: {e}")

# ============================================================================
# 4. ROBUSTNESS LEVEL DISTRIBUTION
# ============================================================================
def plot_robustness_distribution():
    """Bar chart of robustness levels across diagnoses"""
    
    try:
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
    except Exception as e:
        print(f"⚠ Robustness distribution failed: {e}")

# ============================================================================
# 5. RECALL CURVES (R@K)
# ============================================================================
def plot_recall_curves():
    """Line plot showing recall at different K values"""
    
    try:
        with open("outputs/evaluation/retrieval/results.json") as f:
            results = json.load(f)
        
        k_values = [1, 5, 10, 20]
        
        recalls_text = [results['modes']['text']['metrics'][f'R@{k}'] for k in k_values]
        recalls_image = [results['modes']['image']['metrics'][f'R@{k}'] for k in k_values]
        recalls_fusion = [results['modes']['fusion']['metrics'][f'R@{k}'] for k in k_values]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(k_values, recalls_text, 'o-', label='Text', linewidth=2.5, markersize=8, color='#3498db')
        ax.plot(k_values, recalls_image, 's-', label='Image', linewidth=2.5, markersize=8, color='#e74c3c')
        ax.plot(k_values, recalls_fusion, '^-', label='Fusion', linewidth=2.5, markersize=8, color='#2ecc71')
        
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
    except Exception as e:
        print(f"⚠ Recall curves failed: {e}")

# ============================================================================
# 6. PER-DIAGNOSIS RETRIEVAL PERFORMANCE (NEW)
# ============================================================================
def plot_per_diagnosis_retrieval():
    """Bar chart showing R@1 performance per diagnosis"""
    
    try:
        with open("outputs/evaluation/retrieval/per_diagnosis_metrics.json") as f:
            per_diag = json.load(f)
        
        # Get fusion mode results
        fusion_results = per_diag.get('fusion', {})
        
        # Get top 12 diagnoses
        diag_r1 = {diag: metrics['R@1'] for diag, metrics in fusion_results.items()}
        sorted_diags = sorted(diag_r1.items(), key=lambda x: x[1], reverse=True)[:12]
        
        diagnoses = [d[0].replace('_', ' ').title() for d in sorted_diags]
        r1_scores = [d[1] for d in sorted_diags]
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = ['#2ecc71' if s >= 0.95 else '#f39c12' if s >= 0.85 else '#e74c3c' 
                  for s in r1_scores]
        
        bars = ax.bar(range(len(diagnoses)), r1_scores, color=colors, edgecolor='black')
        
        ax.set_xlabel('Diagnosis', fontsize=12, fontweight='bold')
        ax.set_ylabel('Recall@1', fontsize=12, fontweight='bold')
        ax.set_title('Per-Diagnosis Retrieval Performance (Fusion Mode)', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(diagnoses)))
        ax.set_xticklabels(diagnoses, rotation=45, ha='right', fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, r1_scores)):
            ax.text(i, score + 0.01, f'{score:.3f}', 
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "per_diagnosis_retrieval.png", dpi=300, bbox_inches='tight')
        plt.savefig(OUTPUT_DIR / "per_diagnosis_retrieval.pdf", bbox_inches='tight')
        plt.close()
        
        print("✓ Per-diagnosis retrieval plot saved")
    except Exception as e:
        print(f"⚠ Per-diagnosis retrieval failed: {e}")

# ============================================================================
# 7. STATISTICAL SIGNIFICANCE (NEW)
# ============================================================================
def plot_statistical_significance():
    """Visualize statistical test results"""
    
    try:
        with open("outputs/evaluation/counterfactual/results.json") as f:
            results = json.load(f)
        
        stats = results['analysis']['statistical_tests']
        
        # Extract p-values and effect sizes
        tests = []
        p_values = []
        effect_sizes = []
        significant = []
        
        for test_name in ['text_modality_test', 'image_modality_test', 'noise_robustness_test']:
            if test_name in stats:
                test_data = stats[test_name]
                tests.append(test_data['modality'].title())
                p_values.append(test_data['t_pvalue'])
                effect_sizes.append(abs(test_data['cohens_d']))
                significant.append(test_data['significant'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # P-values
        colors1 = ['#2ecc71' if sig else '#e74c3c' for sig in significant]
        bars1 = ax1.bar(tests, p_values, color=colors1, edgecolor='black')
        ax1.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
        ax1.set_ylabel('P-value', fontsize=12, fontweight='bold')
        ax1.set_title('(a) Statistical Significance Tests', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Effect sizes
        colors2 = ['#2ecc71' if e > 0.5 else '#f39c12' if e > 0.2 else '#95a5a6' 
                   for e in effect_sizes]
        bars2 = ax2.bar(tests, effect_sizes, color=colors2, edgecolor='black')
        ax2.set_ylabel("Cohen's d (Effect Size)", fontsize=12, fontweight='bold')
        ax2.set_title("(b) Effect Size Magnitude", fontsize=13, fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax = bar.axes
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "statistical_significance.png", dpi=300, bbox_inches='tight')
        plt.savefig(OUTPUT_DIR / "statistical_significance.pdf", bbox_inches='tight')
        plt.close()
        
        print("✓ Statistical significance plot saved")
    except Exception as e:
        print(f"⚠ Statistical significance failed: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def generate_all_plots():
    """Generate all research plots"""
    
    print("\n" + "="*70)
    print("GENERATING RESEARCH-GRADE FIGURES")
    print("="*70 + "\n")
    
    # Check if evaluation results exist
    results_exist = {
        'training': Path("outputs/models/trained_lora/training_history.json").exists(),
        'retrieval': Path("outputs/evaluation/retrieval/results.json").exists(),
        'counterfactual': Path("outputs/evaluation/counterfactual/results.json").exists(),
        'per_diagnosis': Path("outputs/evaluation/retrieval/per_diagnosis_metrics.json").exists()
    }
    
    print("Checking available data:")
    for name, exists in results_exist.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {name}")
    print()
    
    # Generate plots based on available data
    if results_exist['training']:
        plot_training_comparison()
    
    if results_exist['retrieval']:
        plot_retrieval_performance()
        plot_recall_curves()
        
        if results_exist['per_diagnosis']:
            plot_per_diagnosis_retrieval()
    
    if results_exist['counterfactual']:
        plot_stability_heatmap()
        plot_robustness_distribution()
        plot_statistical_significance()
    
    print("\n" + "="*70)
    print(f"✓ PLOTS SAVED TO: {OUTPUT_DIR}")
    print("="*70)
    
    # Print file list
    print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {file.name}")
    
    # Summary
    print(f"\n📊 Total plots generated: {len(list(OUTPUT_DIR.glob('*.png')))}")

if __name__ == "__main__":
    generate_all_plots()