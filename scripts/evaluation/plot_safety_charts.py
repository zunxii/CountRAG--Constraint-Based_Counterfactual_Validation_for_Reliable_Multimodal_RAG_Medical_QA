import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from math import pi

# ensure styles
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

def load_data(filepath='outputs/paper_plots/evaluation_data.json'):
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_radar_chart():
    data = load_data()
    safety_data = data['safety_profile']
    
    categories = safety_data['categories']
    N = len(categories)

    # Prepping data for radar chart (circular, so repeat first point)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += [angles[0]]  # close the loop
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors = {
        'Text-only RAG': '#e74c3c', 
        'Standard Multimodal': '#f39c12', 
        'Fusion + LoRA': '#3498db', 
        'CountRAG-Clinic (Ours)': '#27ae60'
    }
    
    styles = {
        'Text-only RAG': ':', 
        'Standard Multimodal': '--', 
        'Fusion + LoRA': '-.', 
        'CountRAG-Clinic (Ours)': '-'
    }
    
    widths = {
        'Text-only RAG': 1.5,
        'Standard Multimodal': 1.5,
        'Fusion + LoRA': 1.5,
        'CountRAG-Clinic (Ours)': 2.5
    }

    alphas = {
        'Text-only RAG': 0.05,
        'Standard Multimodal': 0.05,
        'Fusion + LoRA': 0.1,
        'CountRAG-Clinic (Ours)': 0.2
    }

    for model_name, values in safety_data['models'].items():
        # Clean naming consistency if json key differs slightly
        # We assume json keys match the plot labels desired
        
        # Handle None values: None means "Not Applicable" or Max Risk in this view.
        # User requested to attach to the outest circle (Max value).
        cleaned_values = [v if v is not None else 45.0 for v in values]
        
        vals = cleaned_values + [cleaned_values[0]] # Close the loop
        
        ax.plot(angles, vals, linewidth=widths.get(model_name, 1.5), 
                linestyle=styles.get(model_name, '-'), 
                label=model_name, 
                color=colors.get(model_name, 'black'))
        
        # Only fill if no NaNs (otherwise fill behaves variably, but usually ignores validation)
        # Using a masked array for fill is safer but standard fill handles nans by not filling that segment
        ax.fill(angles, vals, color=colors.get(model_name, 'black'), alpha=alphas.get(model_name, 0.1))

    # Formatting
    plt.xticks(angles[:-1], categories, size=12, fontweight='bold')
    ax.set_rlabel_position(0)
    plt.yticks([10, 20, 30, 40], ["10%", "20%", "30%", "40%"], color="grey", size=10)
    
    # Standard Axis: 0 (Center, Good) -> 45 (Edge, Bad)
    plt.ylim(0, 45)
    
    # Title & Legend
    plt.title("Safety Risk Profile\n(Lower is Better)", size=16, weight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    # Adjust layout to make room for legend (Right side)
    # Removing tight_layout which conflicts with custom margins
    plt.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.1)
    
    plt.savefig('outputs/paper_plots/safety_radar_chart.png', dpi=300)
    plt.close()
    print("Generated radar chart: outputs/paper_plots/safety_radar_chart.png")

def plot_violation_summary():
    data = load_data()
    summary_data = data['violation_summary']
    
    models = summary_data['models']
    violations = summary_data['total_violation_rate']
    
    colors = ['#e74c3c', '#f39c12', '#3498db', '#27ae60']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, violations, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

    # Add values on top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height}%',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Formatting
    plt.ylabel('Total Constraint Violation Rate (%)', fontsize=12)
    plt.title('Reliability Comparison: Rate of "Silent Failures"', fontsize=14, fontweight='bold')
    plt.yticks(np.arange(0, 81, 10))
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('outputs/paper_plots/violation_summary_bar.png', dpi=300)
    plt.close()
    print("Generated bar chart: outputs/paper_plots/violation_summary_bar.png")

if __name__ == "__main__":
    import os
    os.makedirs('outputs/paper_plots', exist_ok=True)
    plot_radar_chart()
    plot_violation_summary()
