import json
import pandas as pd
import os
import argparse
from typing import List, Dict, Any

def load_json(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r') as f:
        return json.load(f)

def get_ground_truth(df: pd.DataFrame, image_path: str) -> Dict[str, str]:
    # Image paths in JSON might have "data/" prefix or be absolute
    # In CSV they are usually relative or filenames. 
    # Let's try to match by filename.
    
    filename = os.path.basename(image_path)
    
    # Try exact match on image_path column if possible, otherwise filename
    row = df[df['image_path'].apply(lambda x: os.path.basename(str(x))) == filename]
    
    if row.empty:
        return None
    
    return {
        "diagnosis": row.iloc[0]['category'],
        "summary": row.iloc[0]['Question_summ'],
        "full_row": row.iloc[0].to_dict()
    }

def analyze_prediction(data: Dict[str, Any], ground_truth: Dict[str, str]) -> Dict[str, Any]:
    gt_diagnosis = ground_truth['diagnosis'].lower()
    
    # Check baseline clusters
    baseline_clusters = data.get('stability', {}).get('baseline', {}).get('distribution', {})
    baseline_top_cluster = max(baseline_clusters, key=baseline_clusters.get) if baseline_clusters else "None"
    
    # Check if GT is in clusters
    gt_in_baseline = gt_diagnosis in baseline_clusters
    
    # Check noisy clusters (latent knowledge)
    noisy_clusters = data.get('stability', {}).get('noisy', {}).get('distribution', {})
    gt_in_noisy = gt_diagnosis in noisy_clusters
    
    # Check retrieved items
    retrieved = data.get('stability', {}).get('retrieved', [])
    retrieved_diagnoses = [r['diagnosis_label'].lower() for r in retrieved]
    gt_in_retrieved = gt_diagnosis in retrieved_diagnoses
    
    # Retrieval Recall @ K (is it in the retrieved list?)
    recall = 1.0 if gt_in_retrieved else 0.0
    
    return {
        "gt_diagnosis": gt_diagnosis,
        "baseline_top_prediction": baseline_top_cluster,
        "gt_in_baseline": gt_in_baseline,
        "gt_in_noisy": gt_in_noisy,
        "baseline_clusters": baseline_clusters,
        "noisy_clusters": noisy_clusters,
        "recall": recall
    }

def main():
    parser = argparse.ArgumentParser(description="Compare RAG outputs with CLIPSyntel Baseline")
    parser.add_argument("--json_files", nargs='+', default=["1.json", "2.json", "3.json"], help="Path to JSON output files")
    parser.add_argument("--csv_file", default="data/processed/train.csv", help="Path to ground truth CSV")
    parser.add_argument("--output_report", default="outputs/comparison_report.md", help="Path to output report")
    
    args = parser.parse_args()
    
    print(f"Loading Ground Truth from {args.csv_file}...")
    try:
        df = pd.read_csv(args.csv_file)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    report_lines = []
    report_lines.append("# Comparison Report: Research RAG vs CLIPSyntel Baseline")
    report_lines.append(f"**Date**: {pd.Timestamp.now()}")
    report_lines.append("")
    report_lines.append("| Image | Ground Truth (CLIPSyntel) | RAG Top Prediction | Match (Baseline)? | Match (Noisy)? | Recall |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    detailed_sections = []
    
    for json_file in args.json_files:
        if not os.path.exists(json_file):
            print(f"Warning: {json_file} does not exist. Skipping.")
            continue
            
        print(f"Processing {json_file}...")
        data = load_json(json_file)
        
        image_path = data['query']['image']
        ground_truth = get_ground_truth(df, image_path)
        
        if not ground_truth:
            print(f"  Warning: No ground truth found for image {image_path}")
            report_lines.append(f"| {os.path.basename(json_file)} | Not Found | - | - | - | - |")
            continue
            
        analysis = analyze_prediction(data, ground_truth)
        
        # Table Row
        match_base_icon = "✅" if analysis['gt_in_baseline'] else "❌"
        match_noisy_icon = "✅" if analysis['gt_in_noisy'] else "❌"
        recall_icon = "✅" if analysis['recall'] > 0 else "❌"
        
        row_str = f"| {os.path.basename(json_file)} (`{os.path.basename(image_path)}`) | **{analysis['gt_diagnosis']}** | {analysis['baseline_top_prediction']} | {match_base_icon} | {match_noisy_icon} | {recall_icon} |"
        report_lines.append(row_str)
        
        # Detailed Section
        detail = []
        detail.append(f"## {os.path.basename(json_file)}: {analysis['gt_diagnosis'].title()}")
        detail.append(f"- **Query**: \"{data['query']['text'][:100]}...\"")
        detail.append(f"- **Ground Truth Summary**: {ground_truth['summary']}")
        detail.append(f"- **RAG Baseline Clusters**: {analysis['baseline_clusters']}")
        detail.append(f"- **RAG Noisy Clusters** (Uncertainty): {analysis['noisy_clusters']}")
        detail.append(f"- **Analysis**: The model {'successfully retrieved' if analysis['recall'] else 'failed to retrieve'} the correct diagnosis.")
        if analysis['gt_in_noisy'] and not analysis['gt_in_baseline']:
            detail.append("  - **Insight**: The model was uncertain in the baseline but found the correct diagnosis in the noisy perturbation, indicating latent knowledge.")
        elif not analysis['recall']:
            detail.append("  - **Failure**: The ground truth was not present in the top retrieved contexts.")
        
        detailed_sections.append("\n".join(detail))

    # Write Report
    full_report = "\n".join(report_lines) + "\n\n" + "\n\n".join(detailed_sections)
    
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, "w") as f:
        f.write(full_report)
    
    print(f"Report generated at {args.output_report}")

if __name__ == "__main__":
    main()
