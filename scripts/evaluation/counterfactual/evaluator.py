from pathlib import Path
from datetime import datetime
import json
import random
import numpy as np
from typing import Dict, List
from collections import defaultdict

from scripts.evaluation.counterfactual.stability_tester import StabilityTester
from scripts.evaluation.counterfactual.robustness_analyzer import RobustnessAnalyzer


class CounterfactualEvaluator:
    """Research-grade counterfactual evaluation orchestrator"""
    
    def __init__(self, kb_dir: str, num_samples: int, 
                 output_dir: str, device: str = "cpu"):
        self.kb_dir = Path(kb_dir)
        self.num_samples = num_samples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "num_samples": num_samples,
            "stability_tests": []
        }
        
        self.stability_tester = StabilityTester(kb_dir, device)
        self.robustness_analyzer = RobustnessAnalyzer()
    
    def run_evaluation(self):
        """Run comprehensive counterfactual evaluation"""
        print("\n" + "="*70)
        print("COUNTERFACTUAL STABILITY EVALUATION")
        print("Research-Grade Analysis")
        print("="*70 + "\n")
        
        metadata = self.stability_tester.get_metadata()
        sample_indices = self._stratified_sampling(metadata, self.num_samples)
        
        print(f"✓ Selected {len(sample_indices)} samples (stratified by diagnosis)")
        
        print("\n[1/2] Running stability tests...")
        for idx in sample_indices:
            result = self.stability_tester.test_sample(idx)
            self.results["stability_tests"].append(result)
        
        print("\n[2/2] Analyzing results...")
        self.results["analysis"] = self.robustness_analyzer.analyze(
            self.results["stability_tests"]
        )
        
        print("\n✓ Evaluation complete")
    
    def _stratified_sampling(self, metadata: List[Dict], n: int) -> List[int]:
        """Stratified sampling by diagnosis"""
        by_diagnosis = defaultdict(list)
        for idx, entry in enumerate(metadata):
            by_diagnosis[entry["diagnosis_label"]].append(idx)
        
        samples = []
        diagnoses = list(by_diagnosis.keys())
        samples_per_diag = max(1, n // len(diagnoses))
        
        for diag in diagnoses:
            available = by_diagnosis[diag]
            k = min(samples_per_diag, len(available))
            samples.extend(random.sample(available, k))
        
        if len(samples) < n:
            remaining = [i for i in range(len(metadata)) if i not in samples]
            samples.extend(random.sample(remaining, min(n - len(samples), len(remaining))))
        
        return samples[:n]
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def save_results(self):
        """Save evaluation results"""
        # Convert all numpy types to native Python types
        serializable_results = self._convert_to_json_serializable(self.results)
        
        results_path = self.output_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\n✓ Results saved to {results_path}")
        
        summary_path = self.output_dir / "summary.txt"
        self._save_summary(summary_path)
        print(f"✓ Summary saved to {summary_path}")
    
    def _save_summary(self, path: Path):
        """Save human-readable summary"""
        analysis = self.results["analysis"]
        
        with open(path, 'w') as f:
            f.write("COUNTERFACTUAL STABILITY EVALUATION - RESEARCH REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Samples tested: {self.num_samples}\n\n")
            
            # Basic metrics
            basic = analysis.get("basic_metrics", {})
            f.write("AVERAGE JS DIVERGENCES:\n")
            for pert, div in basic.get("avg_divergences", {}).items():
                f.write(f"  {pert}: {div:.4f}\n")
            
            f.write(f"\nROBUSTNESS DISTRIBUTION:\n")
            for level, count in basic.get("robustness_distribution", {}).items():
                f.write(f"  {level}: {count}\n")
            
            # Statistical tests
            stats = analysis.get("statistical_tests", {})
            if stats and stats.get("status") != "insufficient_samples":
                f.write(f"\n\nSTATISTICAL SIGNIFICANCE:\n")
                
                text_test = stats.get("text_modality_test", {})
                f.write(f"\nText Modality Effect:\n")
                f.write(f"  p-value: {text_test.get('t_pvalue', 0):.4f}\n")
                f.write(f"  Cohen's d: {text_test.get('cohens_d', 0):.4f} ({text_test.get('effect_size_interpretation', 'unknown')})\n")
                f.write(f"  Significant: {text_test.get('significant', False)}\n")
                
                image_test = stats.get("image_modality_test", {})
                f.write(f"\nImage Modality Effect:\n")
                f.write(f"  p-value: {image_test.get('t_pvalue', 0):.4f}\n")
                f.write(f"  Cohen's d: {image_test.get('cohens_d', 0):.4f} ({image_test.get('effect_size_interpretation', 'unknown')})\n")
                f.write(f"  Significant: {image_test.get('significant', False)}\n")
            
            # Modality effects
            modality = analysis.get("modality_effects", {})
            if modality and modality.get("status") != "insufficient_data":
                f.write(f"\n\nMODALITY ATTRIBUTION:\n")
                attr = modality.get("attribution_summary", {})
                
                text_contrib = attr.get("text_contribution", {})
                f.write(f"\nText Contribution:\n")
                f.write(f"  Mean: {text_contrib.get('mean', 0):.4f}\n")
                f.write(f"  Median: {text_contrib.get('median', 0):.4f}\n")
                
                image_contrib = attr.get("image_contribution", {})
                f.write(f"\nImage Contribution:\n")
                f.write(f"  Mean: {image_contrib.get('mean', 0):.4f}\n")
                f.write(f"  Median: {image_contrib.get('median', 0):.4f}\n")
                
                interaction = attr.get("interaction_strength", {})
                f.write(f"\nInteraction Effect:\n")
                f.write(f"  Mean: {interaction.get('mean', 0):.4f}\n")
                f.write(f"  Synergy ratio: {interaction.get('positive_ratio', 0):.2%}\n")
            
            # Diagnostic profiles
            profiles = analysis.get("diagnostic_profiles", {})
            comparative = profiles.get("_comparative", {})
            if comparative:
                f.write(f"\n\nPER-DIAGNOSIS ANALYSIS:\n")
                f.write(f"Most stable: {comparative.get('most_stable', 'N/A')}\n")
                f.write(f"Least stable: {comparative.get('least_stable', 'N/A')}\n")
                f.write(f"Stability variance: {comparative.get('stability_variance', 0):.4f}\n")

