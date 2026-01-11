"""
Counterfactual evaluator - FIXED to use eval queries and save complete metadata
"""
from pathlib import Path
from datetime import datetime
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.evaluation.counterfactual.stability_tester import StabilityTester
from scripts.evaluation.counterfactual.robustness_analyzer import RobustnessAnalyzer
from scripts.utils.eval_query_loader import EvaluationQueryDataset


class CounterfactualEvaluator:
    """Research-grade counterfactual evaluation using reserved queries"""
    
    def __init__(
        self, 
        kb_dir: str, 
        output_dir: str, 
        device: str = "cpu",
        num_samples: int = None
    ):
        self.kb_dir = Path(kb_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Load evaluation queries
        eval_csv = "data/processed/splits/eval.csv"
        print(f"Loading evaluation queries from {eval_csv}...")
        self.eval_dataset = EvaluationQueryDataset(eval_csv)
        
        # Sample if requested
        if num_samples is not None and num_samples < len(self.eval_dataset):
            print(f"Sampling {num_samples} queries from {len(self.eval_dataset)} available...")
            self.eval_queries = self.eval_dataset.sample(num_samples, seed=42)
        else:
            self.eval_queries = list(self.eval_dataset)
        
        print(f"✓ Using {len(self.eval_queries)} evaluation queries")
        
        # Initialize tester (auto-detects KB mode)
        self.stability_tester = StabilityTester(kb_dir, device)
        
        # NEW: Print KB mode for transparency
        print(f"✓ KB mode: {self.stability_tester.kb_mode}")
        
        self.robustness_analyzer = RobustnessAnalyzer()
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(self.eval_queries),
            "eval_queries_csv": eval_csv,
            "kb_dir": str(kb_dir),
            "kb_mode": self.stability_tester.kb_mode,  # NEW: Add to results
            "stability_tests": []
        }
        
        self.stability_tester = StabilityTester(kb_dir, device)
        self.robustness_analyzer = RobustnessAnalyzer()
    
    def run_evaluation(self):
        """Run comprehensive counterfactual evaluation"""
        print("\n" + "="*70)
        print("COUNTERFACTUAL STABILITY EVALUATION")
        print(f"Using {len(self.eval_queries)} reserved evaluation queries")
        print("="*70 + "\n")
        
        # Print dataset statistics
        from collections import Counter
        diagnosis_counts = Counter([q['diagnosis_label'] for q in self.eval_queries])
        
        print(f"Evaluation set statistics:")
        print(f"  Total queries: {len(self.eval_queries)}")
        print(f"  Unique diagnoses: {len(diagnosis_counts)}")
        print(f"\nDiagnosis distribution (top 10):")
        for diag, count in diagnosis_counts.most_common(10):
            print(f"  {diag}: {count}")
        print()
        
        print("[1/2] Running stability tests on evaluation queries...")
        
        # Test all evaluation queries
        from tqdm import tqdm
        for query in tqdm(self.eval_queries, desc="Testing stability"):
            result = self.stability_tester.test_query(query)
            self.results["stability_tests"].append(result)
        
        print(f"\n✓ Completed {len(self.results['stability_tests'])} stability tests")
        
        print("\n[2/2] Analyzing results...")
        self.results["analysis"] = self.robustness_analyzer.analyze(
            self.results["stability_tests"]
        )
        
        print("\n✓ Evaluation complete")
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to Python native types"""
        import numpy as np
        
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
        """Save evaluation results with complete metadata"""
        serializable_results = self._convert_to_json_serializable(self.results)
        
        results_path = self.output_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\n✓ Results saved to {results_path}")
        
        summary_path = self.output_dir / "summary.txt"
        self._save_summary(summary_path)
        print(f"✓ Summary saved to {summary_path}")
        
        # Save per-diagnosis breakdown for plotting
        self._save_per_diagnosis_breakdown()
    
    def _save_per_diagnosis_breakdown(self):
        """Save detailed per-diagnosis stability for plotting"""
        from collections import defaultdict
        
        per_diag = defaultdict(lambda: {
            'high': 0, 'medium': 0, 'low': 0,
            'js_divergences': {'no_text': [], 'no_image': [], 'noisy': []}
        })
        
        for test in self.results["stability_tests"]:
            diag = test["diagnosis"]
            robustness = test["stability"]["robustness_level"]
            js_div = test["stability"]["js_divergence"]
            
            per_diag[diag][robustness] += 1
            for key in ['no_text', 'no_image', 'noisy']:
                per_diag[diag]['js_divergences'][key].append(js_div[key])
        
        # Convert to serializable format and compute averages
        per_diag_serializable = {}
        for diag, data in per_diag.items():
            per_diag_serializable[diag] = {
                'robustness_counts': {
                    'high': data['high'],
                    'medium': data['medium'],
                    'low': data['low']
                },
                'avg_js_divergences': {
                    key: sum(vals) / len(vals) if vals else 0.0
                    for key, vals in data['js_divergences'].items()
                }
            }
        
        breakdown_path = self.output_dir / "per_diagnosis_stability.json"
        with open(breakdown_path, 'w') as f:
            json.dump(per_diag_serializable, f, indent=2)
        print(f"✓ Per-diagnosis stability saved to {breakdown_path}")
    
    def _save_summary(self, path: Path):
        """Save human-readable summary"""
        analysis = self.results["analysis"]
        
        with open(path, 'w') as f:
            f.write("COUNTERFACTUAL STABILITY EVALUATION - RESEARCH REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Evaluation queries tested: {self.results['num_samples']}\n")
            f.write(f"Source: {self.results['eval_queries_csv']}\n")
            f.write(f"KB: {self.results['kb_dir']}\n")
            f.write(f"Timestamp: {self.results['timestamp']}\n\n")
            
            basic = analysis.get("basic_metrics", {})
            f.write("AVERAGE JS DIVERGENCES:\n")
            for pert, div in basic.get("avg_divergences", {}).items():
                f.write(f"  {pert}: {div:.4f}\n")
            
            f.write(f"\nROBUSTNESS DISTRIBUTION:\n")
            for level, count in basic.get("robustness_distribution", {}).items():
                f.write(f"  {level}: {count}\n")
            
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
            
            # Add top stable/unstable diagnoses
            profiles = analysis.get("diagnostic_profiles", {})
            if "_comparative" in profiles:
                comp = profiles["_comparative"]
                f.write(f"\n\nDIAGNOSTIC STABILITY:\n")
                f.write(f"Most stable: {comp.get('most_stable', 'N/A')}\n")
                f.write(f"Least stable: {comp.get('least_stable', 'N/A')}\n")