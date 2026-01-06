"""
counterfactual evaluator - uses reserved evaluation queries
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
        eval_queries_csv: str,  # CHANGED: Use eval queries
        output_dir: str, 
        device: str = "cpu"
    ):
        self.kb_dir = Path(kb_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Load evaluation queries
        print(f"Loading evaluation queries from {eval_queries_csv}...")
        self.eval_dataset = EvaluationQueryDataset(eval_queries_csv)
        print(f"✓ Loaded {len(self.eval_dataset)} evaluation queries")
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(self.eval_dataset),
            "eval_queries_csv": eval_queries_csv,
            "stability_tests": []
        }
        
        self.stability_tester = StabilityTester(kb_dir, device)
        self.robustness_analyzer = RobustnessAnalyzer()
    
    def run_evaluation(self):
        """Run comprehensive counterfactual evaluation"""
        print("\n" + "="*70)
        print("COUNTERFACTUAL STABILITY EVALUATION")
        print(f"Using {len(self.eval_dataset)} reserved evaluation queries")
        print("="*70 + "\n")
        
        # Print dataset statistics
        stats = self.eval_dataset.get_statistics()
        print(f"Evaluation set statistics:")
        print(f"  Total queries: {stats['total_queries']}")
        print(f"  Unique diagnoses: {stats['num_unique_diagnoses']}")
        print()
        
        print("[1/2] Running stability tests on all evaluation queries...")
        
        # Test ALL evaluation queries
        for query in self.eval_dataset:
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
        """Save evaluation results"""
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
            
            f.write(f"Evaluation queries tested: {self.results['num_samples']}\n")
            f.write(f"Source: {self.results['eval_queries_csv']}\n\n")
            
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
                f.write(f"  Cohen's d: {text_test.get('cohens_d', 0):.4f}\n")
                f.write(f"  Significant: {text_test.get('significant', False)}\n")