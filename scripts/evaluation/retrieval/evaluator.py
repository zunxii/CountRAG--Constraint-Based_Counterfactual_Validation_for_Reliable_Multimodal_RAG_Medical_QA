"""
Retrieval evaluator - FIXED to save complete metadata for plotting
"""
from pathlib import Path
from datetime import datetime
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.evaluation.retrieval.metrics import MetricsCalculator
from scripts.evaluation.retrieval.modes import ModeEvaluator
from scripts.evaluation.retrieval.analysis import ResultsAnalyzer
from scripts.utils.eval_query_loader import EvaluationQueryDataset


class RetrievalEvaluator:
    """Orchestrates retrieval evaluation using reserved eval queries"""
    
    def __init__(
        self, 
        kb_dir: str, 
        output_dir: str, 
        device: str = "cpu"
    ):
        self.kb_dir = Path(kb_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Load evaluation queries from correct path
        eval_csv = "data/processed/eval_queries.csv"
        print(f"Loading evaluation queries from {eval_csv}...")
        self.eval_dataset = EvaluationQueryDataset(eval_csv)
        print(f"✓ Loaded {len(self.eval_dataset)} evaluation queries")
        
        # Initialize evaluators
        self.mode_eval = ModeEvaluator(kb_dir, self.eval_dataset, device)
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "kb_dir": str(kb_dir),
            "eval_queries_csv": eval_csv,
            "num_eval_queries": len(self.eval_dataset),
            "modes": {}
        }
        
        self.metrics_calc = MetricsCalculator()
        self.analyzer = ResultsAnalyzer()
    
    def run_all_evaluations(self):
        """Run all retrieval evaluations"""
        print("\n" + "="*70)
        print("RETRIEVAL EVALUATION")
        print(f"Using {len(self.eval_dataset)} reserved evaluation queries")
        print("="*70 + "\n")
        
        # Print dataset statistics
        stats = self.eval_dataset.get_statistics()
        print(f"Evaluation set statistics:")
        print(f"  Total queries: {stats['total_queries']}")
        print(f"  Unique diagnoses: {stats['num_unique_diagnoses']}")
        print(f"\nDiagnosis distribution (top 10):")
        sorted_diag = sorted(stats['diagnosis_distribution'].items(), 
                            key=lambda x: x[1], reverse=True)[:10]
        for diag, count in sorted_diag:
            print(f"  {diag}: {count}")
        print()
        
        # Evaluate each mode with detailed per-diagnosis tracking
        for mode in ["text", "image", "fusion"]:
            print(f"\n📊 Evaluating {mode.upper()} mode...")
            results = self.mode_eval.evaluate_mode(mode)
            self.results["modes"][mode] = results
            
            # Print summary
            metrics = results["metrics"]
            print(f"\n{mode.upper()} Results:")
            print(f"  R@1:  {metrics['R@1']:.4f}")
            print(f"  R@5:  {metrics['R@5']:.4f}")
            print(f"  R@10: {metrics['R@10']:.4f}")
            print(f"  MRR:  {metrics['MRR']:.4f}")
            print(f"  MAP:  {metrics['MAP']:.4f}")
        
        # Run analysis
        print("\n📈 Analyzing results...")
        self.results["analysis"] = self.analyzer.analyze(self.results["modes"])
        
        print("\n✓ Evaluation complete")
    
    def save_results(self):
        """Save evaluation results with complete metadata"""
        results_path = self.output_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to {results_path}")
        
        summary_path = self.output_dir / "summary.txt"
        self._save_summary(summary_path)
        print(f"✓ Summary saved to {summary_path}")
        
        # Save per-diagnosis breakdown for plotting
        self._save_per_diagnosis_breakdown()
    
    def _save_per_diagnosis_breakdown(self):
        """Save detailed per-diagnosis metrics for plotting"""
        breakdown = {}
        
        for mode, results in self.results["modes"].items():
            if "per_diagnosis_metrics" in results:
                breakdown[mode] = results["per_diagnosis_metrics"]
        
        if breakdown:
            breakdown_path = self.output_dir / "per_diagnosis_metrics.json"
            with open(breakdown_path, 'w') as f:
                json.dump(breakdown, f, indent=2)
            print(f"✓ Per-diagnosis metrics saved to {breakdown_path}")
    
    def _save_summary(self, path: Path):
        """Save human-readable summary"""
        with open(path, 'w') as f:
            f.write("RETRIEVAL EVALUATION SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Evaluation queries: {self.results['num_eval_queries']}\n")
            f.write(f"KB directory: {self.results['kb_dir']}\n")
            f.write(f"Timestamp: {self.results['timestamp']}\n\n")
            
            for mode, results in self.results["modes"].items():
                f.write(f"{mode.upper()} Mode:\n")
                f.write(f"  R@1:  {results['metrics']['R@1']:.4f}\n")
                f.write(f"  R@5:  {results['metrics']['R@5']:.4f}\n")
                f.write(f"  R@10: {results['metrics']['R@10']:.4f}\n")
                f.write(f"  MRR:  {results['metrics']['MRR']:.4f}\n")
                f.write(f"  MAP:  {results['metrics']['MAP']:.4f}\n\n")
            
            # Add best mode analysis
            if "analysis" in self.results:
                f.write("\nBest Modes:\n")
                best = self.results["analysis"].get("best_modes", {})
                for metric, info in best.items():
                    f.write(f"  {metric}: {info['mode']} ({info['value']:.4f})\n")