"""
Counterfactual evaluator for stability, modality effects, and constraint safety.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import json
import sys
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.evaluation.counterfactual.stability_tester import StabilityTester
from scripts.evaluation.counterfactual.robustness_analyzer import RobustnessAnalyzer
from scripts.utils.eval_query_loader import EvaluationQueryDataset


class CounterfactualEvaluator:
    def __init__(
        self,
        contract: dict,
        kb_dir: str,
        output_dir: str,
        device: str = "cpu",
        num_samples: int | None = None,
    ):
        self.contract = contract
        self.kb_dir = Path(kb_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        eval_csv = contract["paths"]["eval_split_csv"]
        print(f"Loading evaluation queries from {eval_csv}...")
        self.eval_dataset = EvaluationQueryDataset(eval_csv)

        if num_samples is not None and num_samples < len(self.eval_dataset):
            self.eval_queries = self.eval_dataset.sample(
                num_samples,
                seed=contract["environment"]["seed"],
            )
        else:
            self.eval_queries = list(self.eval_dataset)

        self.stability_tester = StabilityTester(str(self.kb_dir), contract, device)
        self.robustness_analyzer = RobustnessAnalyzer()

        self.results = {
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(self.eval_queries),
            "eval_queries_csv": eval_csv,
            "kb_dir": str(kb_dir),
            "kb_mode": self.stability_tester.kb_mode,
            "stability_tests": [],
            "analysis": {},
        }

    def run_evaluation(self):
        print("\n" + "=" * 70)
        print("COUNTERFACTUAL STABILITY EVALUATION")
        print(f"Using {len(self.eval_queries)} reserved evaluation queries")
        print("=" * 70 + "\n")

        diagnosis_counts = Counter([q.get("diagnosis_label", "unknown") for q in self.eval_queries])
        print("Evaluation set statistics:")
        print(f"  Total queries: {len(self.eval_queries)}")
        print(f"  Unique diagnoses: {len(diagnosis_counts)}")
        print("\nDiagnosis distribution (top 10):")
        for diag, count in diagnosis_counts.most_common(10):
            print(f"  {diag}: {count}")
        print()

        print("[1/2] Running stability tests on evaluation queries...")
        from tqdm import tqdm
        for query in tqdm(self.eval_queries, desc="Testing stability"):
            result = self.stability_tester.test_query(query)
            self.results["stability_tests"].append(result)

        print(f"\n✓ Completed {len(self.results['stability_tests'])} stability tests")

        print("\n[2/2] Analyzing results...")
        self.results["analysis"] = self.robustness_analyzer.analyze(self.results["stability_tests"])

        print("\n✓ Evaluation complete")

    def _convert_to_json_serializable(self, obj):
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
        serializable_results = self._convert_to_json_serializable(self.results)

        results_path = self.output_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\n✓ Results saved to {results_path}")

        summary_path = self.output_dir / "summary.txt"
        self._save_summary(summary_path)
        print(f"✓ Summary saved to {summary_path}")

        self._save_per_diagnosis_breakdown()

    def _save_per_diagnosis_breakdown(self):
        """
        Save per-diagnosis summary metrics for plotting / manuscript tables.
        """
        by_diag: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for test in self.results.get("stability_tests", []):
            diag = test.get("diagnosis", "unknown")
            by_diag[diag].append(test)

        breakdown = {}
        for diag, tests in by_diag.items():
            stability_levels = [t.get("stability", {}).get("robustness_level", "unknown") for t in tests]
            js_no_text = [t.get("stability", {}).get("js_divergence", {}).get("no_text", 0.0) for t in tests]
            js_no_image = [t.get("stability", {}).get("js_divergence", {}).get("no_image", 0.0) for t in tests]
            js_noisy = [t.get("stability", {}).get("js_divergence", {}).get("noisy", 0.0) for t in tests]
            aggregate_scores = [t.get("constraints", {}).get("aggregate_score", 0.0) for t in tests if "constraints" in t]
            violations = [bool(t.get("constraints", {}).get("overall_violation", False)) for t in tests if "constraints" in t]

            breakdown[diag] = {
                "count": len(tests),
                "robustness_distribution": {
                    "high": int(stability_levels.count("high")),
                    "medium": int(stability_levels.count("medium")),
                    "low": int(stability_levels.count("low")),
                },
                "js_divergence_mean": {
                    "no_text": float(sum(js_no_text) / len(js_no_text)) if js_no_text else 0.0,
                    "no_image": float(sum(js_no_image) / len(js_no_image)) if js_no_image else 0.0,
                    "noisy": float(sum(js_noisy) / len(js_noisy)) if js_noisy else 0.0,
                },
                "constraint_summary": {
                    "mean_aggregate_score": float(sum(aggregate_scores) / len(aggregate_scores)) if aggregate_scores else 0.0,
                    "overall_violation_rate": float(sum(violations) / len(violations)) if violations else 0.0,
                },
            }

        if breakdown:
            breakdown_path = self.output_dir / "per_diagnosis_metrics.json"
            with open(breakdown_path, "w", encoding="utf-8") as f:
                json.dump(self._convert_to_json_serializable(breakdown), f, indent=2)
            print(f"✓ Per-diagnosis metrics saved to {breakdown_path}")

    def _save_summary(self, path: Path):
        analysis = self.results.get("analysis", {})
        basic = analysis.get("basic_metrics", {})
        perturb = analysis.get("perturbation_analysis", {})
        modality = analysis.get("modality_effects", {})
        stats = analysis.get("statistical_tests", {})
        diagnostic = analysis.get("diagnostic_profiles", {})
        constraints = analysis.get("constraints_analysis", {})

        with open(path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("COUNTERFACTUAL / CONSTRAINT EVALUATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp        : {self.results.get('timestamp', 'N/A')}\n")
            f.write(f"Eval samples     : {self.results.get('num_samples', 'N/A')}\n")
            f.write(f"KB mode          : {self.results.get('kb_mode', 'N/A')}\n")
            f.write(f"KB directory     : {self.results.get('kb_dir', 'N/A')}\n")
            f.write(f"Contract         : {self.contract.get('_contract_path', 'N/A')}\n\n")

            f.write("ROBUSTNESS\n")
            f.write("-" * 80 + "\n")
            avg_div = basic.get("avg_divergences", {})
            f.write(f"Mean JS divergence (no_text) : {avg_div.get('no_text', 0.0):.4f}\n")
            f.write(f"Mean JS divergence (no_image): {avg_div.get('no_image', 0.0):.4f}\n")
            f.write(f"Mean JS divergence (noisy)   : {avg_div.get('noisy', 0.0):.4f}\n")
            f.write(f"Robustness distribution      : {basic.get('robustness_distribution', {})}\n\n")

            f.write("MODALITY EFFECTS\n")
            f.write("-" * 80 + "\n")
            if isinstance(modality, dict) and "attribution_summary" in modality:
                attr = modality["attribution_summary"]
                f.write(f"Text contribution mean  : {attr['text_contribution']['mean']:.4f}\n")
                f.write(f"Image contribution mean : {attr['image_contribution']['mean']:.4f}\n")
                f.write(f"Interaction strength    : {attr['interaction_strength']['mean']:.4f}\n")
                f.write(f"Dominant modalities     : {attr.get('dominant_modality_distribution', {})}\n")
            else:
                f.write(f"{modality}\n")
            f.write("\n")

            f.write("PERTURBATION ANALYSIS\n")
            f.write("-" * 80 + "\n")
            if isinstance(perturb, dict):
                f.write(f"Robustness threshold    : {perturb.get('robustness_threshold', None)}\n")
                f.write(f"Scale analysis          : {perturb.get('scale_analysis', {})}\n")
            else:
                f.write(f"{perturb}\n")
            f.write("\n")

            f.write("STATISTICAL TESTS\n")
            f.write("-" * 80 + "\n")
            if isinstance(stats, dict):
                for k in ("text_modality_test", "image_modality_test", "noise_robustness_test"):
                    if k in stats and isinstance(stats[k], dict):
                        f.write(f"{k}: p={stats[k].get('t_pvalue', 1.0):.4g}, d={stats[k].get('cohens_d', 0.0):.4f}\n")
                if "comparative_tests" in stats:
                    f.write(f"Comparative tests: {stats['comparative_tests']}\n")
            else:
                f.write(f"{stats}\n")
            f.write("\n")

            f.write("CONSTRAINTS\n")
            f.write("-" * 80 + "\n")
            if isinstance(constraints, dict) and "axis_summary" in constraints:
                f.write(f"Overall violation rate : {constraints.get('overall_violation_rate', 0.0):.4f}\n")
                for axis, vals in constraints["axis_summary"].items():
                    f.write(
                        f"{axis}: mean={vals['mean_score']:.4f}, "
                        f"violation_rate={vals['violation_rate']:.4f}\n"
                    )
                if "reliability_correlations" in constraints:
                    f.write(f"Reliability correlations: {constraints['reliability_correlations']}\n")
            else:
                f.write(f"{constraints}\n")
            f.write("\n")

            f.write("DIAGNOSTIC PROFILES\n")
            f.write("-" * 80 + "\n")
            if isinstance(diagnostic, dict) and "_comparative" in diagnostic:
                comp = diagnostic["_comparative"]
                f.write(f"Most stable diagnosis : {comp.get('most_stable', 'N/A')}\n")
                f.write(f"Least stable diagnosis: {comp.get('least_stable', 'N/A')}\n")
                f.write(f"Stability variance    : {comp.get('stability_variance', 0.0):.4f}\n")
            else:
                f.write(f"{diagnostic}\n")
            f.write("\n")

            f.write("=" * 80 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Counterfactual Evaluation")
    parser.add_argument("--kb-dir", type=str, default="outputs/kb/kb_final_concept", help="Path to Knowledge Base")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation/counterfactual", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu/cuda)")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of queries to sample")

    args = parser.parse_args()

    from configs.eval_contract import load_eval_contract

    contract = load_eval_contract()
    evaluator = CounterfactualEvaluator(
        contract=contract,
        kb_dir=args.kb_dir,
        output_dir=args.output_dir,
        device=args.device,
        num_samples=args.num_samples,
    )

    evaluator.run_evaluation()
    evaluator.save_results()