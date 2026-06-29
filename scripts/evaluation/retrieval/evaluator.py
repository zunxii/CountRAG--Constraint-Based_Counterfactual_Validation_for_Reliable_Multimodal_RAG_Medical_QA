from pathlib import Path
from datetime import datetime
import json
from typing import Any, Dict

from scripts.utils.eval_query_loader import EvaluationQueryDataset
from scripts.evaluation.retrieval.modes import ModeEvaluator
from scripts.evaluation.retrieval.analysis import ResultsAnalyzer


class RetrievalEvaluator:
    def __init__(self, contract: dict, output_dir: str, device: str = "cpu"):
        self.contract = contract
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        eval_csv = contract["paths"]["eval_split_csv"]
        print(f"Loading evaluation queries from {eval_csv}...")
        self.eval_dataset = EvaluationQueryDataset(eval_csv)

        self.results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "contract_path": contract.get("_contract_path"),
            "eval_queries_csv": eval_csv,
            "num_eval_queries": len(self.eval_dataset),
            "runs": {},
            "analysis": {},
        }

        self.analyzer = ResultsAnalyzer()

        # Full-stack run specs.
        self.run_specs = [
            {
                "name": "base_flat",
                "kb_dir": contract["paths"]["kb_flat_dir"],
                "lora_path": None,
            },
            {
                "name": "base_concept",
                "kb_dir": contract["paths"]["kb_concept_dir"],
                "lora_path": None,
            },
            {
                "name": "lora_concept",
                "kb_dir": contract["paths"]["kb_concept_dir"],
                "lora_path": contract["paths"]["models"]["lora_dir"],
            },
        ]

    def run_all_evaluations(self):
        print("\n" + "=" * 70)
        print("RETRIEVAL EVALUATION (LOCKED)")
        print("=" * 70 + "\n")

        for spec in self.run_specs:
            print(f"\n=== Running {spec['name']} ===")

            mode_eval = ModeEvaluator(
                kb_dir=spec["kb_dir"],
                eval_dataset=self.eval_dataset,
                device=self.device,
                lora_path=spec["lora_path"],
                fusion_path=self.contract["paths"]["models"]["fusion_model_file"],
            )

            self.results["runs"][spec["name"]] = {}

            for mode in ["text", "image", "fusion"]:
                self.results["runs"][spec["name"]][mode] = mode_eval.evaluate_mode(mode)

        analysis = self.analyzer.analyze(self.results["runs"])
        if analysis is None:
            analysis = {}
        self.results["analysis"] = analysis

    def save_results(self):
        results_path = self.output_dir / "results.json"

        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Results saved to {results_path}")

        summary_path = self.output_dir / "summary.txt"
        self._save_summary(summary_path)
        print(f"✓ Summary saved to {summary_path}")

        self._save_per_diagnosis_breakdown()

    def _save_per_diagnosis_breakdown(self):
        """
        Save per-diagnosis metrics if they are present in the mode outputs.
        """
        breakdown = {}

        for run_name, run in self.results.get("runs", {}).items():
            if not isinstance(run, dict):
                continue

            for mode_name, mode_result in run.items():
                if not isinstance(mode_result, dict):
                    continue

                per_diag = None

                if "per_diagnosis_metrics" in mode_result:
                    per_diag = mode_result["per_diagnosis_metrics"]
                elif "per_diagnosis" in mode_result:
                    per_diag = mode_result["per_diagnosis"]
                elif "diagnosis_breakdown" in mode_result:
                    per_diag = mode_result["diagnosis_breakdown"]

                if per_diag is not None:
                    breakdown[f"{run_name}_{mode_name}"] = per_diag

        if breakdown:
            breakdown_path = self.output_dir / "per_diagnosis_metrics.json"
            with open(breakdown_path, "w") as f:
                json.dump(breakdown, f, indent=2, ensure_ascii=False)

            print(f"✓ Per diagnosis metrics saved to {breakdown_path}")

    def _save_summary(self, path: Path):
        """
        Save human-readable evaluation summary.
        Compatible with the current retrieval results schema.
        """
        with open(path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("COUNT-RAG RETRIEVAL EVALUATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Timestamp          : {self.results.get('timestamp', 'N/A')}\n")
            f.write(f"Evaluation Queries : {self.results.get('num_eval_queries', 'N/A')}\n")
            f.write(f"Contract           : {self.results.get('contract_path', 'N/A')}\n\n")

            f.write("Knowledge Bases Used\n")
            f.write("-" * 80 + "\n")
            for spec in self.run_specs:
                f.write(f"{spec['name']:18s} -> {spec['kb_dir']}\n")
            f.write("\n")

            metric_order = [
                "R@1",
                "R@3",
                "R@5",
                "R@10",
                "MRR",
                "MAP",
                "P@1",
                "P@3",
                "P@5",
                "P@10",
                "NDCG@5",
                "NDCG@10",
                "entropy",
                "margin",
            ]

            for run_name, run in self.results.get("runs", {}).items():
                if not isinstance(run, dict):
                    continue

                f.write("=" * 80 + "\n")
                f.write(f"{run_name.upper()}\n")
                f.write("=" * 80 + "\n\n")

                for mode_name, mode_result in run.items():
                    if not isinstance(mode_result, dict):
                        continue

                    metrics = mode_result.get("metrics", mode_result)

                    f.write(f"{mode_name.upper()} MODE\n")
                    f.write("-" * 40 + "\n")

                    for metric in metric_order:
                        if metric in metrics:
                            value = metrics[metric]
                            if isinstance(value, float):
                                f.write(f"{metric:<10}: {value:.4f}\n")
                            else:
                                f.write(f"{metric:<10}: {value}\n")

                    f.write("\n")

            analysis = self.results.get("analysis", {})
            if analysis:
                f.write("=" * 80 + "\n")
                f.write("GLOBAL ANALYSIS\n")
                f.write("=" * 80 + "\n\n")

                best_block = analysis.get("global_best") or analysis.get("best_modes")
                if best_block:
                    f.write("Best Performing Configurations\n")
                    f.write("-" * 40 + "\n")

                    for metric, info in best_block.items():
                        if not isinstance(info, dict):
                            continue

                        run = info.get("run", info.get("setting", ""))
                        mode = info.get("mode", info.get("variant", ""))
                        value = info.get("value", info.get("score", None))

                        if value is None:
                            continue

                        if isinstance(value, (int, float)):
                            f.write(f"{metric:<10}: {run} / {mode} ({value:.4f})\n")
                        else:
                            f.write(f"{metric:<10}: {run} / {mode} ({value})\n")

                    f.write("\n")

                if "comparison" in analysis:
                    f.write("Comparison\n")
                    f.write("-" * 40 + "\n")
                    comparison = analysis["comparison"]
                    if isinstance(comparison, dict):
                        for key, value in comparison.items():
                            f.write(f"{key}: {value}\n")
                    else:
                        f.write(str(comparison) + "\n")
                    f.write("\n")

                if "notes" in analysis:
                    f.write("Notes\n")
                    f.write("-" * 40 + "\n")
                    notes = analysis["notes"]
                    if isinstance(notes, list):
                        for note in notes:
                            f.write(f"- {note}\n")
                    else:
                        f.write(str(notes) + "\n")
                    f.write("\n")

            f.write("=" * 80 + "\n")