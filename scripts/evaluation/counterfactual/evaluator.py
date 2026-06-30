"""
Counterfactual evaluator for system-level constraint comparison.

Runs:
- traditional  = flat KB + base encoder
- concept      = concept KB + base encoder
- countrag     = concept KB + LoRA encoder

This is the evaluation you want for the radar figure.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import csv
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.evaluation.counterfactual.stability_tester import StabilityTester
from scripts.evaluation.counterfactual.robustness_analyzer import RobustnessAnalyzer
from scripts.utils.eval_query_loader import EvaluationQueryDataset


@dataclass(frozen=True)
class SystemSpec:
    name: str
    kb_dir: str
    lora_path: Optional[str]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path: Path, obj: Any) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_safe_json(obj), f, indent=2, ensure_ascii=False)


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _safe_json(row.get(k, "")) for k in fieldnames})


class CounterfactualEvaluator:
    def __init__(
        self,
        contract: dict,
        kb_dir: Optional[str],
        output_dir: str,
        device: str = "cpu",
        num_samples: int | None = None,
        compare_systems: bool = True,
    ):
        self.contract = contract
        self.kb_dir = Path(kb_dir) if kb_dir else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.compare_systems = compare_systems

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

        self.robustness_analyzer = RobustnessAnalyzer()

        self.system_specs = self._build_system_specs()

        self.results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(self.eval_queries),
            "eval_queries_csv": eval_csv,
            "compare_systems": self.compare_systems,
            "systems": [spec.name for spec in self.system_specs],
            "runs": {},
            "analysis": {},
            "radar_ready": [],
        }

    def _build_system_specs(self) -> List[SystemSpec]:
        paths = self.contract["paths"]
        models = paths["models"]

        if self.compare_systems:
            return [
                SystemSpec(
                    name="traditional",
                    kb_dir=paths["kb_flat_dir"],
                    lora_path=None,
                ),
                SystemSpec(
                    name="concept",
                    kb_dir=paths["kb_concept_dir"],
                    lora_path=None,
                ),
                SystemSpec(
                    name="countrag",
                    kb_dir=paths["kb_concept_dir"],
                    lora_path=models["lora_dir"],
                ),
            ]

        # legacy single-system fallback
        kb_dir = str(self.kb_dir or paths["kb_concept_dir"])
        return [
            SystemSpec(
                name="single",
                kb_dir=kb_dir,
                lora_path=models["lora_dir"],
            )
        ]

    def run_evaluation(self):
        print("\n" + "=" * 70)
        print("COUNTERFACTUAL / CONSTRAINT SYSTEM COMPARISON")
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

        for spec in self.system_specs:
            print("\n" + "=" * 70)
            print(f"RUNNING SYSTEM: {spec.name.upper()}")
            print(f"KB: {spec.kb_dir}")
            print(f"LoRA: {spec.lora_path if spec.lora_path else 'None'}")
            print("=" * 70)

            tester = StabilityTester(
                kb_dir=spec.kb_dir,
                contract=self.contract,
                device=self.device,
                lora_path=spec.lora_path,
                fusion_path=self.contract["paths"]["models"]["fusion_model_file"],
            )

            stability_tests: List[Dict[str, Any]] = []
            print(f"[1/2] Running stability tests for {spec.name}...")
            from tqdm import tqdm
            for query in tqdm(self.eval_queries, desc=f"Testing {spec.name}"):
                result = tester.test_query(query)
                result["system"] = spec.name
                stability_tests.append(result)

            print(f"✓ Completed {len(stability_tests)} stability tests for {spec.name}")

            print(f"[2/2] Analyzing results for {spec.name}...")
            analysis = self.robustness_analyzer.analyze(stability_tests)

            self.results["runs"][spec.name] = {
                "kb_dir": spec.kb_dir,
                "lora_path": spec.lora_path,
                "kb_mode": tester.kb_mode,
                "stability_tests": stability_tests,
                "analysis": analysis,
            }

            self.results["analysis"][spec.name] = analysis
            self.results["radar_ready"].append(self._build_radar_row(spec.name, analysis))

        print("\n✓ Evaluation complete")

    def _build_radar_row(self, system_name: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        constraints = analysis.get("constraints_analysis", {}) if isinstance(analysis, dict) else {}
        axis_summary = constraints.get("axis_summary", {}) if isinstance(constraints, dict) else {}
        basic = analysis.get("basic_metrics", {}) if isinstance(analysis, dict) else {}
        avg_div = basic.get("avg_divergences", {}) if isinstance(basic, dict) else {}

        row = {
            "system": system_name,
            "aggregate_score_mean": float(constraints.get("aggregate_summary", {}).get("mean_score", 0.0)) if isinstance(constraints, dict) else 0.0,
            "overall_violation_rate": float(constraints.get("overall_violation_rate", 0.0)) if isinstance(constraints, dict) else 0.0,
            "mean_js": float(np.mean(list(avg_div.values()))) if avg_div else 0.0,
            "num_queries": int(constraints.get("sample_size", 0)) if isinstance(constraints, dict) else 0,
        }

        for axis in [
            "evidence_concentration",
            "modality_consistency",
            "decision_boundary_proximity",
            "evidence_diversity",
            "ood_validity",
        ]:
            row[axis] = float(axis_summary.get(axis, {}).get("mean_score", 0.0))
            row[f"{axis}_violation_rate"] = float(axis_summary.get(axis, {}).get("violation_rate", 0.0))

        return row

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

        save_csv(self.output_dir / "radar_ready.csv", self.results["radar_ready"])
        save_json(self.output_dir / "combined_summary.json", self.results["analysis"])

        summary_path = self.output_dir / "summary.txt"
        self._save_summary(summary_path)
        print(f"✓ Summary saved to {summary_path}")

        self._save_per_system_breakdown()

    def _save_per_system_breakdown(self):
        breakdown = {}
        for system_name, bundle in self.results.get("runs", {}).items():
            if not isinstance(bundle, dict):
                continue
            analysis = bundle.get("analysis", {})
            constraints = analysis.get("constraints_analysis", {})
            if constraints:
                breakdown[system_name] = constraints

        if breakdown:
            breakdown_path = self.output_dir / "constraints_per_system.json"
            with open(breakdown_path, "w", encoding="utf-8") as f:
                json.dump(self._convert_to_json_serializable(breakdown), f, indent=2)
            print(f"✓ Per-system constraint summary saved to {breakdown_path}")

    # ── Column widths for tabular sections ──────────────────────────────────
    _AXIS_LABELS = {
        "evidence_concentration":      "Evidence Concentration",
        "modality_consistency":        "Modality Consistency",
        "decision_boundary_proximity": "Decision Boundary Proximity",
        "evidence_diversity":          "Evidence Diversity",
        "ood_validity":                "OOD Validity",
    }
    _SYSTEMS_ORDER = ["traditional", "concept", "countrag"]
    _SYSTEM_DISPLAY = {
        "traditional": "Traditional",
        "concept":     "Concept-KB",
        "countrag":    "CountRAG (Ours)",
    }

    def _save_summary(self, path: Path):
        runs = self.results.get("runs", {})
        # Respect canonical order when present, else use insertion order
        ordered_systems = [s for s in self._SYSTEMS_ORDER if s in runs] + \
                          [s for s in runs if s not in self._SYSTEMS_ORDER]

        W = 88  # total width

        def hr(char="="):
            return char * W + "\n"

        def col_header(systems):
            """Right-aligned system name columns (18 chars each)."""
            line = f"{'Axis / Metric':<36}"
            for s in systems:
                line += f"{self._SYSTEM_DISPLAY.get(s, s):>17}"
            return line + "\n"

        def row(label, values, fmt=".4f"):
            line = f"  {label:<34}"
            for v in values:
                if isinstance(v, float):
                    line += f"{v:>17{fmt}}"
                else:
                    line += f"{str(v):>17}"
            return line + "\n"

        def delta_row(label, values, baseline_idx=0, fmt=".4f", higher_better=True):
            """Row with Δ vs baseline in parentheses for non-baseline columns."""
            base = values[baseline_idx] if isinstance(values[baseline_idx], float) else 0.0
            line = f"  {label:<34}"
            for i, v in enumerate(values):
                if isinstance(v, float):
                    if i == baseline_idx:
                        line += f"{v:>17{fmt}}"
                    else:
                        delta = v - base
                        sign = "+" if delta >= 0 else ""
                        d_str = f"({sign}{delta:.3f})"
                        cell = f"{v:{fmt}} {d_str}"
                        line += f"{cell:>17}"
                else:
                    line += f"{str(v):>17}"
            return line + "\n"

        def viol_row(label, values, baseline_idx=0):
            """Violation-rate row: lower is better → Δ sign flipped display."""
            base = values[baseline_idx] if isinstance(values[baseline_idx], float) else 0.0
            line = f"  {label:<34}"
            for i, v in enumerate(values):
                if isinstance(v, float):
                    if i == baseline_idx:
                        pct = f"{v*100:.1f}%"
                        line += f"{pct:>17}"
                    else:
                        delta = v - base
                        sign = "+" if delta >= 0 else ""
                        pct = f"{v*100:.1f}%"
                        d_str = f"({sign}{delta*100:.1f}pp)"
                        cell = f"{pct} {d_str}"
                        line += f"{cell:>17}"
                else:
                    line += f"{str(v):>17}"
            return line + "\n"

        with open(path, "w", encoding="utf-8") as f:
            # ── Header ────────────────────────────────────────────────────────
            f.write(hr())
            f.write("THREE-STAGE CONSTRAINT EVIDENCE VIOLATION REPORT\n")
            f.write("CountRAG vs Concept-KB vs Traditional — System Comparison\n")
            f.write(hr())
            f.write(f"Timestamp   : {self.results.get('timestamp', 'N/A')}\n")
            f.write(f"Eval samples: {self.results.get('num_samples', 'N/A')}\n")
            f.write(f"Contract    : {self.contract.get('_contract_path', 'N/A')}\n")
            f.write(f"Systems     : {', '.join(self._SYSTEM_DISPLAY.get(s, s) for s in ordered_systems)}\n")
            f.write("\n")

            # ── Stage 1: Evidence Retrieval Constraints ────────────────────────
            f.write(hr())
            f.write("STAGE 1 — EVIDENCE RETRIEVAL CONSTRAINTS\n")
            f.write("Measures how well each system retrieves coherent, diverse,\n")
            f.write("and modality-consistent evidence from the knowledge base.\n")
            f.write(hr("-"))
            f.write(col_header(ordered_systems))
            f.write(hr("-"))

            stage1_axes = ["evidence_concentration", "modality_consistency", "evidence_diversity"]
            for axis in stage1_axes:
                label = self._AXIS_LABELS[axis]
                # Score rows
                scores = []
                for s in ordered_systems:
                    c = runs[s]["analysis"].get("constraints_analysis", {}) if s in runs else {}
                    scores.append(c.get("axis_summary", {}).get(axis, {}).get("mean_score", float("nan"))
                                  if isinstance(c, dict) else float("nan"))
                f.write(delta_row(f"{label} (score)", scores, higher_better=True))

                # Violation-rate rows
                viols = []
                for s in ordered_systems:
                    c = runs[s]["analysis"].get("constraints_analysis", {}) if s in runs else {}
                    viols.append(c.get("axis_summary", {}).get(axis, {}).get("violation_rate", float("nan"))
                                 if isinstance(c, dict) else float("nan"))
                f.write(viol_row(f"  └─ Violation rate", viols))

            f.write(hr("-"))
            # Stage 1 summary: mean score across the 3 axes
            for axis in stage1_axes:
                pass  # already printed above
            f.write("\n")

            # ── Stage 2: Decision Boundary & OOD Constraints ──────────────────
            f.write(hr())
            f.write("STAGE 2 — DECISION BOUNDARY & OOD CONSTRAINTS\n")
            f.write("Measures margin confidence near the decision boundary and\n")
            f.write("out-of-distribution validity of the query against the KB.\n")
            f.write(hr("-"))
            f.write(col_header(ordered_systems))
            f.write(hr("-"))

            stage2_axes = ["decision_boundary_proximity", "ood_validity"]
            for axis in stage2_axes:
                label = self._AXIS_LABELS[axis]
                scores = []
                for s in ordered_systems:
                    c = runs[s]["analysis"].get("constraints_analysis", {}) if s in runs else {}
                    scores.append(c.get("axis_summary", {}).get(axis, {}).get("mean_score", float("nan"))
                                  if isinstance(c, dict) else float("nan"))
                f.write(delta_row(f"{label} (score)", scores, higher_better=True))

                viols = []
                for s in ordered_systems:
                    c = runs[s]["analysis"].get("constraints_analysis", {}) if s in runs else {}
                    viols.append(c.get("axis_summary", {}).get(axis, {}).get("violation_rate", float("nan"))
                                 if isinstance(c, dict) else float("nan"))
                f.write(viol_row(f"  └─ Violation rate", viols))

            f.write(hr("-"))
            # JS divergence rows
            f.write(f"\n  {'Robustness (JSD — lower is better)':<34}")
            for s in ordered_systems:
                basic = runs[s]["analysis"].get("basic_metrics", {}) if s in runs else {}
                avg_div = basic.get("avg_divergences", {}) if isinstance(basic, dict) else {}
                mean_jsd = float(np.mean(list(avg_div.values()))) if avg_div else float("nan")
                f.write(f"{mean_jsd:>17.4f}")
            f.write("\n")

            for pert, label in [("no_text", "  JSD (no-text)"), ("no_image", "  JSD (no-image)"), ("noisy", "  JSD (noisy)")]:
                f.write(f"  {label:<34}")
                for s in ordered_systems:
                    basic = runs[s]["analysis"].get("basic_metrics", {}) if s in runs else {}
                    avg_div = basic.get("avg_divergences", {}) if isinstance(basic, dict) else {}
                    f.write(f"{avg_div.get(pert, float('nan')):>17.4f}")
                f.write("\n")
            f.write("\n")

            # ── Stage 3: Aggregate Constraint & Overall Violation ─────────────
            f.write(hr())
            f.write("STAGE 3 — AGGREGATE CONSTRAINT & OVERALL VIOLATION ANALYSIS\n")
            f.write("Weighted aggregate of all five constraint axes plus the\n")
            f.write("four-trigger overall violation rate (cross-links JSD probing).\n")
            f.write(hr("-"))
            f.write(col_header(ordered_systems))
            f.write(hr("-"))

            # Aggregate score
            agg_scores = []
            for s in ordered_systems:
                c = runs[s]["analysis"].get("constraints_analysis", {}) if s in runs else {}
                agg_scores.append(
                    c.get("aggregate_summary", {}).get("mean_score", float("nan"))
                    if isinstance(c, dict) else float("nan")
                )
            f.write(delta_row("Aggregate constraint score", agg_scores, higher_better=True))

            # Overall violation rate
            ovr = []
            for s in ordered_systems:
                c = runs[s]["analysis"].get("constraints_analysis", {}) if s in runs else {}
                ovr.append(c.get("overall_violation_rate", float("nan")) if isinstance(c, dict) else float("nan"))
            f.write(viol_row("Overall violation rate", ovr))

            # Robustness distribution
            f.write(hr("-"))
            for level in ["high", "medium", "low"]:
                f.write(f"  {'Robustness: ' + level:<34}")
                for s in ordered_systems:
                    basic = runs[s]["analysis"].get("basic_metrics", {}) if s in runs else {}
                    dist = basic.get("robustness_distribution", {}) if isinstance(basic, dict) else {}
                    f.write(f"{dist.get(level, 0):>17d}")
                f.write("\n")

            # All 5 axes full table
            f.write(hr("-"))
            f.write(f"  {'Per-axis violation rates (all 5 axes):'}\n")
            f.write(hr("-"))
            all_axes = list(self._AXIS_LABELS.keys())
            for axis in all_axes:
                label = self._AXIS_LABELS[axis]
                viols = []
                for s in ordered_systems:
                    c = runs[s]["analysis"].get("constraints_analysis", {}) if s in runs else {}
                    viols.append(c.get("axis_summary", {}).get(axis, {}).get("violation_rate", float("nan"))
                                 if isinstance(c, dict) else float("nan"))
                f.write(viol_row(label, viols))
            f.write("\n")

            # ── Reliability Correlations (CountRAG only, research context) ────
            countrag_constraints = {}
            if "countrag" in runs:
                countrag_constraints = runs["countrag"]["analysis"].get("constraints_analysis", {})
            if isinstance(countrag_constraints, dict) and "reliability_correlations" in countrag_constraints:
                f.write(hr())
                f.write("RELIABILITY CORRELATIONS (CountRAG)\n")
                f.write("Spearman ρ between aggregate constraint score and system signals.\n")
                f.write(hr("-"))
                corrs = countrag_constraints["reliability_correlations"]
                for key, vals in corrs.items():
                    rho = vals.get("rho", float("nan"))
                    p   = vals.get("p_value", float("nan"))
                    label = key.replace("_", " ").title()
                    f.write(f"  {label:<44} ρ={rho:+.3f}  p={p:.4f}\n")
                f.write("\n")

            # ── Radar-ready table (for copy-paste into figures) ────────────────
            f.write(hr())
            f.write("RADAR-READY TABLE  (mean constraint scores per axis, 0–1 scale)\n")
            f.write("Use this to build the radar / spider diagram directly.\n")
            f.write(hr("-"))
            f.write(col_header(ordered_systems))
            f.write(hr("-"))
            for axis in all_axes:
                label = self._AXIS_LABELS[axis]
                scores = []
                for s in ordered_systems:
                    c = runs[s]["analysis"].get("constraints_analysis", {}) if s in runs else {}
                    scores.append(c.get("axis_summary", {}).get(axis, {}).get("mean_score", float("nan"))
                                  if isinstance(c, dict) else float("nan"))
                f.write(row(label, scores))
            f.write(hr("-"))
            # Add aggregate as last row
            agg_row_vals = []
            for s in ordered_systems:
                c = runs[s]["analysis"].get("constraints_analysis", {}) if s in runs else {}
                agg_row_vals.append(
                    c.get("aggregate_summary", {}).get("mean_score", float("nan"))
                    if isinstance(c, dict) else float("nan")
                )
            f.write(row("Aggregate (weighted)", agg_row_vals))
            f.write(hr())

            # ── System metadata footer ─────────────────────────────────────────
            f.write("SYSTEM CONFIGURATION\n")
            f.write(hr("-"))
            for s in ordered_systems:
                bundle = runs.get(s, {})
                disp   = self._SYSTEM_DISPLAY.get(s, s)
                f.write(f"  {disp:<20} KB={bundle.get('kb_dir','N/A')}  "
                        f"LoRA={bundle.get('lora_path','None')}  "
                        f"mode={bundle.get('kb_mode','N/A')}\n")
            f.write(hr())