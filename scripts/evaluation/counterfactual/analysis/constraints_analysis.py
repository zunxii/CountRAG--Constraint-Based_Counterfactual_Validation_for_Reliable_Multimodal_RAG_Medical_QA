"""
Constraints analysis for counterfactual evaluation.
Produces paper-ready violation rates, summaries, and reliability correlations.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Any

import numpy as np
from scipy import stats


class ConstraintsAnalyzer:
    """Aggregate normalized constraint scores into paper-ready summaries."""

    AXES = [
        "evidence_concentration",
        "modality_consistency",
        "decision_boundary_proximity",
        "evidence_diversity",
        "ood_validity",
    ]

    def analyze(self, stability_tests: List[Dict]) -> Dict[str, Any]:
        if not stability_tests:
            return {"status": "empty"}

        rows = []
        for test in stability_tests:
            constraints = test.get("constraints", {})
            if not constraints or "scores" not in constraints:
                continue

            stab = test.get("stability", {})
            js = stab.get("js_divergence", {})
            mean_js = float(np.mean([float(v) for v in js.values()])) if js else 0.0

            scores = constraints.get("scores", {})
            violations = constraints.get("violations", {})
            rows.append({
                "diagnosis": test.get("diagnosis", "unknown"),
                "scores": scores,
                "violations": violations,
                "aggregate_score": float(constraints.get("aggregate_score", 0.0)),
                "overall_violation": bool(constraints.get("overall_violation", False)),
                "mean_js": mean_js,
                "robustness_level": stab.get("robustness_level", "unknown"),
            })

        if not rows:
            return {"status": "no_valid_constraints"}

        summary = {
            "sample_size": len(rows),
            "axis_summary": {},
            "aggregate_summary": {},
            "diagnosis_breakdown": self._diagnosis_breakdown(rows),
            "overall_violation_rate": float(np.mean([r["overall_violation"] for r in rows])),
            "reliability_correlations": self._correlations(rows),
        }

        for axis in self.AXES:
            vals = [r["scores"].get(axis, 0.0) for r in rows]
            viol = [bool(r["violations"].get(axis, False)) for r in rows]
            summary["axis_summary"][axis] = {
                "mean_score": float(np.mean(vals)),
                "std_score": float(np.std(vals)),
                "median_score": float(np.median(vals)),
                "violation_rate": float(np.mean(viol)),
                "min_score": float(np.min(vals)),
                "max_score": float(np.max(vals)),
            }

        agg = [r["aggregate_score"] for r in rows]
        summary["aggregate_summary"] = {
            "mean_score": float(np.mean(agg)),
            "std_score": float(np.std(agg)),
            "median_score": float(np.median(agg)),
            "min_score": float(np.min(agg)),
            "max_score": float(np.max(agg)),
        }
        return summary

    def _diagnosis_breakdown(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        by_diag = defaultdict(list)
        for r in rows:
            by_diag[r["diagnosis"]].append(r)

        out = {}
        for diag, items in by_diag.items():
            out[diag] = {
                "count": len(items),
                "overall_violation_rate": float(np.mean([i["overall_violation"] for i in items])),
                "aggregate_mean": float(np.mean([i["aggregate_score"] for i in items])),
                "mean_js": float(np.mean([i["mean_js"] for i in items])),
            }
            for axis in self.AXES:
                vals = [i["scores"].get(axis, 0.0) for i in items]
                out[diag][f"{axis}_mean"] = float(np.mean(vals))
                out[diag][f"{axis}_violation_rate"] = float(np.mean([bool(i["violations"].get(axis, False)) for i in items]))
        return out

    def _correlations(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        aggregate = np.asarray([r["aggregate_score"] for r in rows], dtype=np.float64)
        mean_js = np.asarray([r["mean_js"] for r in rows], dtype=np.float64)
        overall = np.asarray([1.0 if r["overall_violation"] else 0.0 for r in rows], dtype=np.float64)
        robustness_map = {"high": 2.0, "medium": 1.0, "low": 0.0}
        robustness = np.asarray([robustness_map.get(r["robustness_level"], 0.0) for r in rows], dtype=np.float64)

        def spearman(a, b):
            try:
                if np.std(a) == 0 or np.std(b) == 0:
                    return {"rho": 0.0, "p_value": 1.0}
                rho, p = stats.spearmanr(a, b)
                if not np.isfinite(rho) or not np.isfinite(p):
                    return {"rho": 0.0, "p_value": 1.0}
                return {"rho": float(rho), "p_value": float(p)}
            except Exception:
                return {"rho": 0.0, "p_value": 1.0}

        return {
            "aggregate_vs_mean_js": spearman(aggregate, mean_js),
            "aggregate_vs_overall_violation": spearman(aggregate, overall),
            "aggregate_vs_robustness_level": spearman(aggregate, robustness),
        }