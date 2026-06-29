from __future__ import annotations

import numpy as np
from typing import Dict, List


class PerturbationAnalyzer:
    """Analyzes stability across multiple perturbation scales."""

    def __init__(self, scales: List[float] | None = None):
        self.scales = scales or [0.01, 0.05, 0.1, 0.15, 0.2]

    def analyze(self, stability_tests: List[Dict]) -> Dict:
        results = {
            "scale_analysis": {},
            "degradation_curve": [],
            "robustness_threshold": None,
        }

        by_scale = self._group_by_scale(stability_tests)

        for scale, tests in by_scale.items():
            divergences = []
            for t in tests:
                js_by_scale = t.get("stability", {}).get("js_divergence_by_scale", {})
                if js_by_scale:
                    val = js_by_scale.get(str(scale))
                    if val is not None:
                        divergences.append(float(val))
                    else:
                        divergences.append(float(t.get("stability", {}).get("js_divergence", {}).get("noisy", 0.0)))
                else:
                    divergences.append(float(t.get("stability", {}).get("js_divergence", {}).get("noisy", 0.0)))

            if not divergences:
                continue

            results["scale_analysis"][str(scale)] = {
                "mean_divergence": float(np.mean(divergences)),
                "std_divergence": float(np.std(divergences)),
                "median_divergence": float(np.median(divergences)),
                "max_divergence": float(np.max(divergences)),
                "stable_ratio": float(np.mean([d < 0.15 for d in divergences])),
            }

            results["degradation_curve"].append({
                "scale": float(scale) if isinstance(scale, (int, float, np.number)) else str(scale),
                "divergence": float(np.mean(divergences)),
            })

        results["degradation_curve"] = sorted(results["degradation_curve"], key=lambda x: x["scale"])
        results["robustness_threshold"] = self._find_threshold(results["degradation_curve"])

        return results

    def _group_by_scale(self, tests: List[Dict]) -> Dict:
        grouped = {}
        for test in tests:
            js_by_scale = test.get("stability", {}).get("js_divergence_by_scale", {})
            if js_by_scale:
                for scale_str in js_by_scale.keys():
                    try:
                        scale = float(scale_str)
                    except Exception:
                        scale = scale_str
                    grouped.setdefault(scale, []).append(test)
            else:
                grouped.setdefault("default", []).append(test)

        if not grouped:
            grouped = {"default": tests}
        return grouped

    def _find_threshold(self, curve: List[Dict]) -> float:
        if not curve:
            return 0.0

        for point in curve:
            if point["divergence"] > 0.15:
                return point["scale"]

        return curve[-1]["scale"]