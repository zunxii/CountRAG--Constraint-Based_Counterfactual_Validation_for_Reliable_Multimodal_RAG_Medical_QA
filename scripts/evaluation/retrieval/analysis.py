"""
Result analysis and insights
"""
from typing import Dict, Any


class ResultsAnalyzer:
    """
    Analyze retrieval evaluation outputs.

    This version is schema-tolerant:
    - supports nested inputs: run_name -> mode_name -> result
    - supports flat inputs: mode_name -> result
    - supports metrics stored either at result["metrics"] or at top level
    - supports Entropy/Margin or entropy/margin key variants
    """

    PAPER_METRICS = ["R@1", "R@5", "R@10", "MRR", "MAP", "Entropy", "Margin"]

    def analyze(self, run_results: Dict[str, Any]) -> Dict[str, Any]:
        analysis: Dict[str, Any] = {
            "per_run": {},
            "global_comparison": {},
            "global_best": {},
        }

        if not isinstance(run_results, dict) or not run_results:
            return analysis

        # Detect nested structure: run_name -> mode_name -> result
        is_nested = False
        for v in run_results.values():
            if isinstance(v, dict) and v and all(isinstance(x, dict) for x in v.values()):
                is_nested = True
                break

        if is_nested:
            for run_name, mode_results in run_results.items():
                if not isinstance(mode_results, dict):
                    continue

                analysis["per_run"][run_name] = {
                    "mode_comparison": self._compare_modes(mode_results),
                    "best_modes": self._find_best_modes(mode_results),
                }

            flattened = {}
            for run_name, mode_results in run_results.items():
                if not isinstance(mode_results, dict):
                    continue

                for mode_name, result in mode_results.items():
                    if isinstance(result, dict):
                        flattened[f"{run_name}/{mode_name}"] = result

            analysis["global_comparison"] = self._compare_modes(flattened)
            analysis["global_best"] = self._find_best_modes(flattened)
        else:
            analysis["mode_comparison"] = self._compare_modes(run_results)
            analysis["best_modes"] = self._find_best_modes(run_results)

        return analysis

    def _safe_metrics(self, result: Dict[str, Any]) -> Dict[str, float]:
        if not isinstance(result, dict):
            return {}

        metrics = result.get("metrics", result)
        if not isinstance(metrics, dict):
            return {}

        return metrics

    def _metric_value(self, metrics: Dict[str, Any], metric: str) -> float:
        """
        Read a metric in a tolerant way.

        Supports:
        - exact metric key
        - lowercase variant for entropy/margin
        - common aliases where needed
        """
        if metric in metrics:
            value = metrics[metric]
            return value if isinstance(value, (int, float)) else 0.0

        lower = metric.lower()
        if lower in metrics:
            value = metrics[lower]
            return value if isinstance(value, (int, float)) else 0.0

        # Common aliases if upstream code uses slightly different names.
        aliases = {
            "Entropy": ["entropy", "retrieval_entropy", "H"],
            "Margin": ["margin", "top1_margin", "delta"],
            "MAP": ["map"],
            "MRR": ["mrr"],
            "R@1": ["r1", "recall@1"],
            "R@5": ["r5", "recall@5"],
            "R@10": ["r10", "recall@10"],
        }

        for alias in aliases.get(metric, []):
            if alias in metrics:
                value = metrics[alias]
                return value if isinstance(value, (int, float)) else 0.0

        return 0.0

    def _compare_modes(self, mode_results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        comparison: Dict[str, Dict[str, float]] = {}

        if not isinstance(mode_results, dict):
            return comparison

        for metric in self.PAPER_METRICS:
            comparison[metric] = {}
            for mode, result in mode_results.items():
                metrics = self._safe_metrics(result)
                comparison[metric][mode] = self._metric_value(metrics, metric)

        return comparison

    def _find_best_modes(self, mode_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        best_modes: Dict[str, Dict[str, Any]] = {}

        if not isinstance(mode_results, dict) or not mode_results:
            return best_modes

        for metric in self.PAPER_METRICS:
            values = {}

            for mode, result in mode_results.items():
                metrics = self._safe_metrics(result)
                values[mode] = self._metric_value(metrics, metric)

            if not values:
                continue

            # Lower is better for entropy, higher is better for everything else.
            best_mode = min(values, key=values.get) if metric == "Entropy" else max(values, key=values.get)

            best_modes[metric] = {
                "mode": best_mode,
                "value": values[best_mode],
            }

        return best_modes