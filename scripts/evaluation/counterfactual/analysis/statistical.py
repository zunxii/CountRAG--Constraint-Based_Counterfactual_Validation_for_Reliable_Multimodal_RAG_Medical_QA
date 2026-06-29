from __future__ import annotations

import numpy as np
from typing import Dict, List
from scipy import stats


class StatisticalAnalyzer:
    """Statistical hypothesis testing for stability patterns."""

    def analyze(self, stability_tests: List[Dict]) -> Dict:
        no_text = []
        no_image = []
        noisy = []

        for test in stability_tests:
            stab = test.get("stability", {})
            js_div = stab.get("js_divergence", {})

            no_text.append(float(js_div.get("no_text", 0.0)))
            no_image.append(float(js_div.get("no_image", 0.0)))
            noisy.append(float(js_div.get("noisy", 0.0)))

        if len(no_text) < 2:
            return {"status": "insufficient_samples"}

        return {
            "text_modality_test": self._test_against_zero(no_text, "text"),
            "image_modality_test": self._test_against_zero(no_image, "image"),
            "noise_robustness_test": self._test_against_zero(noisy, "noise"),
            "comparative_tests": self._comparative_tests(no_text, no_image),
            "sample_statistics": self._sample_statistics(no_text, no_image, noisy),
        }

    def _test_against_zero(self, values: List[float], label: str) -> Dict:
        arr = np.asarray(values, dtype=np.float64)
        t_stat, t_pval = stats.ttest_1samp(arr, popmean=0.0, nan_policy="omit")
        try:
            w_stat, w_pval = stats.wilcoxon(arr - 0.0, zero_method="wilcox", correction=False)
        except Exception:
            w_stat, w_pval = float("nan"), float("nan")
        cohens_d = self._cohens_d_against_zero(arr)
        ci_lower, ci_upper = self._bootstrap_ci(arr)
        return {
            "label": label,
            "t_statistic": float(t_stat) if np.isfinite(t_stat) else 0.0,
            "t_pvalue": float(t_pval) if np.isfinite(t_pval) else 1.0,
            "significant": bool(np.isfinite(t_pval) and t_pval < 0.05),
            "wilcoxon_statistic": float(w_stat) if np.isfinite(w_stat) else 0.0,
            "wilcoxon_pvalue": float(w_pval) if np.isfinite(w_pval) else 1.0,
            "cohens_d": float(cohens_d),
            "effect_size_interpretation": self._interpret_effect_size(cohens_d),
            "mean_divergence": float(np.mean(arr)),
            "confidence_interval_95": {"lower": float(ci_lower), "upper": float(ci_upper)},
        }

    def _comparative_tests(self, no_text: List[float], no_image: List[float]) -> Dict:
        a = np.asarray(no_text, dtype=np.float64)
        b = np.asarray(no_image, dtype=np.float64)
        t_stat, t_pval = stats.ttest_rel(a, b, nan_policy="omit")
        try:
            u_stat, u_pval = stats.mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            u_stat, u_pval = float("nan"), float("nan")

        return {
            "text_vs_image_ttest": {
                "t_statistic": float(t_stat) if np.isfinite(t_stat) else 0.0,
                "p_value": float(t_pval) if np.isfinite(t_pval) else 1.0,
                "more_important": "text" if np.mean(a) > np.mean(b) else "image",
            },
            "mann_whitney_u": {
                "u_statistic": float(u_stat) if np.isfinite(u_stat) else 0.0,
                "p_value": float(u_pval) if np.isfinite(u_pval) else 1.0,
            },
            "paired_mean_difference": float(np.mean(a - b)),
        }

    def _sample_statistics(self, no_text: List[float], no_image: List[float], noisy: List[float]) -> Dict:
        return {
            "sample_size": len(no_text),
            "no_text": self._descriptive_stats(no_text),
            "no_image": self._descriptive_stats(no_image),
            "noisy": self._descriptive_stats(noisy),
        }

    def _bootstrap_ci(self, values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
        if values.size == 0:
            return 0.0, 0.0
        rng = np.random.default_rng(42)
        means = []
        for _ in range(n_boot):
            sample = rng.choice(values, size=values.size, replace=True)
            means.append(float(np.mean(sample)))
        lower = float(np.quantile(means, alpha / 2))
        upper = float(np.quantile(means, 1 - alpha / 2))
        return lower, upper

    def _cohens_d_against_zero(self, x: np.ndarray) -> float:
        if x.size < 2:
            return 0.0
        std = float(np.std(x, ddof=1))
        if std <= 1e-12:
            return 0.0
        return float(np.mean(x) / std)

    def _descriptive_stats(self, values: List[float]) -> Dict:
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)) if arr.size else 0.0,
            "std": float(np.std(arr)) if arr.size else 0.0,
            "median": float(np.median(arr)) if arr.size else 0.0,
            "min": float(np.min(arr)) if arr.size else 0.0,
            "max": float(np.max(arr)) if arr.size else 0.0,
        }

    def _interpret_effect_size(self, d: float) -> str:
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        return "large"