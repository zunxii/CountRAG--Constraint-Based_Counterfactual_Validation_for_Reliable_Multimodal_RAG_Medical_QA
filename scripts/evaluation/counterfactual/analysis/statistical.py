import numpy as np
from typing import Dict, List, Tuple
from scipy import stats


class StatisticalAnalyzer:
    """Statistical hypothesis testing for stability patterns"""
    
    def analyze(self, stability_tests: List[Dict]) -> Dict:
        """Run statistical tests on stability patterns"""
        baseline = []
        no_text = []
        no_image = []
        noisy = []
        
        for test in stability_tests:
            stab = test.get("stability", {})
            js_div = stab.get("js_divergence", {})
            
            baseline.append(0.0)
            no_text.append(js_div.get("no_text", 0.0))
            no_image.append(js_div.get("no_image", 0.0))
            noisy.append(js_div.get("noisy", 0.0))
        
        if len(baseline) < 2:
            return {"status": "insufficient_samples"}
        
        return {
            "text_modality_test": self._test_modality_effect(baseline, no_text, "text"),
            "image_modality_test": self._test_modality_effect(baseline, no_image, "image"),
            "noise_robustness_test": self._test_modality_effect(baseline, noisy, "noise"),
            "comparative_tests": self._comparative_tests(no_text, no_image),
            "sample_statistics": self._sample_statistics(baseline, no_text, no_image, noisy)
        }
    
    def _test_modality_effect(self, baseline: List[float], 
                         perturbed: List[float], 
                         modality: str) -> Dict:
        """Test if modality removal has significant effect"""
        t_stat, t_pval = stats.ttest_rel(baseline, perturbed)
        w_stat, w_pval = stats.wilcoxon(perturbed)
        cohens_d = self._cohens_d(baseline, perturbed)
        ci_lower, ci_upper = self._confidence_interval(perturbed)
        
        return {
            "modality": modality,
            "t_statistic": float(t_stat),
            "t_pvalue": float(t_pval),
            "significant": bool(t_pval < 0.05),  # Convert to Python bool
            "wilcoxon_statistic": float(w_stat),
            "wilcoxon_pvalue": float(w_pval),
            "cohens_d": float(cohens_d),
            "effect_size_interpretation": self._interpret_effect_size(cohens_d),
            "mean_divergence": float(np.mean(perturbed)),
            "confidence_interval_95": {
                "lower": float(ci_lower),
                "upper": float(ci_upper)
            }
        }

    def _comparative_tests(self, no_text: List[float], 
                          no_image: List[float]) -> Dict:
        """Compare text vs image modality effects"""
        t_stat, t_pval = stats.ttest_rel(no_text, no_image)
        u_stat, u_pval = stats.mannwhitneyu(no_text, no_image)
        
        return {
            "text_vs_image_ttest": {
                "t_statistic": float(t_stat),
                "p_value": float(t_pval),
                "more_important": "text" if np.mean(no_text) > np.mean(no_image) else "image"
            },
            "mann_whitney_u": {
                "u_statistic": float(u_stat),
                "p_value": float(u_pval)
            }
        }
    
    def _sample_statistics(self, baseline, no_text, no_image, noisy) -> Dict:
        """Descriptive statistics"""
        return {
            "sample_size": len(baseline),
            "no_text": self._descriptive_stats(no_text),
            "no_image": self._descriptive_stats(no_image),
            "noisy": self._descriptive_stats(noisy)
        }
    
    def _cohens_d(self, x: List[float], y: List[float]) -> float:
        """Cohen's d effect size"""
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        return (np.mean(x) - np.mean(y)) / np.sqrt(
            ((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof
        )
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d"""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _confidence_interval(self, data: List[float], confidence=0.95) -> Tuple[float, float]:
        """95% confidence interval"""
        n = len(data)
        mean = np.mean(data)
        se = stats.sem(data)
        margin = se * stats.t.ppf((1 + confidence) / 2, n - 1)
        return mean - margin, mean + margin
    
    def _descriptive_stats(self, data: List[float]) -> Dict:
        """Descriptive statistics"""
        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "median": float(np.median(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "q25": float(np.percentile(data, 25)),
            "q75": float(np.percentile(data, 75))
        }
