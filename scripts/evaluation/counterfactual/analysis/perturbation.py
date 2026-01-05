import numpy as np
from typing import Dict, List


class PerturbationAnalyzer:
    """Analyzes stability across multiple perturbation scales"""
    
    def __init__(self, scales: List[float] = None):
        self.scales = scales or [0.01, 0.05, 0.1, 0.15, 0.2]
    
    def analyze(self, stability_tests: List[Dict]) -> Dict:
        """Analyze stability across perturbation scales"""
        results = {
            "scale_analysis": {},
            "degradation_curve": [],
            "robustness_threshold": None
        }
        
        by_scale = self._group_by_scale(stability_tests)
        
        for scale, tests in by_scale.items():
            divergences = [t["stability"]["js_divergence"].get("noisy", 0.0) 
                        for t in tests]
            
            results["scale_analysis"][str(scale)] = {  # Convert scale to string for JSON
                "mean_divergence": float(np.mean(divergences)),
                "std_divergence": float(np.std(divergences)),
                "median_divergence": float(np.median(divergences)),
                "max_divergence": float(np.max(divergences)),
                "stable_ratio": float(np.mean([d < 0.15 for d in divergences]))
            }
            
            results["degradation_curve"].append({
                "scale": float(scale) if isinstance(scale, (int, float, np.number)) else str(scale),
                "divergence": float(np.mean(divergences))
            })
        
        results["robustness_threshold"] = self._find_threshold(
            results["degradation_curve"]
        )
        
        return results
    def _group_by_scale(self, tests: List[Dict]) -> Dict:
        """Group tests by perturbation scale"""
        return {"default": tests}
    
    def _find_threshold(self, curve: List[Dict]) -> float:
        """Find scale where stability starts to degrade"""
        if not curve:
            return 0.0
        
        for point in curve:
            if point["divergence"] > 0.15:
                return point["scale"]
        
        return curve[-1]["scale"] if curve else 0.0

