import numpy as np
from typing import Dict


def confidence_metrics(distribution: Dict[str, float]) -> Dict:
    """Comprehensive confidence analysis"""
    if not distribution:
        return {"peak": 0.0, "margin": 0.0, "spread": 0.0}
    
    sorted_probs = sorted(distribution.values(), reverse=True)
    
    return {
        "peak_confidence": sorted_probs[0],
        "margin": sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0],
        "top3_mass": sum(sorted_probs[:3]) if len(sorted_probs) >= 3 else sum(sorted_probs),
        "spread": float(np.std(list(distribution.values())))
    }


def stability_confidence(baseline: Dict, perturbed_variants: Dict) -> float:
    """Confidence that predictions are stable (0-1 where 1=highly stable)"""
    from scripts.evaluation.counterfactual.metrics.divergence import js_divergence
    
    divergences = []
    baseline_dist = baseline["distribution"]
    
    for variant in perturbed_variants.values():
        div = js_divergence(baseline_dist, variant["distribution"])
        divergences.append(div)
    
    avg_div = np.mean(divergences)
    return float(1.0 / (1.0 + avg_div))

