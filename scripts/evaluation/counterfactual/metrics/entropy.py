import numpy as np
from typing import Dict


def shannon_entropy(distribution: Dict[str, float]) -> float:
    """Shannon entropy of probability distribution"""
    if not distribution:
        return 0.0
    probs = np.array(list(distribution.values()))
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def normalized_entropy(distribution: Dict[str, float]) -> float:
    """Normalized entropy (0 to 1)"""
    n = len(distribution)
    if n <= 1:
        return 0.0
    max_entropy = np.log2(n)
    return shannon_entropy(distribution) / max_entropy if max_entropy > 0 else 0.0


def effective_hypotheses(distribution: Dict[str, float]) -> float:
    """Effective number of hypotheses (inverse participation ratio)"""
    probs = list(distribution.values())
    if not probs:
        return 0.0
    return 1.0 / sum(p**2 for p in probs if p > 0)


def peak_confidence(distribution: Dict[str, float]) -> float:
    """Maximum probability (confidence in top prediction)"""
    return max(distribution.values()) if distribution else 0.0


def entropy_gap(baseline: Dict[str, float], perturbed: Dict[str, float]) -> float:
    """Change in entropy under perturbation"""
    return abs(shannon_entropy(perturbed) - shannon_entropy(baseline))
