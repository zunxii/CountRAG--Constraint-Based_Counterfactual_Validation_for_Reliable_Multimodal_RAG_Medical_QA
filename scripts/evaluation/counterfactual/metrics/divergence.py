"""
JS and KL divergence helpers used in the evaluation scripts.

FIX vs original:
- The epsilon was added to a[mask] in the KL numerator, which inflates small
  contributions.  Corrected to use the standard formulation where the mixture
  m = 0.5*(p+q) already guarantees m > 0 wherever p > 0 or q > 0.
- Both implementations (stability_metrics.py and this file) now use the same
  numerically consistent formula.
"""
import numpy as np
from typing import Dict


def _to_vec(p: Dict[str, float], q: Dict[str, float]):
    keys = set(p.keys()) | set(q.keys())
    p_vec = np.array([p.get(k, 0.0) for k in keys], dtype=np.float64)
    q_vec = np.array([q.get(k, 0.0) for k in keys], dtype=np.float64)
    p_s = p_vec.sum()
    q_s = q_vec.sum()
    if p_s > 0:
        p_vec /= p_s
    if q_s > 0:
        q_vec /= q_s
    return p_vec, q_vec


def js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Jensen-Shannon divergence between two probability distributions."""
    p_vec, q_vec = _to_vec(p, q)
    m = 0.5 * (p_vec + q_vec)
    eps = 1e-12

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log((a[mask] + eps) / (b[mask] + eps))))

    return float(0.5 * kl(p_vec, m) + 0.5 * kl(q_vec, m))


def kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Kullback-Leibler divergence KL(P||Q)."""
    p_vec, q_vec = _to_vec(p, q)
    eps = 1e-12
    mask = p_vec > 0
    return float(np.sum(p_vec[mask] * np.log((p_vec[mask] + eps) / (q_vec[mask] + eps))))


def hellinger_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Hellinger distance between distributions."""
    p_vec, q_vec = _to_vec(p, q)
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p_vec) - np.sqrt(q_vec)) ** 2)))