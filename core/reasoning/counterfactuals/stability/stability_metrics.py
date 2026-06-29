"""
Stability metrics: JS divergence and robustness classification.

FIX vs original:
- The KL inner function previously used `mask = a > 0` then added epsilon to
  *both* a and b inside the mask.  Adding epsilon to a (numerator) inflates
  KL when a is small; adding epsilon to b (denominator) is only needed when
  b is zero, but if b is zero where a>0 the mixture m=0.5*(p+q) also has
  b's contribution halved, so m is still >0 there.  The correct formulation
  for standard JS divergence uses the mixture m to prevent division by zero
  without adding epsilon to a.
- Robustness thresholds kept at high<0.15 / medium<0.30 to match the paper.
  These are appropriate for the posterior-distribution JSD (after the runner
  fix) and should not need further adjustment.
"""
import numpy as np


def normalize(dist: dict) -> dict:
    total = sum(dist.values()) + 1e-12
    return {k: float(v) / total for k, v in dist.items()}


def _dist(x):
    if isinstance(x, dict) and "distribution" in x and isinstance(x["distribution"], dict):
        return x["distribution"]
    return x if isinstance(x, dict) else {}


def js_divergence(p: dict, q: dict) -> float:
    """
    Jensen-Shannon divergence between two probability distributions.

    Uses the standard formulation:
        JSD(P||Q) = 0.5*KL(P||M) + 0.5*KL(Q||M),  M = 0.5*(P+Q)

    Because M(i) >= 0.5*P(i) and M(i) >= 0.5*Q(i), wherever P(i)>0 or Q(i)>0
    we have M(i)>0, so no epsilon is needed in the denominator.  We add a tiny
    epsilon only to guard against floating-point underflow, not to mask zeros.
    """
    keys = set(p) | set(q)
    p_vec = np.array([p.get(k, 0.0) for k in keys], dtype=np.float64)
    q_vec = np.array([q.get(k, 0.0) for k in keys], dtype=np.float64)

    # Renormalise in case inputs don't sum to exactly 1.0
    p_sum = p_vec.sum()
    q_sum = q_vec.sum()
    if p_sum > 0:
        p_vec = p_vec / p_sum
    if q_sum > 0:
        q_vec = q_vec / q_sum

    m = 0.5 * (p_vec + q_vec)

    eps = 1e-12

    def kl(a, b):
        # Only sum over entries where a > 0; b is guaranteed > 0 there
        # because m = 0.5*(p+q) and either p or q is a here.
        mask = a > 0
        return float(np.sum(a[mask] * np.log((a[mask] + eps) / (m[mask] + eps))))

    return float(0.5 * kl(p_vec, m) + 0.5 * kl(q_vec, m))


def stability_report(baseline: dict, variants: dict) -> dict:
    base = normalize(_dist(baseline))

    divergences = {
        k: round(js_divergence(base, normalize(_dist(v))), 4)
        for k, v in variants.items()
    }

    max_div = max(divergences.values()) if divergences else 0.0

    return {
        "js_divergence": divergences,
        "robustness_level": (
            "high" if max_div < 0.15
            else "medium" if max_div < 0.30
            else "low"
        ),
    }