import numpy as np
from typing import Dict


def js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Jensen-Shannon divergence between two probability distributions"""
    keys = set(p.keys()) | set(q.keys())
    p_vec = np.array([p.get(k, 0.0) for k in keys])
    q_vec = np.array([q.get(k, 0.0) for k in keys])
    
    p_vec = p_vec / (p_vec.sum() + 1e-9)
    q_vec = q_vec / (q_vec.sum() + 1e-9)
    
    m = 0.5 * (p_vec + q_vec)
    
    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log((a[mask] + 1e-9) / (b[mask] + 1e-9)))
    
    return float(0.5 * kl(p_vec, m) + 0.5 * kl(q_vec, m))


def kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Kullback-Leibler divergence from q to p"""
    keys = set(p.keys()) | set(q.keys())
    p_vec = np.array([p.get(k, 0.0) for k in keys])
    q_vec = np.array([q.get(k, 0.0) for k in keys])
    
    p_vec = p_vec / (p_vec.sum() + 1e-9)
    q_vec = q_vec / (q_vec.sum() + 1e-9)
    
    mask = p_vec > 0
    return float(np.sum(p_vec[mask] * np.log((p_vec[mask] + 1e-9) / (q_vec[mask] + 1e-9))))


def hellinger_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Hellinger distance between distributions"""
    keys = set(p.keys()) | set(q.keys())
    p_vec = np.array([p.get(k, 0.0) for k in keys])
    q_vec = np.array([q.get(k, 0.0) for k in keys])
    
    p_vec = p_vec / (p_vec.sum() + 1e-9)
    q_vec = q_vec / (q_vec.sum() + 1e-9)
    
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p_vec) - np.sqrt(q_vec))**2)))
