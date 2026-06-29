from typing import Dict

def boundary_analysis_constraint(top1_prob: float, top2_prob: float) -> Dict:
    margin = float(top1_prob - top2_prob)
    return {
        "near_boundary": bool(margin < 0.05),
        "margin": round(margin, 4),
        "top_probs": [round(float(top1_prob), 4), round(float(top2_prob), 4)],
    }