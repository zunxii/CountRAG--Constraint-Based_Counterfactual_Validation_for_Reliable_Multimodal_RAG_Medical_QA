from typing import Dict

def distribution_check_constraint(query_distance: float, percentile_95: float) -> Dict:
    return {
        "in_distribution": bool(query_distance <= percentile_95),
        "distance": round(float(query_distance), 4),
        "threshold": round(float(percentile_95), 4),
    }