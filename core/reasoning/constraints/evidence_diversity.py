from typing import Dict

def evidence_diversity_constraint(retrieved_metadata: list) -> Dict:
    top = retrieved_metadata[0] if retrieved_metadata else {}
    support = 0

    if isinstance(top, dict):
        if "num_images" in top:
            try:
                support = int(top["num_images"])
            except Exception:
                support = 0
        elif isinstance(top.get("image_paths"), list):
            support = len([p for p in top["image_paths"] if p])

    if support < 10:
        level = "low"
    elif support < 40:
        level = "medium"
    else:
        level = "high"

    return {
        "level": level,
        "prototype_support": int(support),
        "unique_cases": int(support),
    }