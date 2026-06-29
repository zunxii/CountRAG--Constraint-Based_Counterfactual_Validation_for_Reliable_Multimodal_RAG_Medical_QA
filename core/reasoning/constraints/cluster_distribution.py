from collections import Counter
from typing import Dict

def cluster_distribution_constraint(retrieved_metadata: list) -> Dict:
    labels = [m.get("diagnosis_label", "unknown") for m in retrieved_metadata if isinstance(m, dict)]
    counts = Counter(labels)
    total = sum(counts.values()) or 1

    distribution = {label: round(count / total, 4) for label, count in counts.items()}

    return {
        "distribution": distribution,
        "num_clusters": len(distribution),
    }