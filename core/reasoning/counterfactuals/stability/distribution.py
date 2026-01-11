from collections import Counter

def cluster_distribution(retrieved: list) -> dict:
    """
    Build diagnosis distribution from retrieved results
    
    Works for both:
    - Flat KB: metadata has 'diagnosis_label'
    - Concept KB: metadata has 'diagnosis_label' 
    
    Both structures have diagnosis_label, so no change needed!
    """
    labels = [r["diagnosis_label"] for r in retrieved]
    counts = Counter(labels)
    total = sum(counts.values()) + 1e-9

    return {
        "distribution": {k: round(v / total, 4) for k, v in counts.items()},
        "num_clusters": len(counts),
    }