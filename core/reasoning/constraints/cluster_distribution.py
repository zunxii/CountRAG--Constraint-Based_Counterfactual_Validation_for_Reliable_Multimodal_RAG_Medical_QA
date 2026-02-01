from collections import Counter
from typing import Dict
import numpy as np

def softmax(x, temperature=1.0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum(axis=0)

def cluster_distribution_constraint(retrieved_metadata: list) -> Dict:
    """
    Returns soft distribution over diagnosis clusters.
    Uses Softmax normalization (T=0.1) and Top-P filtering (0.95).
    """
    
    # Check if scores are available
    if retrieved_metadata and "score" in retrieved_metadata[0]:
        # 1. Aggregate scores by label
        label_scores = {}
        for r in retrieved_metadata:
            lbl = r["diagnosis_label"]
            score = r["score"]
            if lbl not in label_scores or score > label_scores[lbl]:
                label_scores[lbl] = score
        
        # 2. Prepare for Softmax
        unique_labels = list(label_scores.keys())
        scores = np.array([label_scores[l] for l in unique_labels])
        
        # 3. Apply Softmax with Temperature (Sharpening)
        temperature = 0.01 
        probs = softmax(scores, temperature=temperature)
        
        # 4. Create sorted distribution
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        sorted_labels = [unique_labels[i] for i in sorted_indices]
        
        # 5. Dynamic Filtering (Nucleus Sampling / Top-P)
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff_index = np.searchsorted(cumulative_probs, 0.90) + 1
        cutoff_index = max(1, min(cutoff_index, len(sorted_labels)))
        
        final_labels = sorted_labels[:cutoff_index]
        final_probs = sorted_probs[:cutoff_index]
        
        final_probs = final_probs / final_probs.sum()
        
        distribution = {l: round(float(p), 4) for l, p in zip(final_labels, final_probs)}
        num_clusters = len(distribution)
        
    else:
        # Fallback to simple counting
        labels = [m["diagnosis_label"] for m in retrieved_metadata]
        counts = Counter(labels)
        total = sum(counts.values()) or 1
        distribution = {
            label: round(count / total, 4)
            for label, count in counts.items()
        }
        num_clusters = len(distribution)

    return {
        "distribution": distribution,
        "num_clusters": num_clusters,
    }
