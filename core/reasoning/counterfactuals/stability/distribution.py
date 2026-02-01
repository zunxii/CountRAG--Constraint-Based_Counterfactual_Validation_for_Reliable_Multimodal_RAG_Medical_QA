from collections import Counter
import numpy as np

def softmax(x, temperature=1.0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum(axis=0)

def cluster_distribution(retrieved: list) -> dict:
    """
    Build diagnosis distribution from retrieved results
    
    Works for both:
    - Flat KB: metadata has 'diagnosis_label'
    - Concept KB: metadata has 'diagnosis_label' 
    """
    labels = [r["diagnosis_label"] for r in retrieved]
    
    # Check if scores are available
    if retrieved and "score" in retrieved[0]:
        # 1. Aggregate scores by label (taking max score per label if duplicates exist)
        # In Concept KB, labels are unique, but good to be robust.
        label_scores = {}
        for r in retrieved:
            lbl = r["diagnosis_label"]
            score = r["score"]
            if lbl not in label_scores or score > label_scores[lbl]:
                label_scores[lbl] = score
        
        # 2. Prepare for Softmax
        unique_labels = list(label_scores.keys())
        scores = np.array([label_scores[l] for l in unique_labels])
        
        # 3. Apply Softmax with Temperature (Sharpening)
        # Low temperature (<1.0) makes distribution sharper (peakier)
        temperature = 0.01 
        probs = softmax(scores, temperature=temperature)
        
        # 4. Create sorted distribution
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        sorted_labels = [unique_labels[i] for i in sorted_indices]
        
        # 5. Dynamic Filtering (Nucleus Sampling / Top-P)
        # Keep cumulative probability mass up to 0.90
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff_index = np.searchsorted(cumulative_probs, 0.90) + 1
        # Ensure at least 1 remains
        cutoff_index = max(1, min(cutoff_index, len(sorted_labels)))
        
        final_labels = sorted_labels[:cutoff_index]
        final_probs = sorted_probs[:cutoff_index]
        
        # Re-normalize the filtered set
        final_probs = final_probs / final_probs.sum()
        
        distribution = {l: round(float(p), 4) for l, p in zip(final_labels, final_probs)}
        num_clusters = len(distribution)
            
    else:
        # Fallback to simple counting
        counts = Counter(labels)
        total = sum(counts.values()) + 1e-9
        distribution = {k: round(v / total, 4) for k, v in counts.items()}
        num_clusters = len(counts)

    return {
        "distribution": distribution,
        "num_clusters": num_clusters,
    }