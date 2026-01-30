import faiss
import numpy as np

class StabilityRetriever:
    def __init__(self, index, metadata):
        self.index = index
        self.metadata = metadata

    def retrieve(self, emb, top_k=10):
        """
        Retrieve top-K results
        
        Works for both flat and concept KB - just returns metadata
        """
        q = emb.cpu().numpy().astype("float32")
        
        # Normalize query
        faiss.normalize_L2(q)
        
        # Search
        scores, idxs = self.index.search(q, top_k)
        
        # Build results with bounds checking
        results = []
        for s, i in zip(scores[0], idxs[0]):
            # Skip invalid indices
            if i < 0 or i >= len(self.metadata):
                continue
            
            results.append({
                "diagnosis_label": self.metadata[i]["diagnosis_label"],
                "image_path": self.metadata[i].get("image_path"),
                "case_id": self.metadata[i].get("case_id"),
                "score": float(s),
                "embedding": self.index.reconstruct(int(i)).tolist()
            })
        
        return results