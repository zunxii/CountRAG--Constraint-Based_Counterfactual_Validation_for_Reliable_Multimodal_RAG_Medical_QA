import faiss
import json
import numpy as np
from pathlib import Path


class KBRetriever:
    """
    Knowledge Base Retriever with distinct results guarantee
    """
    
    def __init__(self, kb_dir: str):
        kb_dir = Path(kb_dir)

        self.embeddings = np.load(kb_dir / "embeddings.npy")
        self.index = faiss.read_index(str(kb_dir / "index.faiss"))

        with open(kb_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        assert self.index.ntotal == len(self.metadata), \
            f"Index size ({self.index.ntotal}) != metadata size ({len(self.metadata)})"
    
    def search(self, query_embedding: np.ndarray, top_k: int, exclude_indices=None):
        """
        Search for top-K most similar entries.
        
        Args:
            query_embedding: Query embedding (1, D) or (D,)
            top_k: Number of results to return
            exclude_indices: List/set of indices to exclude (e.g., self-match)
        
        Returns:
            scores: Similarity scores (1, K)
            indices: Result indices (1, K)
        """
        # Normalize query
        query_embedding = query_embedding.astype("float32")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        faiss.normalize_L2(query_embedding)
        
        # Convert exclude_indices to set for fast lookup
        exclude_set = set(exclude_indices) if exclude_indices is not None else set()
        
        # Retrieve more than needed to account for exclusions
        # Fetch up to 2*top_k to ensure we have enough after filtering
        fetch_k = min(top_k * 2 + len(exclude_set), self.index.ntotal)
        scores, indices = self.index.search(query_embedding, fetch_k)
        
        # Filter out excluded indices and ensure distinctness
        filtered_scores = []
        filtered_indices = []
        seen_cases = set()  # Track by case_id to avoid duplicates
        
        for score, idx in zip(scores[0], indices[0]):
            idx = int(idx)
            
            # Skip if in exclude list
            if idx in exclude_set:
                continue
            
            # Skip if we've already seen this case (ensure distinctness)
            case_id = self.metadata[idx].get("case_id", f"case_{idx}")
            if case_id in seen_cases:
                continue
            
            seen_cases.add(case_id)
            filtered_scores.append(score)
            filtered_indices.append(idx)
            
            # Stop when we have enough
            if len(filtered_indices) >= top_k:
                break
        
        # Convert back to numpy arrays with shape (1, K)
        result_scores = np.array(filtered_scores[:top_k]).reshape(1, -1)
        result_indices = np.array(filtered_indices[:top_k]).reshape(1, -1)
        
        # Pad with -1 if we don't have enough results
        if result_indices.shape[1] < top_k:
            padding_needed = top_k - result_indices.shape[1]
            result_scores = np.pad(result_scores, ((0, 0), (0, padding_needed)), 
                                   constant_values=0.0)
            result_indices = np.pad(result_indices, ((0, 0), (0, padding_needed)), 
                                    constant_values=-1)
        
        return result_scores, result_indices
    
    def search_exclude_self(self, query_idx: int, top_k: int):
        """
        Search using an entry from the KB itself, excluding self-match.
        
        Args:
            query_idx: Index of the query entry in KB
            top_k: Number of results to return
        
        Returns:
            scores: Similarity scores (1, K)
            indices: Result indices (1, K) - guaranteed not to include query_idx
        """
        query_emb = self.embeddings[query_idx:query_idx+1]
        return self.search(query_emb, top_k, exclude_indices=[query_idx])
    
    def get_metadata(self, indices):
        """
        Get metadata for given indices.
        
        Args:
            indices: Array of indices or single index
        
        Returns:
            List of metadata dicts or single dict
        """
        if isinstance(indices, (int, np.integer)):
            if indices < 0 or indices >= len(self.metadata):
                return None
            return self.metadata[indices]
        
        results = []
        for idx in indices:
            if idx < 0 or idx >= len(self.metadata):
                results.append(None)
            else:
                results.append(self.metadata[int(idx)])
        return results