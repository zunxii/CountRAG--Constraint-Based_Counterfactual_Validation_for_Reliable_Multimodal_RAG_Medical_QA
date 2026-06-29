"""
KB Retriever - UPDATED to support both flat and concept modes
Replaces: core/retrieval/retriever.py
"""
import faiss
import json
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict


class KBRetriever:
    """
    Unified retriever supporting:
    - Flat mode: Original one-entry-per-image
    - Concept mode: Concept-based retrieval
    """
    
    def __init__(self, kb_dir: str, mode: str = "auto"):
        """
        Args:
            kb_dir: Path to KB directory
            mode: 'auto', 'text', 'image', or 'fusion'
                  'auto' detects flat vs concept automatically
        """
        kb_dir = Path(kb_dir)
        self.kb_dir = kb_dir
        
        # Load config to detect KB type
        with open(kb_dir / "kb_config.json") as f:
            config = json.load(f)
        
        self.kb_mode = config.get('mode', 'flat')  # 'flat' or 'concept'
        self.retrieval_mode = mode if mode != "auto" else "fusion"
        
        print(f"Loading KB: mode={self.kb_mode}, retrieval={self.retrieval_mode}")
        
        if self.kb_mode == "flat":
            self._load_flat(kb_dir)
        else:
            self._load_concept(kb_dir, self.retrieval_mode)
    
    def _load_flat(self, kb_dir: Path):
        """Load flat KB (original)"""
        self.embeddings = np.load(kb_dir / "embeddings.npy")
        self.index = faiss.read_index(str(kb_dir / "index.faiss"))
        
        with open(kb_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        
        assert self.index.ntotal == len(self.metadata), \
            f"Index size mismatch: {self.index.ntotal} != {len(self.metadata)}"
        
        print(f"✓ Loaded flat KB: {len(self.metadata)} entries")
    
    def _load_concept(self, kb_dir: Path, retrieval_mode: str):
        """Load concept KB"""
        # Load appropriate index based on mode
        if retrieval_mode == "text":
            self.index = faiss.read_index(str(kb_dir / "text_index.faiss"))
            emb_path = kb_dir / "text_embeddings.npy"
        elif retrieval_mode == "image":
            self.index = faiss.read_index(str(kb_dir / "image_index.faiss"))
            emb_path = kb_dir / "image_embeddings.npy"
        else:  # fusion
            self.index = faiss.read_index(str(kb_dir / "concept_index.faiss"))
            emb_path = kb_dir / "concept_embeddings.npy"

        # Load metadata
        with open(kb_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        # Load embeddings for downstream calibration / diagnostics
        if emb_path.exists():
            self.embeddings = np.load(emb_path)
        else:
            self.embeddings = None

        # Load mappings
        with open(kb_dir / "image_to_concept.json") as f:
            self.image_to_concept = json.load(f)

        print(f"✓ Loaded concept KB: {len(self.metadata)} concepts")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int, 
        exclude_indices: list = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search KB (works for both flat and concept modes)
        
        Returns:
            scores: (1, K) similarity scores
            indices: (1, K) result indices
        """
        if self.kb_mode == "flat":
            return self._search_flat(query_embedding, top_k, exclude_indices)
        else:
            return self._search_concept(query_embedding, top_k, exclude_indices)
    
    def _search_flat(
        self, 
        query_embedding: np.ndarray, 
        top_k: int,
        exclude_indices: list = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search flat KB (original logic)"""
        query_embedding = query_embedding.astype("float32")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        faiss.normalize_L2(query_embedding)
        
        exclude_set = set(exclude_indices) if exclude_indices else set()
        
        # Fetch more to account for exclusions
        fetch_k = min(top_k * 2 + len(exclude_set), self.index.ntotal)
        scores, indices = self.index.search(query_embedding, fetch_k)
        
        # Filter exclusions
        filtered_scores = []
        filtered_indices = []
        seen_cases = set()
        
        for score, idx in zip(scores[0], indices[0]):
            idx = int(idx)
            
            if idx in exclude_set:
                continue
            
            case_id = self.metadata[idx].get("case_id", f"case_{idx}")
            if case_id in seen_cases:
                continue
            
            seen_cases.add(case_id)
            filtered_scores.append(score)
            filtered_indices.append(idx)
            
            if len(filtered_indices) >= top_k:
                break
        
        result_scores = np.array(filtered_scores[:top_k]).reshape(1, -1)
        result_indices = np.array(filtered_indices[:top_k]).reshape(1, -1)
        
        # Pad if needed
        if result_indices.shape[1] < top_k:
            padding_needed = top_k - result_indices.shape[1]
            result_scores = np.pad(result_scores, ((0, 0), (0, padding_needed)), 
                                   constant_values=0.0)
            result_indices = np.pad(result_indices, ((0, 0), (0, padding_needed)), 
                                    constant_values=-1)
        
        return result_scores, result_indices
    
    def _search_concept(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        exclude_indices: list = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search concept KB
        Returns concept indices (NOT image indices)
        """
        query = query_embedding.astype('float32')
        if query.ndim == 1:
            query = query.reshape(1, -1)
        faiss.normalize_L2(query)
        
        # Search concepts
        fetch_k = min(top_k * 2, len(self.metadata))
        scores, indices = self.index.search(query, fetch_k)
        
        # Filter (concept-level exclusion would go here if needed)
        result_scores = scores[0][:top_k]
        result_indices = indices[0][:top_k]
        
        return result_scores.reshape(1, -1), result_indices.reshape(1, -1)
    
    def search_images(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        concepts_to_retrieve: int = 5,
        exclude_image_paths: List[str] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Search and expand to images (concept mode only)
        
        Returns:
            scores: Image-level scores
            images: List of image result dicts
        """
        if self.kb_mode == "flat":
            raise ValueError("search_images() only works in concept mode")
        
        # Get top concepts
        concept_scores, concept_indices = self._search_concept(
            query_embedding,
            top_k=concepts_to_retrieve
        )
        
        # Expand to images
        exclude_set = set(exclude_image_paths) if exclude_image_paths else set()
        
        results = []
        for score, idx in zip(concept_scores[0], concept_indices[0]):
            if idx < 0:
                break
            
            concept = self.metadata[int(idx)]
            
            for img_path in concept['image_paths']:
                if img_path in exclude_set:
                    continue
                
                results.append({
                    'image_path': img_path,
                    'diagnosis_label': concept['diagnosis_label'],
                    'canonical_text': concept['canonical_text'],
                    'concept_id': concept['concept_id'],
                    'concept_score': float(score),
                })
                
                if len(results) >= top_k:
                    break
            
            if len(results) >= top_k:
                break
        
        scores = np.array([r['concept_score'] for r in results])
        return scores, results
    
    def get_metadata(self, indices):
        """Get metadata for indices (works for both modes)"""
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
    
    def get_statistics(self) -> Dict:
        """Get KB statistics"""
        if self.kb_mode == "flat":
            return {
                'mode': 'flat',
                'num_entries': len(self.metadata),
            }
        else:
            total_images = sum(c['num_images'] for c in self.metadata)
            return {
                'mode': 'concept',
                'num_concepts': len(self.metadata),
                'num_images': total_images,
                'avg_images_per_concept': total_images / len(self.metadata),
            }