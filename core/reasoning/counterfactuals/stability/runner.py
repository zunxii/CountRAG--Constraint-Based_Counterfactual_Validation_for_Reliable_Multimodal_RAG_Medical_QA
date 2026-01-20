"""
Changes:
1. Return retrieved metadata alongside distribution
2. Compute centroid distances for boundary analysis
3. Pass query distance for distribution check
"""
import torch
import numpy as np
from typing import Any, Dict, List

from .perturbations import remove_text, remove_image, add_noise
from .distribution import cluster_distribution
from .stability_metrics import stability_report

# Optional constraints extractor
try:
    from core.reasoning.constraints.extractor import ConstraintExtractor
except Exception:
    ConstraintExtractor = None


class StabilityRunner:
    def __init__(self, retriever: Any, fusion: Any, device: str = "cpu", top_k: int = 10):
        self.retriever = retriever
        self.fusion = fusion.eval() if hasattr(fusion, "eval") else fusion
        self.device = device
        self.top_k = top_k
        
        # Instantiate extractor if available
        self.constraint_extractor = ConstraintExtractor() if ConstraintExtractor is not None else None

    def _call_kb_search(self, fused_embedding: torch.Tensor) -> List[Dict]:
        """Call retriever.search(...) + get_metadata(...) and normalize results"""
        q = fused_embedding.detach().cpu().numpy().astype("float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)

        scores, indices = self.retriever.search(q, top_k=self.top_k)
        scores = np.array(scores)
        indices = np.array(indices)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            idx = int(idx)
            meta = {}
            if hasattr(self.retriever, "get_metadata"):
                md = self.retriever.get_metadata([idx])
                meta = md[0] if isinstance(md, list) and md else md
            
            diag = None
            if isinstance(meta, dict):
                diag = meta.get("diagnosis_label") or meta.get("diagnosis") or meta.get("label")
            
            results.append({
                "score": float(score) if score is not None else 0.0,
                "metadata": meta,
                "diagnosis_label": diag or "unknown"
            })
        return results

    def _call_legacy_retrieve(self, fused_embedding: torch.Tensor) -> List[Dict]:
        """Call retriever.retrieve(...) for legacy compatibility"""
        out = self.retriever.retrieve(fused_embedding)
        normalized = []
        for r in out:
            meta = r.get("metadata", {}) if isinstance(r, dict) else {}
            diag = None
            if isinstance(r, dict):
                diag = r.get("diagnosis_label") or meta.get("diagnosis_label") or meta.get("label")
            normalized.append({
                "score": float(r.get("score", 0.0)) if isinstance(r, dict) else 0.0,
                "metadata": meta,
                "diagnosis_label": diag or "unknown"
            })
        return normalized

    def _call_retriever(self, fused_embedding: torch.Tensor) -> List[Dict]:
        """Dispatch to the correct retriever interface"""
        if hasattr(self.retriever, "retrieve"):
            return self._call_legacy_retrieve(fused_embedding)
        if hasattr(self.retriever, "search") and hasattr(self.retriever, "get_metadata"):
            return self._call_kb_search(fused_embedding)
        if hasattr(self.retriever, "search_images"):
            try:
                out = self.retriever.search_images(fused_embedding, top_k=self.top_k)
                normalized = []
                for r in out:
                    meta = r.get("metadata", {}) if isinstance(r, dict) else {}
                    diag = meta.get("diagnosis_label") if isinstance(meta, dict) else None
                    normalized.append({
                        "score": float(r.get("score", 0.0)) if isinstance(r, dict) else 0.0,
                        "metadata": meta,
                        "diagnosis_label": diag or "unknown"
                    })
                return normalized
            except Exception:
                pass
        raise RuntimeError("Unsupported retriever interface")

    def _infer(self, img_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        """Run fusion inference"""
        img_emb = img_emb.to(self.device)
        txt_emb = txt_emb.to(self.device)
        with torch.no_grad():
            out = self.fusion(img_emb, txt_emb)
        return out.detach().cpu()
    
    def _compute_centroid_distances(self, query_emb: torch.Tensor, retrieved: List[Dict]) -> Dict[str, float]:
        """Compute distances to diagnosis centroids for boundary analysis"""
        from collections import defaultdict
        
        # Group by diagnosis
        by_diagnosis = defaultdict(list)
        for r in retrieved:
            diag = r["diagnosis_label"]
            score = r["score"]
            by_diagnosis[diag].append(score)
        
        # Compute centroid distance (1 - avg_score as proxy)
        centroid_distances = {}
        for diag, scores in by_diagnosis.items():
            avg_score = np.mean(scores)
            centroid_distances[diag] = float(1.0 - avg_score)
        
        return centroid_distances

    def run(self, img_emb: torch.Tensor, txt_emb: torch.Tensor) -> Dict[str, Any]:
        """
        Run baseline and counterfactuals, compute retrievals and constraints.
        
        NEW: Returns retrieved metadata and computes constraints
        """
        # Baseline
        baseline_fused = self._infer(img_emb, txt_emb)

        # Variants
        no_text_fused = self._infer(img_emb, torch.zeros_like(txt_emb))
        no_image_fused = self._infer(torch.zeros_like(img_emb), txt_emb)
        noisy_fused = self._infer(img_emb + (torch.randn_like(img_emb) * 0.01), txt_emb)

        # Retrieve results for baseline
        retrieved = []
        retrieval_error = None
        try:
            retrieved = self._call_retriever(baseline_fused)
        except Exception as e:
            retrieval_error = str(e)

        # Build distributions
        baseline_dist = cluster_distribution(retrieved)
        no_text_dist = cluster_distribution(self._call_retriever(no_text_fused) if retrieved else [])
        no_image_dist = cluster_distribution(self._call_retriever(no_image_fused) if retrieved else [])
        noisy_dist = cluster_distribution(self._call_retriever(noisy_fused) if retrieved else [])

        # Compute stability metrics
        stability = stability_report(baseline_dist, {
            "no_text": no_text_dist,
            "no_image": no_image_dist,
            "noisy": noisy_dist
        })

        # NEW: Compute constraints if extractor available
        constraints = {}
        if self.constraint_extractor is not None and retrieved:
            try:
                # Compute centroid distances
                centroid_distances = self._compute_centroid_distances(baseline_fused, retrieved)
                
                # Compute query distance (use top score as proxy)
                query_distance = float(1.0 - retrieved[0].get("score", 0.0)) if retrieved else 1.0
                
                # Compute 95th percentile from all scores
                all_scores = [r.get("score", 0.0) for r in retrieved]
                percentile_95 = float(np.percentile([1.0 - s for s in all_scores], 95)) if all_scores else 1.0
                
                # Extract constraints
                constraints = self.constraint_extractor.extract(
                    retrieved_metadata=[r["metadata"] for r in retrieved],
                    img_emb=img_emb,
                    txt_emb=txt_emb,
                    centroid_distances=centroid_distances,
                    query_distance=query_distance,
                    percentile_95=percentile_95
                )
            except Exception as e:
                # Don't break if constraints fail
                constraints = {"error": str(e)}

        result = {
            "baseline": baseline_dist,
            "no_text": no_text_dist,
            "no_image": no_image_dist,
            "noisy": noisy_dist,
            "stability": stability,
            "constraints": constraints,  # NEW
            "retrieved": retrieved,  # NEW: Full metadata
        }
        
        if retrieval_error:
            result["_retrieval_error"] = retrieval_error
        
        return result