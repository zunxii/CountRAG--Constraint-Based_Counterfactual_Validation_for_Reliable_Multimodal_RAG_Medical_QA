"""
StabilityRunner (KB-agnostic)
-----------------------------

This runner accepts either:
- the legacy StabilityRetriever (with .retrieve())
- the unified KBRetriever (with .search() and .get_metadata())
- any other retriever exposing either `retrieve(emb)` or `search(emb, top_k)` + `get_metadata(indices)`

The runner produces:
{
    "baseline": {...distribution...},
    "no_text": {...},
    "no_image": {...},
    "noisy": {...},
    "stability": {...metrics...},
    "constraints": {...diagnostics...},
    "retrieved": [ { "diagnosis_label": ..., "score": ... , "metadata": {...} }, ... ]
}

This file replaces the previous, more fragile implementation and performs defensive checks.
"""
import torch
import numpy as np
from typing import Any, Dict, List

from .perturbations import remove_text, remove_image, add_noise
from .distribution import cluster_distribution
from .stability_metrics import stability_report

# Optional constraints extractor - keep import local to avoid circular issues
try:
    from core.reasoning.constraints.extractor import ConstraintExtractor
except Exception:
    try:
        from core.reasoning.constraints.extractor import ConstraintExtractor  # relative fallback
    except Exception:
        ConstraintExtractor = None


class StabilityRunner:
    def __init__(self, retriever: Any, fusion: Any, device: str = "cpu", top_k: int = 10):
        self.retriever = retriever
        self.fusion = fusion.eval() if hasattr(fusion, "eval") else fusion
        self.device = device
        self.top_k = top_k

        # instantiate extractor if available
        self.constraint_extractor = ConstraintExtractor() if ConstraintExtractor is not None else None

    def _call_kb_search(self, fused_embedding: torch.Tensor) -> List[Dict]:
        """
        Call retriever.search(...) + get_metadata(...) and normalize results to:
        [ {"score": float, "metadata": {...}, "diagnosis_label": str}, ... ]
        """
        # Convert to numpy float32 (KBRetriever expects this)
        q = fused_embedding.detach().cpu().numpy().astype("float32")
        # If 1D, ensure shape (1, D)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        # call search -> returns (scores, indices)
        scores, indices = self.retriever.search(q, top_k=self.top_k)

        # scores, indices are (1, K)
        scores = np.array(scores)
        indices = np.array(indices)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            idx = int(idx)
            # Retrieve metadata if retriever supports it
            meta = {}
            if hasattr(self.retriever, "get_metadata"):
                md = self.retriever.get_metadata([idx])
                meta = md[0] if isinstance(md, list) and md else md
            # Best-effort extraction of diagnosis_label
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
        """
        Call retriever.retrieve(...) which is expected to return a list of dicts
        containing at minimum 'diagnosis_label' and optional 'score' or 'metadata'.
        """
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
        """
        Dispatch to the correct retriever interface.
        """
        # If retriever has a direct 'retrieve' method (legacy StabilityRetriever)
        if hasattr(self.retriever, "retrieve"):
            return self._call_legacy_retrieve(fused_embedding)

        # If retriever exposes 'search' and 'get_metadata' (KBRetriever)
        if hasattr(self.retriever, "search") and hasattr(self.retriever, "get_metadata"):
            return self._call_kb_search(fused_embedding)

        # If retriever has 'search_images' for concept-image expansion
        if hasattr(self.retriever, "search_images"):
            # Attempt to call and expect list-of-dicts (best-effort)
            try:
                out = self.retriever.search_images(fused_embedding, top_k=self.top_k)
                # Normalize similar to legacy format
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

        raise RuntimeError("Unsupported retriever interface. Expected one of: retrieve(), search()+get_metadata(), or search_images().")

    def _infer(self, img_emb: torch.Tensor, txt_emb: torch.Tensor) -> Dict:
        # Ensure on device
        img_emb = img_emb.to(self.device)
        txt_emb = txt_emb.to(self.device)
        with torch.no_grad():
            out = self.fusion(img_emb, txt_emb)
        # Return fused tensor (detach, CPU)
        return out.detach().cpu()

    def run(self, img_emb: torch.Tensor, txt_emb: torch.Tensor) -> Dict[str, Any]:
        """
        Run baseline and counterfactuals, compute retrievals and constraints.
        """
        # Baseline
        baseline_fused = self._infer(img_emb, txt_emb)

        # Variants
        no_text_fused = self._infer(img_emb, torch.zeros_like(txt_emb))
        no_image_fused = self._infer(torch.zeros_like(img_emb), txt_emb)
        noisy_fused = self._infer(img_emb + (torch.randn_like(img_emb) * 0.01), txt_emb)

        # Retrieve results for baseline fused embedding
        retrieved = []
        try:
            retrieved = self._call_retriever(baseline_fused)
        except Exception as e:
            # Best-effort: continue with empty retrievals but include error message
            retrieved = []
            retrieval_error = str(e)
        else:
            retrieval_error = None

        # Build distributions
        baseline_dist = cluster_distribution(retrieved)
        no_text_dist = cluster_distribution(self._call_retriever(no_text_fused) if retrieved is not None else [])
        no_image_dist = cluster_distribution(self._call_retriever(no_image_fused) if retrieved is not None else [])
        noisy_dist = cluster_distribution(self._call_retriever(noisy_fused) if retrieved is not None else [])

        # Compute stability metrics
        stability = stability_report(baseline_dist, {
            "no_text": no_text_dist,
            "no_image": no_image_dist,
            "noisy": noisy_dist
        })

        # Constraints extraction (best-effort)
        constraints = {}
        if self.constraint_extractor is not None:
            try:
                # compute simple centroid distances placeholder:
                centroid_distances = {}
                query_distance = 0.0
                if retrieved:
                    # use top score (interpreted as similarity) to compute a query_distance heuristic
                    top_score = retrieved[0].get("score", 0.0)
                    query_distance = float(1.0 - top_score)
                constraints = self.constraint_extractor.extract(
                    retrieved_metadata=[r["metadata"] for r in retrieved],
                    img_emb=img_emb,
                    txt_emb=txt_emb,
                    centroid_distances=centroid_distances,
                    query_distance=query_distance,
                    percentile_95=95.0
                )
            except Exception:
                constraints = {}

        result = {
            "baseline": baseline_dist,
            "no_text": no_text_dist,
            "no_image": no_image_dist,
            "noisy": noisy_dist,
            "stability": stability,
            "constraints": constraints,
            "retrieved": retrieved,
        }
        if retrieval_error:
            result["_retrieval_error"] = retrieval_error
        return result
