from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .stability_metrics import stability_report
from collections import Counter

try:
    from core.reasoning.constraints.extractor import ConstraintExtractor
except Exception:
    ConstraintExtractor = None


class StabilityRunner:
    def __init__(
        self,
        retriever: Any,
        fusion: Any,
        contract: dict | None = None,
        device: str = "cpu",
        top_k: int = 10,
        kb_centroid=None,
        support_reference=None,
        ood_threshold=None,
    ):
        self.retriever = retriever
        self.fusion = fusion.eval() if hasattr(fusion, "eval") else fusion
        self.device = device
        self.top_k = top_k
        self.contract = contract or {}
        self.constraint_extractor = ConstraintExtractor(self.contract) if ConstraintExtractor is not None else None

        self.perturbation_scales = list(
            self.contract.get("counterfactual", {}).get("perturbation_scales", [0.01, 0.05, 0.1])
        )

        target_scale = 0.05
        self._default_noisy_scale = min(
            self.perturbation_scales, key=lambda s: abs(float(s) - target_scale)
        ) if self.perturbation_scales else target_scale

        self.train_centroid = None
        centroid_path = self.contract.get("paths", {}).get("train_centroid_file")
        if centroid_path and Path(centroid_path).exists():
            self.train_centroid = np.load(centroid_path).astype(np.float32)

        if self.train_centroid is None and kb_centroid is not None:
            self.train_centroid = np.asarray(kb_centroid, dtype=np.float32)

        self.support_reference = float(support_reference) if support_reference is not None else None
        self.ood_threshold = float(ood_threshold) if ood_threshold is not None else None

        self._text_neutral_emb: Optional[torch.Tensor] = None
        self._image_neutral_emb: Optional[torch.Tensor] = None
        self._build_neutral_embeddings()

    def _build_neutral_embeddings(self):
        if not hasattr(self.retriever, "metadata"):
            return

        metadata = self.retriever.metadata
        img_vecs: List[np.ndarray] = []
        txt_vecs: List[np.ndarray] = []
        all_vecs: List[np.ndarray] = []

        for entry in metadata:
            if not isinstance(entry, dict):
                continue
            for key in ("aggregated_image_embedding", "image_embedding"):
                v = entry.get(key)
                if v is not None:
                    try:
                        arr = np.asarray(v, dtype=np.float32).reshape(-1)
                        if arr.size > 0:
                            img_vecs.append(arr)
                    except Exception:
                        pass
            for key in ("canonical_text_embedding", "text_embedding"):
                v = entry.get(key)
                if v is not None:
                    try:
                        arr = np.asarray(v, dtype=np.float32).reshape(-1)
                        if arr.size > 0:
                            txt_vecs.append(arr)
                    except Exception:
                        pass
            for key in ("concept_embedding", "embedding"):
                v = entry.get(key)
                if v is not None:
                    try:
                        arr = np.asarray(v, dtype=np.float32).reshape(-1)
                        if arr.size > 0:
                            all_vecs.append(arr)
                    except Exception:
                        pass

        def _mean_normalise(vecs: List[np.ndarray]) -> Optional[torch.Tensor]:
            if not vecs:
                return None
            mat = np.stack(vecs, axis=0)
            mean = mat.mean(axis=0).astype(np.float32)
            norm = float(np.linalg.norm(mean))
            if norm > 0:
                mean = mean / norm
            return torch.from_numpy(mean).unsqueeze(0)

        fallback = _mean_normalise(all_vecs)
        self._text_neutral_emb = _mean_normalise(img_vecs) or fallback
        self._image_neutral_emb = _mean_normalise(txt_vecs) or fallback

    def _neutral_text_emb(self, ref_emb: torch.Tensor) -> torch.Tensor:
        if self._text_neutral_emb is not None:
            n = self._text_neutral_emb.to(ref_emb.device)
            if n.shape[-1] == ref_emb.shape[-1]:
                return n.expand_as(ref_emb)
        return torch.zeros_like(ref_emb)

    def _neutral_image_emb(self, ref_emb: torch.Tensor) -> torch.Tensor:
        if self._image_neutral_emb is not None:
            n = self._image_neutral_emb.to(ref_emb.device)
            if n.shape[-1] == ref_emb.shape[-1]:
                return n.expand_as(ref_emb)
        return torch.zeros_like(ref_emb)

    def _call_kb_search(self, fused_embedding: torch.Tensor) -> List[Dict]:
        q = fused_embedding.detach().cpu().numpy().astype("float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)

        scores, indices = self.retriever.search(q, top_k=self.top_k)
        scores = np.asarray(scores)
        indices = np.asarray(indices)

        results: List[Dict] = []
        for score, idx in zip(scores[0], indices[0]):
            idx = int(idx)
            meta: Dict[str, Any] = {}
            if hasattr(self.retriever, "get_metadata"):
                md = self.retriever.get_metadata([idx])
                meta = md[0] if isinstance(md, list) and md else (md if isinstance(md, dict) else {})

            diag = None
            if isinstance(meta, dict):
                diag = meta.get("diagnosis_label") or meta.get("diagnosis") or meta.get("label")

            results.append({
                "score": float(score) if score is not None else 0.0,
                "metadata": meta,
                "diagnosis_label": diag or "unknown",
            })
        return results

    def _call_legacy_retrieve(self, fused_embedding: torch.Tensor) -> List[Dict]:
        out = self.retriever.retrieve(fused_embedding)
        normalized: List[Dict] = []
        for r in out:
            meta = r.get("metadata", {}) if isinstance(r, dict) else {}
            diag = None
            if isinstance(r, dict):
                diag = r.get("diagnosis_label") or meta.get("diagnosis_label") or meta.get("label")
            normalized.append({
                "score": float(r.get("score", 0.0)) if isinstance(r, dict) else 0.0,
                "metadata": meta,
                "diagnosis_label": diag or "unknown",
            })
        return normalized

    def _call_retriever(self, fused_embedding: torch.Tensor) -> List[Dict]:
        if hasattr(self.retriever, "retrieve"):
            return self._call_legacy_retrieve(fused_embedding)
        if hasattr(self.retriever, "search") and hasattr(self.retriever, "get_metadata"):
            return self._call_kb_search(fused_embedding)
        raise RuntimeError("Unsupported retriever interface")

    def _infer(self, img_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        img_emb = img_emb.to(self.device)
        txt_emb = txt_emb.to(self.device)
        with torch.no_grad():
            out = self.fusion(img_emb, txt_emb)
        return out.detach().cpu()

    def _maybe_distribution(self, fused_embedding: torch.Tensor) -> Dict[str, Any]:
        if hasattr(self.retriever, "posterior_distribution"):
            try:
                result = self.retriever.posterior_distribution(fused_embedding)
                return result
            except Exception:
                pass

        retrieved = self._call_retriever(fused_embedding)

        if not retrieved:
            return {"distribution": {}, "num_clusters": 0, "top_label": "unknown", "top_prob": 0.0}

        counts = Counter(r.get("diagnosis_label", "unknown") for r in retrieved)
        total = sum(counts.values()) or 1
        distribution = {k: round(v / total, 4) for k, v in counts.items()}
        top_label, top_count = counts.most_common(1)[0]

        return {
            "distribution": distribution,
            "num_clusters": len(counts),
            "top_label": top_label,
            "top_prob": round(top_count / total, 4),
        }

    def _query_distance(self, fused: torch.Tensor) -> float:
        q = fused.detach().cpu().numpy().astype("float32").reshape(-1)
        if self.train_centroid is not None:
            c = np.asarray(self.train_centroid, dtype=np.float32).reshape(-1)
            if c.shape == q.shape:
                return float(np.linalg.norm(q - c))
        return 1.0

    def run(self, img_emb: torch.Tensor, txt_emb: torch.Tensor) -> Dict[str, Any]:
        baseline_fused  = self._infer(img_emb, txt_emb)
        no_text_fused   = self._infer(img_emb, self._neutral_text_emb(txt_emb))
        no_image_fused  = self._infer(self._neutral_image_emb(img_emb), txt_emb)

        baseline = self._maybe_distribution(baseline_fused)
        no_text  = self._maybe_distribution(no_text_fused)
        no_image = self._maybe_distribution(no_image_fused)

        noisy_by_scale: Dict[str, Dict[str, Any]] = {}
        for scale in self.perturbation_scales:
            noisy_fused = baseline_fused + torch.randn_like(baseline_fused) * float(scale)
            noisy_by_scale[str(scale)] = self._maybe_distribution(noisy_fused)

        noisy_default = noisy_by_scale.get(
            str(self._default_noisy_scale), baseline
        ) if self.perturbation_scales else baseline

        # Stability report — computed BEFORE constraints so we can pass
        # robustness_level into extractor.extract()
        stability = stability_report(
            baseline["distribution"],
            {"no_text": no_text, "no_image": no_image, "noisy": noisy_default},
        )
        stability["js_divergence_by_scale"] = {
            str(scale): stability_report(
                baseline["distribution"],
                {"noisy": dist},
            )["js_divergence"]["noisy"]
            for scale, dist in noisy_by_scale.items()
        }

        # Pull the two signals we need to pass to extractor
        robustness_level = stability.get("robustness_level", "high")
        max_js_divergence = float(max(stability["js_divergence"].values())) \
            if stability.get("js_divergence") else 0.0

        retrieved = self._call_retriever(baseline_fused)
        top_scores = [float(r.get("score", 0.0)) for r in retrieved]

        baseline_dist = baseline.get("distribution", {})
        sorted_probs = sorted(baseline_dist.values(), reverse=True)
        top1_prob = sorted_probs[0] if len(sorted_probs) > 0 else 0.0
        top2_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0

        query_distance = self._query_distance(baseline_fused)

        if self.ood_threshold is not None:
            percentile_95 = self.ood_threshold
        else:
            percentile_95 = float(np.percentile([1.0 - s for s in top_scores], 95)) if top_scores else 1.0

        constraints = {}
        if self.constraint_extractor is not None:
            try:
                constraints = self.constraint_extractor.extract(
                    retrieved_metadata=[r.get("metadata", {}) for r in retrieved],
                    retrieved=retrieved,
                    baseline_distribution=baseline["distribution"],
                    img_emb=img_emb,
                    txt_emb=txt_emb,
                    top1_prob=top1_prob,
                    top2_prob=top2_prob,
                    query_distance=query_distance,
                    percentile_95=percentile_95,
                    support_reference=self.support_reference,
                    # KEY FIX: pass stability signals so overall_violation
                    # can incorporate JSD / robustness level
                    robustness_level=robustness_level,
                    max_js_divergence=max_js_divergence,
                )
            except Exception as exc:
                constraints = {"error": str(exc)}

        return {
            "baseline": baseline["distribution"],
            "no_text": no_text["distribution"],
            "no_image": no_image["distribution"],
            "noisy": noisy_default["distribution"],
            "baseline_distribution": baseline,
            "no_text_distribution": no_text,
            "no_image_distribution": no_image,
            "noisy_distribution": noisy_default,
            "noisy_by_scale": {k: v for k, v in noisy_by_scale.items()},
            "perturbation_scales": self.perturbation_scales,
            "stability": stability,
            "constraints": constraints,
            "retrieved": retrieved,
        }