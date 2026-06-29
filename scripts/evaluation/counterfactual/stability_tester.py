from __future__ import annotations

import sys
from pathlib import Path
from statistics import median
from typing import Any, Dict, Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import torch

from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.fusion.adaptive_fusion import AdaptiveFusion
from core.kb.image_loader import ImageLoader
from core.reasoning.counterfactuals.stability.retrieval import StabilityRetriever
from core.reasoning.counterfactuals.stability.runner import StabilityRunner
from core.retrieval.retriever import KBRetriever


class StabilityTester:
    def __init__(self, kb_dir: str, contract: dict, device: str = "cpu"):
        self.device = device
        self.contract = contract
        kb_dir = Path(kb_dir)

        self.kb_retriever = KBRetriever(str(kb_dir))
        self.kb_mode = self.kb_retriever.kb_mode
        self.kb_metadata = self.kb_retriever.metadata

        self.stability_retriever = StabilityRetriever(
            self.kb_retriever.index,
            self.kb_retriever.metadata,
        )

        self._kb_centroid = self._compute_kb_centroid()
        self._support_reference = self._compute_support_reference()
        self._ood_threshold_95 = self._compute_ood_threshold_95()

        lora_path = contract["paths"]["models"]["lora_dir"]
        fusion_path = Path(contract["paths"]["models"]["fusion_model_file"])

        self.encoder = BioMedCLIPEncoder(device=device, lora_path=lora_path)
        self.image_loader = ImageLoader()

        self.fusion = AdaptiveFusion().to(device)
        self.fusion.load_state_dict(torch.load(fusion_path, map_location=device))
        self.fusion.eval()

        top_k = int(contract.get("retrieval", {}).get("top_k_for_inference", 10))
        self.runner = StabilityRunner(
            self.stability_retriever,
            self.fusion,
            contract=contract,
            device=device,
            top_k=top_k,
            kb_centroid=self._kb_centroid,
            support_reference=self._support_reference,
            ood_threshold=self._ood_threshold_95,
        )

    def _extract_vector(self, entry: Dict[str, Any]) -> Optional[np.ndarray]:
        for key in ("concept_embedding", "embedding", "aggregated_image_embedding"):
            vec = entry.get(key)
            if vec is not None:
                try:
                    arr = np.asarray(vec, dtype=np.float32)
                    if arr.ndim == 1 and arr.size > 0:
                        return arr
                except Exception:
                    continue
        return None

    def _collect_kb_vectors(self) -> List[np.ndarray]:
        vectors: List[np.ndarray] = []

        index = getattr(self.kb_retriever, "index", None)
        if index is not None:
            try:
                ntotal = int(getattr(index, "ntotal", 0))
                if ntotal > 0 and hasattr(index, "reconstruct_n"):
                    arr = index.reconstruct_n(0, ntotal)
                    arr = np.asarray(arr, dtype=np.float32)
                    if arr.ndim == 2 and arr.shape[0] > 0:
                        return [v.astype(np.float32) for v in arr]
            except Exception:
                pass

            try:
                ntotal = int(getattr(index, "ntotal", 0))
                if ntotal > 0 and hasattr(index, "reconstruct"):
                    for i in range(ntotal):
                        vec = np.asarray(index.reconstruct(i), dtype=np.float32)
                        if vec.ndim == 1 and vec.size > 0:
                            vectors.append(vec)
                    if vectors:
                        return vectors
            except Exception:
                pass

        if hasattr(self.kb_retriever, "embeddings"):
            try:
                arr = np.asarray(self.kb_retriever.embeddings, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[0] > 0:
                    return [v for v in arr]
            except Exception:
                pass

        for entry in self.kb_metadata:
            if not isinstance(entry, dict):
                continue
            vec = self._extract_vector(entry)
            if vec is not None:
                vectors.append(vec)

        return vectors

    def _compute_kb_centroid(self) -> Optional[np.ndarray]:
        vectors = self._collect_kb_vectors()
        if not vectors:
            return None

        centroid = np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid = centroid / norm
        return centroid

    def _compute_support_reference(self) -> float:
        supports = []
        for entry in self.kb_metadata:
            if not isinstance(entry, dict):
                continue
            if "num_images" in entry:
                try:
                    supports.append(float(entry["num_images"]))
                    continue
                except Exception:
                    pass
            if isinstance(entry.get("image_paths"), list):
                supports.append(float(len([p for p in entry["image_paths"] if p])))
            else:
                supports.append(1.0)

        if not supports:
            return 1.0
        return float(median(supports))

    def _compute_ood_threshold_95(self) -> float:
        vectors = self._collect_kb_vectors()
        if not vectors or self._kb_centroid is None:
            return 1.0

        centroid = self._kb_centroid.astype(np.float32)
        dists = []
        for vec in vectors:
            v = np.asarray(vec, dtype=np.float32)
            if v.shape != centroid.shape:
                continue
            dists.append(float(np.linalg.norm(v - centroid)))

        if not dists:
            return 1.0
        return float(np.percentile(dists, 95))

    def test_query(self, query: dict) -> dict:
        img = self.image_loader.load(query["image_path"])

        with torch.no_grad():
            img_emb = self.encoder.encode_image(img).unsqueeze(0)
            txt_emb = self.encoder.encode_text(query["combined_text"]).unsqueeze(0)

        stability_output = self.runner.run(img_emb, txt_emb)

        result = {
            "query_id": query.get("query_id", ""),
            "diagnosis": query.get("diagnosis_label", "unknown"),
            "image_path": query.get("image_path", ""),
            "query_text": query.get("combined_text", ""),
            "baseline": stability_output["baseline"],
            "no_text": stability_output["no_text"],
            "no_image": stability_output["no_image"],
            "noisy": stability_output["noisy"],
            "stability": stability_output["stability"],
            "baseline_distribution": stability_output["baseline_distribution"],
            "no_text_distribution": stability_output["no_text_distribution"],
            "no_image_distribution": stability_output["no_image_distribution"],
            "noisy_distribution": stability_output["noisy_distribution"],
            "noisy_distributions": {
                str(scale): dist for scale, dist in stability_output.get("noisy_by_scale", {}).items()
            },
            "retrieved": stability_output.get("retrieved", []),
        }

        if "constraints" in stability_output:
            result["constraints"] = stability_output["constraints"]

        if "_retrieval_error" in stability_output:
            result["_retrieval_error"] = stability_output["_retrieval_error"]

        return result

    def get_metadata(self):
        return self.kb_metadata

    def test_sample(self, idx: int):
        entry = self.kb_metadata[idx]
        img = self.image_loader.load(entry["image_path"])

        with torch.no_grad():
            img_emb = self.encoder.encode_image(img).unsqueeze(0)
            txt_emb = self.encoder.encode_text(entry["clinical_text"]["combined"]).unsqueeze(0)

        stability_output = self.runner.run(img_emb, txt_emb)

        result = {
            "case_id": entry.get("case_id", idx),
            "diagnosis": entry.get("diagnosis_label", "unknown"),
            "baseline": stability_output["baseline"],
            "no_text": stability_output["no_text"],
            "no_image": stability_output["no_image"],
            "noisy": stability_output["noisy"],
            "stability": stability_output["stability"],
            "baseline_distribution": stability_output["baseline_distribution"],
            "no_text_distribution": stability_output["no_text_distribution"],
            "no_image_distribution": stability_output["no_image_distribution"],
            "noisy_distribution": stability_output["noisy_distribution"],
            "noisy_distributions": {
                str(scale): dist for scale, dist in stability_output.get("noisy_by_scale", {}).items()
            },
            "retrieved": stability_output.get("retrieved", []),
        }

        if "constraints" in stability_output:
            result["constraints"] = stability_output["constraints"]

        if "_retrieval_error" in stability_output:
            result["_retrieval_error"] = stability_output["_retrieval_error"]

        return result