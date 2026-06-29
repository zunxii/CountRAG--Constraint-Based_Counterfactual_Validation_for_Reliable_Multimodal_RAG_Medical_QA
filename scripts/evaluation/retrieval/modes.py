"""
Mode evaluator - hierarchical retrieval with global image re-ranking
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
import torch.nn.functional as F
from tqdm import tqdm

from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.fusion.adaptive_fusion import AdaptiveFusion
from core.kb.image_loader import ImageLoader
from core.retrieval.retriever import KBRetriever
from scripts.evaluation.retrieval.metrics import MetricsCalculator


class ModeEvaluator:
    def __init__(
        self,
        kb_dir: str,
        eval_dataset,
        device: str = "cpu",
        lora_path: str | None = None,
        fusion_path: str | None = None,
    ):
        self.device = device
        self.kb_dir = kb_dir
        self.eval_dataset = eval_dataset

        self.retriever = KBRetriever(kb_dir)
        self.kb_metadata = self.retriever.metadata
        self.kb_mode = self.retriever.kb_mode

        self.encoder = BioMedCLIPEncoder(device=device, lora_path=lora_path)
        self.image_loader = ImageLoader()

        fp = Path(fusion_path or "outputs/models/trained_fusion/fusion.pt")
        if fp.exists():
            self.fusion = AdaptiveFusion().to(device)
            self.fusion.load_state_dict(torch.load(fp, map_location=device))
            self.fusion.eval()
        else:
            self.fusion = None

        print(f"Evaluating {self.kb_mode} KB")

        if self.kb_mode == "flat":
            self.image_to_kb_indices = defaultdict(list)
            for idx, entry in enumerate(self.kb_metadata):
                img_path = entry.get("image_path", "")
                self.image_to_kb_indices[img_path].append(idx)
        else:
            self.image_to_concept = self.retriever.image_to_concept

        self.metrics_calc = MetricsCalculator()

        # Global rerank settings for concept KB.
        self.concept_pool_size = 10
        self.rerank_lambda = 0.35  # concept score weight
        self._image_embedding_cache: Dict[str, torch.Tensor] = {}

        # Precompute all KB image embeddings once so reranking stays fast.
        if self.kb_mode == "concept":
            self._precompute_kb_image_embeddings()

    def _precompute_kb_image_embeddings(self):
        """
        Precompute and cache embeddings for every unique KB image path.
        This makes the global rerank fast enough for a full 587-query eval.
        """
        unique_paths = []
        seen = set()

        for entry in self.kb_metadata:
            for path in self._extract_image_paths(entry):
                if path and path not in seen:
                    seen.add(path)
                    unique_paths.append(path)

        if not unique_paths:
            print("No concept-image paths found for caching.")
            return

        print(f"Precomputing KB image embeddings for {len(unique_paths)} images...")
        for path in tqdm(unique_paths, desc="Caching KB images"):
            try:
                _ = self._encode_image_path(path)
            except Exception as exc:
                print(f"  ! Skipping image embedding cache for {path}: {exc}")

        print(f"✓ Cached {len(self._image_embedding_cache)} image embeddings")

    @staticmethod
    def _extract_image_paths(entry: Dict) -> List[str]:
        paths = entry.get("image_paths")
        if isinstance(paths, list) and paths:
            return [p for p in paths if isinstance(p, str) and p]

        single = entry.get("image_path")
        if isinstance(single, str) and single:
            return [single]

        return []

    def evaluate_mode(self, mode: str, top_k: int = 20):
        all_metrics = defaultdict(list)
        per_diagnosis_metrics = defaultdict(lambda: defaultdict(list))

        # Use a local retriever for this mode instead of mutating self.retriever
        mode_retriever = KBRetriever(
            self.kb_dir,
            mode=mode if self.kb_mode == "concept" else "auto"
        )

        for query in tqdm(self.eval_dataset, desc=f"Evaluating {mode}"):
            gt_label = query["diagnosis_label"]
            query_image_path = query["image_path"]

            query_emb = self._encode_query(query, mode)
            if query_emb is None:
                continue

            if self.kb_mode == "flat":
                retrieved = self._retrieve_flat(
                    mode_retriever, query_emb, query_image_path, top_k
                )
            else:
                retrieved = self._retrieve_concept(
                    mode_retriever,
                    query_emb,
                    query_image_path,
                    top_k,
                )

            if not retrieved:
                continue

            metrics = self.metrics_calc.compute_all_metrics(retrieved, gt_label)

            for k, v in metrics.items():
                all_metrics[k].append(v)
                per_diagnosis_metrics[gt_label][k].append(v)

        avg_metrics = {
            k: (sum(v) / len(v)) if len(v) > 0 else 0.0
            for k, v in all_metrics.items()
        }

        per_diag_avg = {}
        for diag, metrics_dict in per_diagnosis_metrics.items():
            per_diag_avg[diag] = {
                k: (sum(v) / len(v)) if len(v) > 0 else 0.0
                for k, v in metrics_dict.items()
            }

        return {
            "metrics": avg_metrics,
            "num_queries": len(self.eval_dataset),
            "per_diagnosis_metrics": per_diag_avg,
        }

    def _encode_query(self, query: Dict, mode: str):
        """Encode query based on mode"""
        with torch.no_grad():
            if mode == "text":
                return self.encoder.encode_text(query["combined_text"])

            elif mode == "image":
                img = self.image_loader.load(query["image_path"])
                return self.encoder.encode_image(img)

            elif mode == "fusion":
                if self.fusion is None:
                    return None

                img = self.image_loader.load(query["image_path"])
                img_emb = self.encoder.encode_image(img).unsqueeze(0)
                txt_emb = self.encoder.encode_text(
                    query["combined_text"]
                ).unsqueeze(0)

                return self.fusion(img_emb, txt_emb).squeeze(0)

        return None

    def _encode_image_path(self, image_path: str) -> torch.Tensor:
        """
        Return a cached image embedding for the given path.
        Stored on CPU to keep memory stable.
        """
        if image_path in self._image_embedding_cache:
            return self._image_embedding_cache[image_path]

        img = self.image_loader.load(image_path)
        with torch.no_grad():
            emb = self.encoder.encode_image(img)

        if emb.dim() > 1:
            emb = emb.squeeze(0)

        emb = emb.detach().cpu()
        self._image_embedding_cache[image_path] = emb
        return emb

    @staticmethod
    def _cosine_similarity(q: torch.Tensor, x: torch.Tensor) -> float:
        q = q.float().view(-1)
        x = x.float().view(-1)
        q = F.normalize(q, dim=0)
        x = F.normalize(x, dim=0)
        return float(torch.dot(q, x).item())

    def _retrieve_flat(self, retriever, query_emb, query_image_path, top_k):
        """Retrieve from flat KB"""
        exclude_indices = self.image_to_kb_indices.get(query_image_path, [])

        query_np = query_emb.detach().cpu().numpy().reshape(1, -1).astype("float32")
        scores, indices = retriever.search(
            query_np,
            top_k,
            exclude_indices=exclude_indices
        )

        valid_mask = indices[0] >= 0
        valid_indices = indices[0][valid_mask]
        valid_scores = scores[0][valid_mask]

        if len(valid_indices) == 0:
            return []

        results = []
        for idx, score in zip(valid_indices, valid_scores):
            meta = self.kb_metadata[int(idx)]
            results.append({
                **meta,
                "score": float(score),
                "diagnosis_label": meta.get("diagnosis_label", meta.get("diagnosis", "unknown")),
            })
        return results

    def _retrieve_concept(self, retriever, query_emb, query_image_path, top_k):
        """
        Two-stage retrieval:
        1) retrieve top concepts
        2) expand all images from those concepts
        3) re-rank all candidate images globally
        """
        query_np = query_emb.detach().cpu().numpy().astype("float32").reshape(1, -1)

        # Retrieve top concepts first.
        top_concepts = min(max(5, top_k // 2), len(retriever.metadata))
        concept_scores, concept_indices = retriever.search(query_np, top_concepts)

        exclude_set = {query_image_path} if query_image_path else set()
        candidates = []

        for concept_rank, (c_score, c_idx) in enumerate(
            zip(concept_scores[0], concept_indices[0]),
            start=1
        ):
            if int(c_idx) < 0:
                continue

            concept = retriever.metadata[int(c_idx)]
            image_paths = self._extract_image_paths(concept)

            for image_rank, img_path in enumerate(image_paths, start=1):
                if img_path in exclude_set:
                    continue

                image_emb = self._encode_image_path(img_path)
                image_score = self._cosine_similarity(query_emb, image_emb)

                # Weighted combination of concept confidence and image similarity.
                final_score = (
                    self.rerank_lambda * float(c_score)
                    + (1.0 - self.rerank_lambda) * float(image_score)
                )

                candidates.append({
                    **concept,
                    "image_path": img_path,
                    "score": float(final_score),
                    "concept_score": float(c_score),
                    "image_score": float(image_score),
                    "diagnosis_label": concept.get(
                        "diagnosis_label",
                        concept.get("diagnosis", "unknown")
                    ),
                    "concept_rank": concept_rank,
                    "image_rank": image_rank,
                })

        if not candidates:
            return self._fallback_concept_results(
                retriever=retriever,
                query_emb=query_emb,
                top_k=top_k,
                top_concepts=top_concepts,
            )

        candidates.sort(
            key=lambda x: (
                x["score"],
                x["concept_score"],
                x["image_score"],
            ),
            reverse=True,
        )

        return candidates[:top_k]

    def _fallback_concept_results(
        self,
        retriever,
        query_emb,
        top_k: int,
        top_concepts: int,
    ):
        """
        Safety fallback if a concept has no usable image paths.
        Returns concept-level results rather than failing the eval.
        """
        query_np = query_emb.detach().cpu().numpy().astype("float32").reshape(1, -1)
        concept_scores, concept_indices = retriever.search(query_np, top_concepts)

        results = []
        for c_score, c_idx in zip(concept_scores[0], concept_indices[0]):
            if int(c_idx) < 0:
                continue

            concept = retriever.metadata[int(c_idx)]
            image_paths = self._extract_image_paths(concept)
            rep_image = image_paths[0] if image_paths else concept.get("concept_id", f"concept_{int(c_idx)}")

            results.append({
                **concept,
                "image_path": rep_image,
                "score": float(c_score),
                "concept_score": float(c_score),
                "image_score": float(c_score),
                "diagnosis_label": concept.get(
                    "diagnosis_label",
                    concept.get("diagnosis", "unknown")
                ),
                "concept_rank": 1,
                "image_rank": 1,
            })

        return results[:top_k]