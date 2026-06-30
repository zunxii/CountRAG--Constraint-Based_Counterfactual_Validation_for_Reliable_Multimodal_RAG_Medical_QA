from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.fusion.adaptive_fusion import AdaptiveFusion
from core.kb.image_loader import ImageLoader
from core.reasoning.constraints.extractor import ConstraintExtractor
from core.reasoning.counterfactuals.stability.runner import StabilityRunner
from core.reasoning.counterfactuals.stability.retrieval import StabilityRetriever
from core.reasoning.counterfactuals.stability.stability_metrics import stability_report
from core.retrieval.retriever import KBRetriever
from scripts.utils.eval_query_loader import load_eval_queries


VARIANT_ORDER = ["baseline", "text_neutral", "image_neutral", "noisy"]
RADAR_AXES = [
    "evidence_concentration",
    "modality_consistency",
    "decision_boundary_proximity",
    "evidence_diversity",
    "ood_validity",
]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


class ModalityConstraintStudy:
    """Run constraint extraction across baseline / text-neutral / image-neutral / noisy queries."""

    def __init__(
        self,
        contract: Dict[str, Any],
        kb_dir: str,
        device: str = "cpu",
        num_samples: int = 200,
        seed: int = 42,
        include_noisy: bool = True,
    ) -> None:
        self.contract = contract
        self.kb_dir = Path(kb_dir)
        self.device = device
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.include_noisy = bool(include_noisy)

        self.output_root = Path(self.contract.get("paths", {}).get("outputs_root", "outputs"))
        self.image_root = Path(self.contract.get("paths", {}).get("image_root", "data/images"))
        self.eval_csv = Path(self.contract.get("paths", {}).get("eval_split_csv", "data/processed/splits/eval.csv"))

        self._dataset = load_eval_queries(str(self.eval_csv))
        self._selected_queries = self._sample_queries(self._dataset, self.num_samples, self.seed)

        self._setup_models()

    @staticmethod
    def _sample_queries(dataset, n: int, seed: int) -> List[Dict[str, Any]]:
        rng = random.Random(seed)
        queries = list(dataset)
        if not queries:
            return []
        n = min(int(n), len(queries))
        if n >= len(queries):
            return queries
        return rng.sample(queries, n)

    def _setup_models(self) -> None:
        lora_path = self.contract.get("paths", {}).get("models", {}).get("lora_dir", "outputs/models/trained_lora")
        fusion_path = self.contract.get("paths", {}).get("models", {}).get("fusion_model_file", "outputs/models/trained_fusion/fusion.pt")

        self.encoder = BioMedCLIPEncoder(device=self.device, lora_path=lora_path)

        self.fusion = AdaptiveFusion().to(self.device)
        fusion_state = torch.load(fusion_path, map_location=self.device)
        self.fusion.load_state_dict(fusion_state)
        self.fusion.eval()

        self.kb_retriever = KBRetriever(str(self.kb_dir))
        self.stability_retriever = StabilityRetriever(self.kb_retriever.index, self.kb_retriever.metadata)
        self.image_loader = ImageLoader()

        self._kb_centroid = self._compute_kb_centroid()
        self._support_reference = self._compute_support_reference()
        self._ood_threshold_95 = self._compute_ood_threshold_95()

        top_k = int(self.contract.get("retrieval", {}).get("top_k_for_inference", 10))
        self.runner = StabilityRunner(
            self.stability_retriever,
            self.fusion,
            contract=self.contract,
            device=self.device,
            top_k=top_k,
            kb_centroid=self._kb_centroid,
            support_reference=self._support_reference,
            ood_threshold=self._ood_threshold_95,
        )
        self.constraint_extractor: Optional[ConstraintExtractor] = self.runner.constraint_extractor

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

        for entry in self.kb_retriever.metadata:
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
        supports: List[float] = []
        for entry in self.kb_retriever.metadata:
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
        return float(np.median(np.asarray(supports, dtype=np.float64)))

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

    def _resolve_image_path(self, image_path: str) -> Path:
        raw = Path(str(image_path).strip())
        if raw.is_absolute() and raw.exists():
            return raw
        if raw.exists():
            return raw
        candidate = self.image_root / raw.name
        if candidate.exists():
            return candidate
        return raw

    def _encode_query(self, query: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Path, str]:
        question = str(query.get("combined_text") or query.get("question") or "").strip()
        img_path = self._resolve_image_path(str(query.get("image_path", "")))
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found for query: {img_path}")

        img = self.image_loader.load(str(img_path))
        with torch.no_grad():
            img_emb = self.encoder.encode_image(img).unsqueeze(0).to(self.device)
            txt_emb = self.encoder.encode_text(question).unsqueeze(0).to(self.device)
        return img_emb, txt_emb, img_path, question

    @staticmethod
    def _top_probs(distribution: Dict[str, float]) -> Tuple[float, float, str]:
        if not distribution:
            return 0.0, 0.0, "unknown"
        items = sorted(((str(k), _to_float(v, 0.0)) for k, v in distribution.items()), key=lambda kv: kv[1], reverse=True)
        top_label = items[0][0]
        top1 = items[0][1]
        top2 = items[1][1] if len(items) > 1 else 0.0
        return top1, top2, top_label

    def _summarize_retrieval(self, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
        top = retrieved[0] if retrieved else {}
        metadata = top.get("metadata", {}) if isinstance(top, dict) else {}
        return {
            "top_label": top.get("diagnosis_label", metadata.get("diagnosis_label", "unknown")) if isinstance(top, dict) else "unknown",
            "top_score": _to_float(top.get("score", 0.0)) if isinstance(top, dict) else 0.0,
            "support": int(metadata.get("num_images", len(metadata.get("image_paths", [])) if isinstance(metadata, dict) else 0)) if isinstance(metadata, dict) else 0,
            "num_retrieved": len(retrieved),
            "retrieved_labels": [r.get("diagnosis_label", "unknown") for r in retrieved[:5]],
        }

    def _evaluate_variant(
        self,
        *,
        variant_name: str,
        img_emb: torch.Tensor,
        txt_emb: torch.Tensor,
        fused_emb: torch.Tensor,
        baseline_distribution: Dict[str, float],
        reference_tag: str,
    ) -> Dict[str, Any]:
        if self.constraint_extractor is None:
            raise RuntimeError("ConstraintExtractor is not available")

        retrieved = self.runner._call_retriever(fused_emb)
        dist_info = self.runner._maybe_distribution(fused_emb)
        variant_distribution = dict(dist_info.get("distribution", {}))
        top1_prob, top2_prob, top_label = self._top_probs(variant_distribution)

        stability_pair = stability_report(baseline_distribution, {reference_tag: variant_distribution})
        js_div = float(stability_pair.get("js_divergence", {}).get(reference_tag, 0.0))
        robustness_level = str(stability_pair.get("robustness_level", "high"))

        query_distance = self.runner._query_distance(fused_emb)

        constraints = self.constraint_extractor.extract(
            retrieved_metadata=[r.get("metadata", {}) for r in retrieved],
            retrieved=retrieved,
            distribution=variant_distribution,
            img_emb=img_emb,
            txt_emb=txt_emb,
            top1_prob=top1_prob,
            top2_prob=top2_prob,
            query_distance=query_distance,
            percentile_95=self._ood_threshold_95,
            support_reference=self._support_reference,
            robustness_level=robustness_level,
            max_js_divergence=js_div,
        )

        return {
            "variant": variant_name,
            "distribution": variant_distribution,
            "distribution_info": dist_info,
            "retrieved": retrieved,
            "top1_prob": top1_prob,
            "top2_prob": top2_prob,
            "top_label": top_label,
            "query_distance": query_distance,
            "js_divergence": js_div,
            "robustness_level": robustness_level,
            "constraints": constraints,
            "retrieval_summary": self._summarize_retrieval(retrieved),
        }

    def evaluate_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        img_emb, txt_emb, img_path, question = self._encode_query(query)
        diagnosis = str(query.get("diagnosis_label") or query.get("category") or "unknown")
        query_id = int(query.get("query_id", 0))

        with torch.no_grad():
            base_fused = self.runner._infer(img_emb, txt_emb)

        baseline_retrieved = self.runner._call_retriever(base_fused)
        baseline_dist_info = self.runner._maybe_distribution(base_fused)
        baseline_distribution = dict(baseline_dist_info.get("distribution", {}))
        baseline_top1, baseline_top2, baseline_top_label = self._top_probs(baseline_distribution)
        baseline_query_distance = self.runner._query_distance(base_fused)

        baseline_record = {
            "query_id": query_id,
            "diagnosis_label": diagnosis,
            "image_path": str(img_path),
            "question": question,
            "variant": "baseline",
            "modality_source": "original",
            "distribution": baseline_distribution,
            "top1_prob": baseline_top1,
            "top2_prob": baseline_top2,
            "top_label": baseline_top_label,
            "query_distance": baseline_query_distance,
            "constraints": None,
            "retrieved": baseline_retrieved,
            "retrieval_summary": self._summarize_retrieval(baseline_retrieved),
            "stability": None,
            "js_divergence": {},
            "robustness_level": "unknown",
            "baseline_reference": baseline_distribution,
            "delta_from_baseline": {axis: 0.0 for axis in RADAR_AXES},
            "aggregate_delta": 0.0,
        }

        records = [baseline_record]

        variant_distributions: Dict[str, Dict[str, float]] = {}
        variant_outputs: Dict[str, Dict[str, Any]] = {}

        with torch.no_grad():
            no_text_emb = self.runner._neutral_text_emb(txt_emb)
            no_text_fused = self.runner._infer(img_emb, no_text_emb)
        no_text_variant = self._evaluate_variant(
            variant_name="text_neutral",
            img_emb=img_emb,
            txt_emb=no_text_emb,
            fused_emb=no_text_fused,
            baseline_distribution=baseline_distribution,
            reference_tag="text_neutral",
        )
        variant_distributions["text_neutral"] = no_text_variant["distribution"]
        variant_outputs["text_neutral"] = no_text_variant

        with torch.no_grad():
            no_image_emb = self.runner._neutral_image_emb(img_emb)
            no_image_fused = self.runner._infer(no_image_emb, txt_emb)
        no_image_variant = self._evaluate_variant(
            variant_name="image_neutral",
            img_emb=no_image_emb,
            txt_emb=txt_emb,
            fused_emb=no_image_fused,
            baseline_distribution=baseline_distribution,
            reference_tag="image_neutral",
        )
        variant_distributions["image_neutral"] = no_image_variant["distribution"]
        variant_outputs["image_neutral"] = no_image_variant

        if self.include_noisy:
            scale = float(getattr(self.runner, "_default_noisy_scale", 0.05))
            with torch.no_grad():
                noisy_base = self.runner._infer(img_emb, txt_emb)
                noisy_fused = noisy_base + torch.randn_like(noisy_base) * scale
            noisy_variant = self._evaluate_variant(
                variant_name="noisy",
                img_emb=img_emb,
                txt_emb=txt_emb,
                fused_emb=noisy_fused,
                baseline_distribution=baseline_distribution,
                reference_tag="noisy",
            )
            noisy_variant["noise_scale"] = scale
            variant_distributions["noisy"] = noisy_variant["distribution"]
            variant_outputs["noisy"] = noisy_variant

        baseline_stability = stability_report(baseline_distribution, variant_distributions)
        baseline_constraints = self.constraint_extractor.extract(
            retrieved_metadata=[r.get("metadata", {}) for r in baseline_retrieved],
            retrieved=baseline_retrieved,
            distribution=baseline_distribution,
            img_emb=img_emb,
            txt_emb=txt_emb,
            top1_prob=baseline_top1,
            top2_prob=baseline_top2,
            query_distance=baseline_query_distance,
            percentile_95=self._ood_threshold_95,
            support_reference=self._support_reference,
            robustness_level=str(baseline_stability.get("robustness_level", "high")),
            max_js_divergence=float(max(baseline_stability.get("js_divergence", {}).values() or [0.0])),
        )

        baseline_record["constraints"] = baseline_constraints
        baseline_record["stability"] = baseline_stability
        baseline_record["js_divergence"] = baseline_stability.get("js_divergence", {})
        baseline_record["robustness_level"] = str(baseline_stability.get("robustness_level", "high"))

        for variant_name in ("text_neutral", "image_neutral", "noisy"):
            if variant_name not in variant_outputs:
                continue
            rec = variant_outputs[variant_name]
            records.append(
                {
                    "query_id": query_id,
                    "diagnosis_label": diagnosis,
                    "image_path": str(img_path),
                    "question": question,
                    "variant": variant_name,
                    "modality_source": (
                        "image_only_survives"
                        if variant_name == "text_neutral"
                        else "text_only_survives"
                        if variant_name == "image_neutral"
                        else f"gaussian_sigma_{rec.get('noise_scale', 0.0)}"
                    ),
                    **rec,
                }
            )

        baseline_scores = baseline_constraints.get("scores", {}) if isinstance(baseline_constraints, dict) else {}
        baseline_aggregate = _to_float(baseline_constraints.get("aggregate_score", 0.0) if isinstance(baseline_constraints, dict) else 0.0)
        baseline_violation = bool(baseline_constraints.get("overall_violation", False) if isinstance(baseline_constraints, dict) else False)

        for rec in records:
            if rec["variant"] == "baseline":
                continue
            scores = rec.get("constraints", {}).get("scores", {}) if isinstance(rec.get("constraints"), dict) else {}
            rec["delta_from_baseline"] = {
                axis: _to_float(scores.get(axis, 0.0)) - _to_float(baseline_scores.get(axis, 0.0))
                for axis in RADAR_AXES
            }
            rec["aggregate_delta"] = _to_float(rec.get("constraints", {}).get("aggregate_score", 0.0)) - baseline_aggregate
            rec["violation_delta"] = int(bool(baseline_violation)) - int(bool(rec.get("constraints", {}).get("overall_violation", False)))

        return {
            "query_id": query_id,
            "diagnosis_label": diagnosis,
            "image_path": str(img_path),
            "question": question,
            "records": records,
            "baseline": baseline_record,
        }

    def run(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        total = len(self._selected_queries)
        for idx, query in enumerate(self._selected_queries, start=1):
            qid = query.get("query_id", f"{idx}")
            print(f"[{idx}/{total}] query_id={qid}", flush=True)
            results.append(self.evaluate_query(query))
        return results


def flatten_records(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in results:
        for record in item.get("records", []):
            constraints = record.get("constraints", {}) if isinstance(record.get("constraints"), dict) else {}
            scores = constraints.get("scores", {}) if isinstance(constraints, dict) else {}
            violations = constraints.get("violations", {}) if isinstance(constraints, dict) else {}
            row = {
                "query_id": item.get("query_id"),
                "diagnosis_label": item.get("diagnosis_label"),
                "image_path": item.get("image_path"),
                "question": item.get("question"),
                "variant": record.get("variant"),
                "modality_source": record.get("modality_source"),
                "top_label": record.get("top_label"),
                "top1_prob": _to_float(record.get("top1_prob", 0.0)),
                "top2_prob": _to_float(record.get("top2_prob", 0.0)),
                "query_distance": _to_float(record.get("query_distance", 0.0)),
                "js_divergence": _to_float(record.get("js_divergence", 0.0)),
                "robustness_level": record.get("robustness_level", "unknown"),
                "aggregate_score": _to_float(constraints.get("aggregate_score", 0.0)),
                "overall_violation": int(bool(constraints.get("overall_violation", False))),
                "noise_scale": _to_float(record.get("noise_scale", 0.0)) if "noise_scale" in record else "",
                "delta_aggregate": _to_float(record.get("aggregate_delta", 0.0)),
                "violation_delta": int(record.get("violation_delta", 0)) if "violation_delta" in record else 0,
                "retrieved_labels": "|".join(record.get("retrieval_summary", {}).get("retrieved_labels", [])),
            }
            for axis in RADAR_AXES:
                row[axis] = _to_float(scores.get(axis, 0.0))
                row[f"{axis}_violation"] = int(bool(violations.get(axis, False)))
                row[f"delta_{axis}"] = _to_float(record.get("delta_from_baseline", {}).get(axis, 0.0))
            rows.append(row)
    return rows


def summarize_by_variant(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("variant", "unknown")), []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for variant, items in grouped.items():
        if not items:
            continue
        summary = {
            "variant": variant,
            "num_queries": len(items),
            "overall_violation_rate": float(np.mean([float(i.get("overall_violation", 0)) for i in items])),
            "aggregate_score_mean": float(np.mean([_to_float(i.get("aggregate_score", 0.0)) for i in items])),
            "aggregate_score_std": float(np.std([_to_float(i.get("aggregate_score", 0.0)) for i in items])),
            "mean_js": float(np.mean([_to_float(i.get("js_divergence", 0.0)) for i in items])),
            "mean_top1_prob": float(np.mean([_to_float(i.get("top1_prob", 0.0)) for i in items])),
            "mean_query_distance": float(np.mean([_to_float(i.get("query_distance", 0.0)) for i in items])),
        }
        for axis in RADAR_AXES:
            summary[f"{axis}_mean"] = float(np.mean([_to_float(i.get(axis, 0.0)) for i in items]))
            summary[f"{axis}_std"] = float(np.std([_to_float(i.get(axis, 0.0)) for i in items]))
            summary[f"{axis}_violation_rate"] = float(np.mean([float(i.get(f"{axis}_violation", 0)) for i in items]))
            summary[f"delta_{axis}_mean"] = float(np.mean([_to_float(i.get(f"delta_{axis}", 0.0)) for i in items]))
        summary_rows.append(summary)

    summary_rows.sort(key=lambda x: VARIANT_ORDER.index(x["variant"]) if x["variant"] in VARIANT_ORDER else 999)
    return summary_rows


def prepare_radar_rows(summary_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    radar_rows: List[Dict[str, Any]] = []
    for row in summary_rows:
        radar_rows.append(
            {
                "variant": row["variant"],
                **{axis: row[f"{axis}_mean"] for axis in RADAR_AXES},
                "aggregate_score_mean": row.get("aggregate_score_mean", 0.0),
                "overall_violation_rate": row.get("overall_violation_rate", 0.0),
                "mean_js": row.get("mean_js", 0.0),
            }
        )
    return radar_rows


def save_json(path: Path, obj: Any) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_safe_json(obj), f, indent=2, ensure_ascii=False)


def save_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    _ensure_dir(path.parent)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    if fieldnames is None:
        ordered = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
        fieldnames = ordered
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _safe_json(row.get(k, "")) for k in fieldnames})