from collections import Counter
from math import log, exp
from typing import Any, Dict, List, Optional

import numpy as np

from .cluster_distribution import cluster_distribution_constraint
from .modality_consistency import modality_consistency_constraint
from .boundary_analysis import boundary_analysis_constraint
from .evidence_diversity import evidence_diversity_constraint
from .distribution_check import distribution_check_constraint


class ConstraintExtractor:
    DEFAULT_WEIGHTS = {
        "evidence_concentration": 0.25,
        "modality_consistency": 0.25,
        "decision_boundary_proximity": 0.20,
        "evidence_diversity": 0.15,
        "ood_validity": 0.15,
    }

    DEFAULT_THRESHOLDS = {
        "evidence_concentration": 0.5,
        "modality_consistency": 0.5,
        "decision_boundary_proximity": 0.5,
        "evidence_diversity": 0.5,
        "ood_validity": 0.5,
    }

    def __init__(self, contract: Optional[Dict[str, Any]] = None):
        self.contract = contract or {}
        constraints_cfg = self.contract.get("constraints", {})
        agg_cfg = constraints_cfg.get("aggregation", {})
        axes_cfg = constraints_cfg.get("axes", {})

        self.weights = dict(self.DEFAULT_WEIGHTS)
        for axis, value in (agg_cfg.get("weights") or {}).items():
            if axis in self.weights:
                self.weights[axis] = float(value)

        self.thresholds = dict(self.DEFAULT_THRESHOLDS)
        for axis, cfg in axes_cfg.items():
            if isinstance(cfg, dict) and "threshold" in cfg and axis in self.thresholds:
                self.thresholds[axis] = float(cfg["threshold"])

        self.overall_threshold = float(agg_cfg.get("overall_violation_threshold", 0.5))

    @staticmethod
    def _normalized_entropy(dist: Dict[str, float]) -> float:
        if not dist:
            return 0.0
        vals = np.asarray(list(dist.values()), dtype=np.float64)
        vals = vals[vals > 0]
        if vals.size == 0:
            return 0.0
        vals = vals / vals.sum()
        ent = float(-(vals * np.log(vals + 1e-12)).sum())
        max_ent = log(len(vals)) if len(vals) > 1 else 1.0
        return float(ent / max_ent) if max_ent > 0 else 0.0

    @staticmethod
    def _label_entropy(retrieved_metadata: List[Dict[str, Any]]) -> float:
        labels = [m.get("diagnosis_label", "unknown") for m in retrieved_metadata if isinstance(m, dict)]
        if not labels:
            return 0.0
        counts = Counter(labels)
        vals = np.asarray(list(counts.values()), dtype=np.float64)
        vals = vals / vals.sum()
        ent = float(-(vals * np.log(vals + 1e-12)).sum())
        max_ent = log(len(vals)) if len(vals) > 1 else 1.0
        return float(ent / max_ent) if max_ent > 0 else 0.0

    @staticmethod
    def _prototype_support(top_metadata: Optional[Dict[str, Any]]) -> float:
        if not isinstance(top_metadata, dict):
            return 0.0
        if "num_images" in top_metadata:
            try:
                return float(top_metadata["num_images"])
            except Exception:
                pass
        if isinstance(top_metadata.get("image_paths"), list):
            return float(len([p for p in top_metadata["image_paths"] if p]))
        return 1.0

    def extract(
        self,
        retrieved_metadata: list,
        img_emb,
        txt_emb,
        top1_prob: float,
        top2_prob: float,
        query_distance: float,
        percentile_95: float,
        baseline_distribution: Optional[Dict[str, float]] = None,
        distribution: Optional[Dict[str, float]] = None,
        retrieved: Optional[List[Dict[str, Any]]] = None,
        support_reference: float | None = None,
        robustness_level: str = "high",
        max_js_divergence: float = 0.0,
    ):
        # Accept both 'baseline_distribution' (from runner.py) and legacy 'distribution'
        dist_input: Dict[str, float] = baseline_distribution or distribution or {}

        # ── Constraint scores ────────────────────────────────────────────────
        concentration = 1.0 - self._normalized_entropy(dist_input)
        modality      = modality_consistency_constraint(img_emb, txt_emb)
        boundary      = boundary_analysis_constraint(top1_prob, top2_prob)

        top_support = self._prototype_support(retrieved_metadata[0] if retrieved_metadata else None)
        ref         = float(support_reference) if support_reference is not None else 1.0
        diversity   = float(np.clip(min(1.0, top_support / max(ref, 1e-9)), 0.0, 1.0))

        dist_check   = distribution_check_constraint(query_distance, percentile_95)
        ratio        = float(query_distance) / max(float(percentile_95), 1e-9)
        ood_validity = 1.0 / (1.0 + np.exp(4.0 * (ratio - 1.0)))

        scores = {
            "evidence_concentration":       float(np.clip(concentration, 0.0, 1.0)),
            "modality_consistency":         float(np.clip((modality["cosine_similarity"] + 1.0) / 2.0, 0.0, 1.0)),
            "decision_boundary_proximity":  float(np.clip(1.0 / (1.0 + exp(-12.0 * (boundary["margin"] - 0.05))), 0.0, 1.0)),
            "evidence_diversity":           float(np.clip(diversity, 0.0, 1.0)),
            "ood_validity":                 float(np.clip(ood_validity, 0.0, 1.0)),
        }

        violations       = {k: bool(v < self.thresholds.get(k, 0.5)) for k, v in scores.items()}
        aggregate        = float(
            sum(scores[k] * self.weights.get(k, 0.0) for k in scores)
            / max(sum(self.weights.get(k, 0.0) for k in scores), 1e-9)
        )
        critical_failures = sum(1 for v in violations.values() if v)

        # ── Overall violation: four independent triggers ──────────────────────
        #
        # THE BUG IN THE PREVIOUS VERSION:
        # Trigger 3 was: (unstable AND critical_failures >= 1)
        # This never fired for itichy eyelid / skin growth / swollen eye because
        # they had critical_failures = 0 — none of the 5 constraint axes (concentration,
        # modality_consistency, boundary_proximity, diversity, ood_validity) actually
        # measures JSD. The constraint axes and the JSD probing are completely
        # decoupled by design. So a case can have no_image JSD = 0.43 (robustness=low)
        # while all 5 axes pass, giving critical_failures = 0 and Trigger 3 = False.
        #
        # THE FIX:
        # robustness_level="low" is itself a direct trigger, with no additional
        # condition on critical_failures. This correctly cross-links the JSD probing
        # result into the overall_violation flag.
        # robustness_level="medium" + any single constraint axis failure also triggers,
        # since medium robustness means the system is borderline and an axis failure
        # confirms it is unsafe.
        #
        # Why not flag medium robustness unconditionally?
        # skin rash has medium robustness (JSD 0.11-0.12) but aggregate 0.71 and
        # no axis failures — it is a genuinely borderline case where the system is
        # uncertain but not clearly broken. Requiring a constraint failure to confirm
        # before flagging is the right conservative choice for medium cases.
        overall_violation = bool(
            aggregate < self.overall_threshold              # T1: weighted score failed
            or critical_failures >= 2                       # T2: multiple axes failed
            or robustness_level == "low"                    # T3: JSD probing says unstable
            or (robustness_level == "medium"                # T4: borderline + any axis
                and critical_failures >= 1)
        )

        return {
            "cluster_distribution": cluster_distribution_constraint(retrieved_metadata),
            "modality_consistency": modality,
            "boundary_analysis":    boundary,
            "evidence_diversity":   evidence_diversity_constraint(retrieved_metadata),
            "distribution_check":   dist_check,
            "scores":               scores,
            "violations":           violations,
            "aggregate_score":      aggregate,
            "overall_violation":    overall_violation,
            "thresholds":           self.thresholds,
            "weights":              self.weights,
            "robustness_level":     robustness_level,
            "max_js_divergence":    max_js_divergence,
        }