from __future__ import annotations

import numpy as np
from typing import Dict, List
from collections import Counter

from scripts.evaluation.counterfactual.metrics.attribution import (
    modality_attribution,
    modality_dependency
)


class ModalityEffectsAnalyzer:
    """Analyze modality contributions and interactions from per-query distributions."""

    def analyze(self, stability_tests: List[Dict]) -> Dict:
        attributions = []
        dependencies = []

        for test in stability_tests:
            if "baseline_distribution" not in test:
                continue

            baseline = test.get("baseline_distribution", {}).get("distribution", {})
            no_text_dist = test.get("no_text_distribution", {}).get("distribution", {})
            no_image_dist = test.get("no_image_distribution", {}).get("distribution", {})

            if not baseline or not no_text_dist or not no_image_dist:
                continue

            attr = modality_attribution(baseline, no_text_dist, no_image_dist)
            attributions.append(attr)

            dep = modality_dependency(
                {"distribution": no_text_dist},
                {"distribution": no_image_dist},
            )
            dependencies.append(dep)

        if not attributions:
            return {"status": "insufficient_data"}

        return {
            "sample_size": len(attributions),
            "attribution_summary": self._summarize_attributions(attributions),
            "dependency_patterns": self._summarize_dependencies(dependencies),
            "interaction_analysis": self._analyze_interactions(attributions),
        }

    def _summarize_attributions(self, attributions: List[Dict]) -> Dict:
        text_attrs = [a["text_attribution"] for a in attributions]
        image_attrs = [a["image_attribution"] for a in attributions]
        interactions = [a["interaction_effect"] for a in attributions]

        return {
            "text_contribution": {
                "mean": float(np.mean(text_attrs)),
                "std": float(np.std(text_attrs)),
                "median": float(np.median(text_attrs)),
            },
            "image_contribution": {
                "mean": float(np.mean(image_attrs)),
                "std": float(np.std(image_attrs)),
                "median": float(np.median(image_attrs)),
            },
            "interaction_strength": {
                "mean": float(np.mean(interactions)),
                "std": float(np.std(interactions)),
                "positive_ratio": float(np.mean([i > 0 for i in interactions])),
            },
            "dominant_modality_distribution": self._count_dominant(attributions),
        }

    def _summarize_dependencies(self, dependencies: List[Dict]) -> Dict:
        patterns = [d["pattern"] for d in dependencies]
        return dict(Counter(patterns))

    def _analyze_interactions(self, attributions: List[Dict]) -> Dict:
        interactions = [a["interaction_effect"] for a in attributions]

        synergy = int(sum(1 for i in interactions if i > 0.05))
        redundancy = int(sum(1 for i in interactions if i < -0.05))
        independent = len(interactions) - synergy - redundancy

        return {
            "synergistic_cases": synergy,
            "redundant_cases": redundancy,
            "independent_cases": independent,
            "predominant_pattern": self._predominant_interaction(synergy, redundancy, independent),
        }

    def _count_dominant(self, attributions: List[Dict]) -> Dict:
        dominant = [a["dominant_modality"] for a in attributions]
        return dict(Counter(dominant))

    def _predominant_interaction(self, synergy: int, redundancy: int, independent: int) -> str:
        counts = {"synergy": synergy, "redundancy": redundancy, "independent": independent}
        return max(counts, key=counts.get) if counts else "unknown"