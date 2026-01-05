import numpy as np
from typing import Dict, List
from scripts.evaluation.counterfactual.metrics.attribution import (
    modality_attribution,
    modality_dependency
)


class ModalityEffectsAnalyzer:
    """Analyzes modality contributions and interactions"""
    
    def analyze(self, stability_tests: List[Dict]) -> Dict:
        """Comprehensive modality analysis"""
        attributions = []
        dependencies = []
        
        for test in stability_tests:
            if "baseline_distribution" not in test:
                continue
            
            baseline = test["baseline_distribution"]["distribution"]
            
            # Extract modality-specific distributions
            no_text_dist = test.get("no_text_distribution", {}).get("distribution", {})
            no_image_dist = test.get("no_image_distribution", {}).get("distribution", {})
            
            if no_text_dist and no_image_dist:
                attr = modality_attribution(baseline, no_text_dist, no_image_dist)
                attributions.append(attr)
                
                dep = modality_dependency(
                    {"distribution": no_text_dist},
                    {"distribution": no_image_dist}
                )
                dependencies.append(dep)
        
        if not attributions:
            return {"status": "insufficient_data"}
        
        return {
            "attribution_summary": self._summarize_attributions(attributions),
            "dependency_patterns": self._summarize_dependencies(dependencies),
            "interaction_analysis": self._analyze_interactions(attributions)
        }
    
    def _summarize_attributions(self, attributions: List[Dict]) -> Dict:
        """Summarize modality attribution scores"""
        text_attrs = [a["text_attribution"] for a in attributions]
        image_attrs = [a["image_attribution"] for a in attributions]
        interactions = [a["interaction_effect"] for a in attributions]
        
        return {
            "text_contribution": {
                "mean": float(np.mean(text_attrs)),
                "std": float(np.std(text_attrs)),
                "median": float(np.median(text_attrs))
            },
            "image_contribution": {
                "mean": float(np.mean(image_attrs)),
                "std": float(np.std(image_attrs)),
                "median": float(np.median(image_attrs))
            },
            "interaction_strength": {
                "mean": float(np.mean(interactions)),
                "std": float(np.std(interactions)),
                "positive_ratio": float(np.mean([i > 0 for i in interactions]))
            },
            "dominant_modality_distribution": self._count_dominant(attributions)
        }
    
    def _summarize_dependencies(self, dependencies: List[Dict]) -> Dict:
        """Summarize dependency patterns"""
        from collections import Counter
        patterns = [d["pattern"] for d in dependencies]
        return dict(Counter(patterns))
    
    def _analyze_interactions(self, attributions: List[Dict]) -> Dict:
        """Analyze modality interaction effects"""
        interactions = [a["interaction_effect"] for a in attributions]
        
        synergy = int(sum(1 for i in interactions if i > 0.05))  # Convert to int
        redundancy = int(sum(1 for i in interactions if i < -0.05))
        independent = len(interactions) - synergy - redundancy
        
        return {
            "synergistic_cases": synergy,
            "redundant_cases": redundancy,
            "independent_cases": independent,
            "predominant_pattern": self._predominant_interaction(
                synergy, redundancy, independent
            )
        }

    
    def _count_dominant(self, attributions: List[Dict]) -> Dict:
        """Count dominant modality distribution"""
        from collections import Counter
        dominant = [a["dominant_modality"] for a in attributions]
        return dict(Counter(dominant))
    
    def _predominant_interaction(self, synergy: int, redundancy: int, 
                                 independent: int) -> str:
        """Determine predominant interaction pattern"""
        total = synergy + redundancy + independent
        if total == 0:
            return "unknown"
        
        ratios = {
            "synergistic": synergy / total,
            "redundant": redundancy / total,
            "independent": independent / total
        }
        
        return max(ratios, key=ratios.get)
