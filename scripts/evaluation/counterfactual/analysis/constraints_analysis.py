"""
Constraints Analysis for Evaluation
Location: scripts/evaluation/counterfactual/analysis/constraints_analysis.py (NEW FILE)

Analyzes constraint patterns across evaluation queries.
"""
import numpy as np
from typing import Dict, List
from collections import Counter, defaultdict


class ConstraintsAnalyzer:
    """Analyzes constraint patterns from stability tests"""
    
    def analyze(self, stability_tests: List[Dict]) -> Dict:
        """Comprehensive constraints analysis"""
        
        # Skip if no constraints
        if not stability_tests or "constraints" not in stability_tests[0]:
            return {"status": "no_constraints"}
        
        # Extract constraint data
        cluster_dists = []
        modality_consistencies = []
        boundary_analyses = []
        evidence_diversities = []
        distribution_checks = []
        
        for test in stability_tests:
            if "constraints" not in test or "error" in test.get("constraints", {}):
                continue
            
            c = test["constraints"]
            
            # Cluster distribution
            if "cluster_distribution" in c:
                cluster_dists.append(c["cluster_distribution"])
            
            # Modality consistency
            if "modality_consistency" in c:
                modality_consistencies.append(c["modality_consistency"])
            
            # Boundary analysis
            if "boundary_analysis" in c:
                boundary_analyses.append(c["boundary_analysis"])
            
            # Evidence diversity
            if "evidence_diversity" in c:
                evidence_diversities.append(c["evidence_diversity"])
            
            # Distribution check
            if "distribution_check" in c:
                distribution_checks.append(c["distribution_check"])
        
        if not cluster_dists:
            return {"status": "no_valid_constraints"}
        
        return {
            "sample_size": len(cluster_dists),
            "cluster_distribution_analysis": self._analyze_cluster_distributions(cluster_dists),
            "modality_consistency_analysis": self._analyze_modality_consistency(modality_consistencies),
            "boundary_analysis_summary": self._analyze_boundary_patterns(boundary_analyses),
            "evidence_diversity_summary": self._analyze_evidence_diversity(evidence_diversities),
            "distribution_check_summary": self._analyze_distribution_checks(distribution_checks)
        }
    
    def _analyze_cluster_distributions(self, cluster_dists: List[Dict]) -> Dict:
        """Analyze cluster distribution patterns"""
        num_clusters = [cd.get("num_clusters", 0) for cd in cluster_dists]
        
        return {
            "avg_num_clusters": float(np.mean(num_clusters)),
            "std_num_clusters": float(np.std(num_clusters)),
            "min_clusters": int(np.min(num_clusters)),
            "max_clusters": int(np.max(num_clusters)),
            "median_clusters": float(np.median(num_clusters))
        }
    
    def _analyze_modality_consistency(self, consistencies: List[Dict]) -> Dict:
        """Analyze modality consistency patterns"""
        if not consistencies:
            return {"status": "no_data"}
        
        similarities = [mc.get("cosine_similarity", 0.0) for mc in consistencies]
        levels = [mc.get("consistency_level", "unknown") for mc in consistencies]
        
        return {
            "avg_similarity": float(np.mean(similarities)),
            "std_similarity": float(np.std(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
            "level_distribution": dict(Counter(levels))
        }
    
    def _analyze_boundary_patterns(self, boundaries: List[Dict]) -> Dict:
        """Analyze boundary proximity patterns"""
        if not boundaries:
            return {"status": "no_data"}
        
        near_boundary_count = sum(1 for b in boundaries if b.get("near_boundary", False))
        margins = [b.get("margin", 0.0) for b in boundaries if b.get("margin") is not None]
        
        result = {
            "near_boundary_count": int(near_boundary_count),
            "near_boundary_ratio": float(near_boundary_count / len(boundaries)),
        }
        
        if margins:
            result.update({
                "avg_margin": float(np.mean(margins)),
                "std_margin": float(np.std(margins)),
                "min_margin": float(np.min(margins)),
                "max_margin": float(np.max(margins))
            })
        
        return result
    
    def _analyze_evidence_diversity(self, diversities: List[Dict]) -> Dict:
        """Analyze evidence diversity patterns"""
        if not diversities:
            return {"status": "no_data"}
        
        unique_images = [ed.get("unique_images", 0) for ed in diversities]
        unique_cases = [ed.get("unique_cases", 0) for ed in diversities]
        levels = [ed.get("level", "unknown") for ed in diversities]
        
        return {
            "avg_unique_images": float(np.mean(unique_images)),
            "avg_unique_cases": float(np.mean(unique_cases)),
            "level_distribution": dict(Counter(levels))
        }
    
    def _analyze_distribution_checks(self, checks: List[Dict]) -> Dict:
        """Analyze distribution check patterns"""
        if not checks:
            return {"status": "no_data"}
        
        in_distribution = sum(1 for dc in checks if dc.get("in_distribution", False))
        distances = [dc.get("distance", 0.0) for dc in checks]
        
        return {
            "in_distribution_count": int(in_distribution),
            "in_distribution_ratio": float(in_distribution / len(checks)),
            "avg_distance": float(np.mean(distances)),
            "std_distance": float(np.std(distances))
        }