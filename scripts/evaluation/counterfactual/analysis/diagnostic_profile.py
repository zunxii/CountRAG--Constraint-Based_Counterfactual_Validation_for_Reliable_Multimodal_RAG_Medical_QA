import numpy as np
from typing import Dict, List
from collections import defaultdict


class DiagnosticProfiler:
    """Analyze stability patterns per diagnosis"""
    
    def analyze(self, stability_tests: List[Dict]) -> Dict:
        """Create stability profiles for each diagnosis"""
        by_diagnosis = defaultdict(list)
        
        for test in stability_tests:
            diag = test.get("diagnosis", "unknown")
            by_diagnosis[diag].append(test)
        
        profiles = {}
        for diag, tests in by_diagnosis.items():
            profiles[diag] = self._create_profile(diag, tests)
        
        profiles["_comparative"] = self._comparative_analysis(profiles)
        
        return profiles
    
    def _create_profile(self, diagnosis: str, tests: List[Dict]) -> Dict:
        """Create stability profile for one diagnosis"""
        divergences = {
            "no_text": [],
            "no_image": [],
            "noisy": []
        }
        robustness_levels = []
        entropies = []
        
        for test in tests:
            stab = test.get("stability", {})
            js_div = stab.get("js_divergence", {})
            
            for key in divergences:
                if key in js_div:
                    divergences[key].append(js_div[key])
            
            robustness_levels.append(stab.get("robustness_level", "unknown"))
            
            dist = test.get("baseline_distribution", {}).get("distribution", {})
            if dist:
                from scripts.evaluation.counterfactual.metrics.entropy import shannon_entropy
                entropies.append(shannon_entropy(dist))
        
        return {
            "sample_count": len(tests),
            "stability_scores": {
                key: {
                    "mean": float(np.mean(vals)) if vals else 0.0,
                    "std": float(np.std(vals)) if vals else 0.0,
                    "median": float(np.median(vals)) if vals else 0.0
                }
                for key, vals in divergences.items()
            },
            "robustness_distribution": self._count_levels(robustness_levels),
            "predominant_robustness": self._predominant(robustness_levels),
            "entropy_stats": {
                "mean": float(np.mean(entropies)) if entropies else 0.0,
                "std": float(np.std(entropies)) if entropies else 0.0
            },
            "stability_rank": None
        }
    
    def _count_levels(self, levels: List[str]) -> Dict:
        """Count robustness level distribution"""
        from collections import Counter
        # Convert Counter values to regular ints
        return {k: int(v) for k, v in Counter(levels).items()}
    
    def _predominant(self, levels: List[str]) -> str:
        """Most common robustness level"""
        if not levels:
            return "unknown"
        from collections import Counter
        return Counter(levels).most_common(1)[0][0]
    
    def _comparative_analysis(self, profiles: Dict) -> Dict:
        """Compare diagnoses against each other"""
        diagnoses = [d for d in profiles.keys() if d != "_comparative"]
        
        avg_divs = {}
        for diag in diagnoses:
            prof = profiles[diag]
            avg_div = np.mean([
                prof["stability_scores"]["no_text"]["mean"],
                prof["stability_scores"]["no_image"]["mean"],
                prof["stability_scores"]["noisy"]["mean"]
            ])
            avg_divs[diag] = avg_div
        
        ranked = sorted(avg_divs.items(), key=lambda x: x[1])
        
        for rank, (diag, _) in enumerate(ranked, 1):
            profiles[diag]["stability_rank"] = rank
        
        return {
            "most_stable": ranked[0][0] if ranked else "none",
            "least_stable": ranked[-1][0] if ranked else "none",
            "stability_ranking": [d for d, _ in ranked],
            "stability_variance": float(np.std(list(avg_divs.values()))) if avg_divs else 0.0
        }
