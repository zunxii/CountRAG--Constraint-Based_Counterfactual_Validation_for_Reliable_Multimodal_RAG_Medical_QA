from collections import Counter
from typing import List, Dict
import numpy as np

from scripts.evaluation.counterfactual.analysis.perturbation import PerturbationAnalyzer
from scripts.evaluation.counterfactual.analysis.modality_effects import ModalityEffectsAnalyzer
from scripts.evaluation.counterfactual.analysis.statistical import StatisticalAnalyzer
from scripts.evaluation.counterfactual.analysis.diagnostic_profile import DiagnosticProfiler


class RobustnessAnalyzer:
    """Enhanced robustness analysis orchestrator"""
    
    def __init__(self):
        self.perturbation_analyzer = PerturbationAnalyzer()
        self.modality_analyzer = ModalityEffectsAnalyzer()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.diagnostic_profiler = DiagnosticProfiler()
    
    def analyze(self, stability_tests: List[Dict]) -> Dict:
        """Comprehensive robustness analysis"""
        return {
            "basic_metrics": self._compute_basic_metrics(stability_tests),
            "perturbation_analysis": self.perturbation_analyzer.analyze(stability_tests),
            "modality_effects": self.modality_analyzer.analyze(stability_tests),
            "statistical_tests": self.statistical_analyzer.analyze(stability_tests),
            "diagnostic_profiles": self.diagnostic_profiler.analyze(stability_tests)
        }
    
    def _compute_basic_metrics(self, tests: List[Dict]) -> Dict:
        """Compute basic stability metrics (backward compatibility)"""
        return {
            "avg_divergences": self._compute_avg_divergences(tests),
            "robustness_distribution": self._compute_robustness_dist(tests),
            "per_diagnosis_stability": self._analyze_per_diagnosis(tests)
        }
    
    def _compute_avg_divergences(self, tests: List[Dict]) -> Dict:
        """Average JS divergences"""
        divergences = {
            "no_text": [],
            "no_image": [],
            "noisy": []
        }
        
        for test in tests:
            js_divs = test["stability"]["js_divergence"]
            for key in divergences:
                if key in js_divs:
                    divergences[key].append(js_divs[key])
        
        return {
            key: float(np.mean(vals)) if vals else 0.0
            for key, vals in divergences.items()
        }
    
    def _compute_robustness_dist(self, tests: List[Dict]) -> Dict:
        """Robustness level distribution"""
        levels = [
            test["stability"]["robustness_level"]
            for test in tests
        ]
        return dict(Counter(levels))
    
    def _analyze_per_diagnosis(self, tests: List[Dict]) -> Dict:
        """Per-diagnosis stability"""
        by_diagnosis = {}
        
        for test in tests:
            diag = test["diagnosis"]
            if diag not in by_diagnosis:
                by_diagnosis[diag] = []
            
            by_diagnosis[diag].append(
                test["stability"]["robustness_level"]
            )
        
        return {
            diag: {
                "count": int(len(levels)),  # Convert to int
                "high": int(levels.count("high")),
                "medium": int(levels.count("medium")),
                "low": int(levels.count("low"))
            }
            for diag, levels in by_diagnosis.items()
        }
