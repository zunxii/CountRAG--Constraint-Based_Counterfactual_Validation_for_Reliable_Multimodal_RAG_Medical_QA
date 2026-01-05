import numpy as np
from typing import Dict
from scripts.evaluation.counterfactual.metrics.entropy import shannon_entropy


def modality_attribution(
    baseline: Dict[str, float],
    no_text: Dict[str, float],
    no_image: Dict[str, float]
) -> Dict:
    """Compute how much each modality contributes to prediction"""
    baseline_ent = shannon_entropy(baseline)
    no_text_ent = shannon_entropy(no_text)
    no_image_ent = shannon_entropy(no_image)
    
    text_attr = abs(no_text_ent - baseline_ent)
    image_attr = abs(no_image_ent - baseline_ent)
    
    interaction = baseline_ent - (no_text_ent + no_image_ent) / 2
    
    return {
        "text_attribution": float(text_attr),
        "image_attribution": float(image_attr),
        "interaction_effect": float(interaction),
        "dominant_modality": "text" if text_attr > image_attr else "image",
        "dominance_ratio": float(max(text_attr, image_attr) / (min(text_attr, image_attr) + 1e-6))
    }


def modality_dependency(no_text: Dict, no_image: Dict) -> Dict:
    """Classify modality dependency pattern"""
    text_labels = set(no_image["distribution"].keys())
    image_labels = set(no_text["distribution"].keys())
    
    both = text_labels & image_labels
    text_only = text_labels - image_labels
    image_only = image_labels - text_labels
    
    return {
        "multimodal_support": len(both),
        "text_specific": len(text_only),
        "image_specific": len(image_only),
        "pattern": _classify_pattern(len(both), len(text_only), len(image_only))
    }


def _classify_pattern(both: int, text_only: int, image_only: int) -> str:
    """Classify modality interaction pattern"""
    total = both + text_only + image_only
    if total == 0:
        return "no_predictions"
    
    both_ratio = both / total
    
    if both_ratio > 0.7:
        return "strong_multimodal"
    elif both_ratio > 0.4:
        return "moderate_multimodal"
    elif text_only > image_only * 1.5:
        return "text_dominant"
    elif image_only > text_only * 1.5:
        return "image_dominant"
    else:
        return "balanced_unimodal"
