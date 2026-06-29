from __future__ import annotations

from typing import Dict
from scripts.evaluation.counterfactual.metrics.divergence import js_divergence


def _normalize(dist: Dict[str, float]) -> Dict[str, float]:
    if not dist:
        return {}
    total = sum(float(v) for v in dist.values())
    if total <= 0:
        return {}
    return {k: float(v) / total for k, v in dist.items() if float(v) >= 0}


def modality_attribution(
    baseline: Dict[str, float],
    no_text: Dict[str, float],
    no_image: Dict[str, float],
) -> Dict:
    """
    Attribution based on distribution shift, not entropy.
    This is much more stable for your retrieval distributions.
    """
    baseline = _normalize(baseline)
    no_text = _normalize(no_text)
    no_image = _normalize(no_image)

    if not baseline or not no_text or not no_image:
        return {
            "text_attribution": 0.0,
            "image_attribution": 0.0,
            "interaction_effect": 0.0,
            "dominant_modality": "image",
            "dominance_ratio": 0.0,
        }

    text_attr = js_divergence(baseline, no_text)
    image_attr = js_divergence(baseline, no_image)

    # Positive when removing both modalities changes the distribution more than
    # the average single-modality removal.
    interaction = js_divergence(no_text, no_image) - 0.5 * (text_attr + image_attr)

    dominant = "text" if text_attr > image_attr else "image"
    ratio = max(text_attr, image_attr) / (min(text_attr, image_attr) + 1e-9)

    return {
        "text_attribution": float(text_attr),
        "image_attribution": float(image_attr),
        "interaction_effect": float(interaction),
        "dominant_modality": dominant,
        "dominance_ratio": float(ratio),
    }


def modality_dependency(no_text: Dict, no_image: Dict) -> Dict:
    """
    Keep this simple and stable: compare label support overlap.
    """
    text_set = set(no_text.get("distribution", {}).keys())
    image_set = set(no_image.get("distribution", {}).keys())

    both = text_set & image_set
    text_only = text_set - image_set
    image_only = image_set - text_set

    total = len(both) + len(text_only) + len(image_only)
    if total == 0:
        pattern = "no_predictions"
    else:
        both_ratio = len(both) / total
        if both_ratio > 0.7:
            pattern = "strong_multimodal"
        elif both_ratio > 0.4:
            pattern = "moderate_multimodal"
        elif len(text_only) > len(image_only) * 1.5:
            pattern = "text_dominant"
        elif len(image_only) > len(text_only) * 1.5:
            pattern = "image_dominant"
        else:
            pattern = "balanced_unimodal"

    return {
        "multimodal_support": len(both),
        "text_specific": len(text_only),
        "image_specific": len(image_only),
        "pattern": pattern,
    }


# def _classify_pattern(both: int, text_only: int, image_only: int) -> str:
#     """Classify modality interaction pattern"""
#     total = both + text_only + image_only
#     if total == 0:
#         return "no_predictions"
    
#     both_ratio = both / total
    
#     if both_ratio > 0.7:
#         return "strong_multimodal"
#     elif both_ratio > 0.4:
#         return "moderate_multimodal"
#     elif text_only > image_only * 1.5:
#         return "text_dominant"
#     elif image_only > text_only * 1.5:
#         return "image_dominant"
#     else:
#         return "balanced_unimodal"
