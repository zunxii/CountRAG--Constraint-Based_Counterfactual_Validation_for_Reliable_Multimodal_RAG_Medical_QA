"""
Perturbation helpers for counterfactual-style probing.

"""
import torch
import torch.nn.functional as F


def remove_text(img_emb: torch.Tensor, txt_emb: torch.Tensor):
    """
    Deprecated: returns zero text embedding (OOD).
    StabilityRunner uses manifold-valid neutral embeddings instead.
    Kept for backward compatibility.
    """
    return img_emb, torch.zeros_like(txt_emb)


def remove_image(img_emb: torch.Tensor, txt_emb: torch.Tensor):
    """
    Deprecated: returns zero image embedding (OOD).
    StabilityRunner uses manifold-valid neutral embeddings instead.
    Kept for backward compatibility.
    """
    return torch.zeros_like(img_emb), txt_emb


def add_noise(emb: torch.Tensor, scale: float = 0.05) -> torch.Tensor:
    """Add Gaussian noise at the specified scale (paper default σ=0.05)."""
    return emb + scale * torch.randn_like(emb)


def apply_neutral_text(img_emb: torch.Tensor, neutral_text: torch.Tensor):
    """Apply semantically neutral text embedding (manifold-valid)."""
    return img_emb, neutral_text.expand_as(img_emb)


def apply_neutral_image(neutral_image: torch.Tensor, txt_emb: torch.Tensor):
    """Apply semantically neutral image embedding (manifold-valid)."""
    return neutral_image.expand_as(txt_emb), txt_emb