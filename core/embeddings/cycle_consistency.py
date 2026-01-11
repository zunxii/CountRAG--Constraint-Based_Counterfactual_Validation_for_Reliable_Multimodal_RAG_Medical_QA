"""
Cycle Consistency Loss for Medical Cross-Domain Alignment

Forces patient questions and clinical descriptions to create 
consistent image representations via cycle reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CycleConsistencyLoss(nn.Module):
    """
    Given text embeddings and image embeddings in the same CLIP space:
      - question_txt_emb (B, D)
      - clinical_txt_emb (B, D)
      - image_emb (B, D)

    We encourage question and clinical to have similar similarity to the image:
      loss = MSE( cosine(question, image), cosine(clinical, image) )
    """
    def __init__(self):
        super().__init__()

    def forward(self, question_txt_emb, clinical_txt_emb, image_emb):
        # normalize
        q = F.normalize(question_txt_emb, dim=-1)
        c = F.normalize(clinical_txt_emb, dim=-1)
        im = F.normalize(image_emb, dim=-1)

        # per-sample cosine similarities
        sim_q_im = (q * im).sum(dim=-1)   # (B,)
        sim_c_im = (c * im).sum(dim=-1)   # (B,)

        return F.mse_loss(sim_q_im, sim_c_im)

class AlignmentLoss(nn.Module):
    """
    Soft cross-entropy alignment between question-text and clinical-text embeddings.
    Works in the shared space. Encourages question[i] to be closest to clinical[i].
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, question_txt_emb, clinical_txt_emb):
        q = F.normalize(question_txt_emb, dim=-1)
        c = F.normalize(clinical_txt_emb, dim=-1)

        logits = (q @ c.T) / self.temperature  # (B,B)
        labels = torch.arange(q.size(0), device=q.device)
        loss = F.cross_entropy(logits, labels)
        return loss
