"""
Mode evaluator - FIXED to exclude self-matches and ensure distinct results
"""
import sys
from pathlib import Path
from typing import Dict
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
from tqdm import tqdm

from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.fusion.adaptive_fusion import AdaptiveFusion
from core.kb.image_loader import ImageLoader
from core.retrieval.retriever import KBRetriever
from scripts.evaluation.retrieval.metrics import MetricsCalculator


class ModeEvaluator:
    """Evaluates different retrieval modes using evaluation queries"""
    
    def __init__(self, kb_dir: str, eval_dataset, device: str = "cpu"):
        self.device = device
        self.retriever = KBRetriever(kb_dir)
        self.kb_metadata = self.retriever.metadata
        self.eval_dataset = eval_dataset
        
        # Build lookup: image_path -> KB indices
        # This helps exclude queries that match KB entries
        self.image_to_kb_indices = defaultdict(list)
        for idx, entry in enumerate(self.kb_metadata):
            img_path = entry.get("image_path", "")
            self.image_to_kb_indices[img_path].append(idx)
        
        # Load models
        self.encoder = BioMedCLIPEncoder(
            device=device,
            lora_path="outputs/models/trained_lora"
        )
        self.image_loader = ImageLoader()
        
        # Load fusion if available
        fusion_path = Path("outputs/models/trained_fusion/fusion.pt")
        if fusion_path.exists():
            self.fusion = AdaptiveFusion().to(device)
            self.fusion.load_state_dict(
                torch.load(fusion_path, map_location=device)
            )
            self.fusion.eval()
        else:
            self.fusion = None
        
        self.metrics_calc = MetricsCalculator()
    
    def evaluate_mode(self, mode: str, top_k: int = 20):
        """
        Evaluate retrieval for a specific mode using evaluation queries.
        NOW EXCLUDES SELF-MATCHES and ensures DISTINCT RESULTS.
        """
        all_metrics = defaultdict(list)
        per_diagnosis_metrics = defaultdict(lambda: defaultdict(list))

        for query in tqdm(self.eval_dataset, desc=f"Evaluating {mode}"):
            gt_label = query["diagnosis_label"]
            query_image_path = query["image_path"]

            # Encode query based on mode
            query_emb = self._encode_query(query, mode)
            if query_emb is None:
                continue

            # Find indices to exclude (queries that appear in KB)
            exclude_indices = self.image_to_kb_indices.get(query_image_path, [])
            
            # Retrieve with exclusions
            query_np = query_emb.cpu().numpy().reshape(1, -1)
            scores, indices = self.retriever.search(
                query_np, 
                top_k,
                exclude_indices=exclude_indices
            )

            # Filter out any invalid indices (-1 padding)
            valid_mask = indices[0] >= 0
            valid_indices = indices[0][valid_mask]
            
            if len(valid_indices) == 0:
                continue  # Skip if no valid results
            
            # Get retrieved metadata
            retrieved = [self.kb_metadata[int(idx)] for idx in valid_indices]

            # Compute metrics
            metrics = self.metrics_calc.compute_all_metrics(
                retrieved, gt_label
            )

            # Aggregate overall metrics
            for k, v in metrics.items():
                all_metrics[k].append(v)
            
            # Track per-diagnosis metrics
            for k, v in metrics.items():
                per_diagnosis_metrics[gt_label][k].append(v)

        # Compute averages
        avg_metrics = {k: sum(v)/len(v) for k, v in all_metrics.items()}
        
        # Compute per-diagnosis averages
        per_diag_avg = {}
        for diag, metrics_dict in per_diagnosis_metrics.items():
            per_diag_avg[diag] = {
                k: sum(v)/len(v) for k, v in metrics_dict.items()
            }

        return {
            "metrics": avg_metrics,
            "num_queries": len(self.eval_dataset),
            "per_diagnosis_metrics": per_diag_avg
        }
    
    def _encode_query(self, query: Dict, mode: str):
        """Encode query based on mode"""
        with torch.no_grad():
            if mode == "text":
                return self.encoder.encode_text(query["combined_text"])
            
            elif mode == "image":
                img = self.image_loader.load(query["image_path"])
                return self.encoder.encode_image(img)
            
            elif mode == "fusion":
                if self.fusion is None:
                    return None
                
                img = self.image_loader.load(query["image_path"])
                img_emb = self.encoder.encode_image(img).unsqueeze(0)
                txt_emb = self.encoder.encode_text(
                    query["combined_text"]
                ).unsqueeze(0)
                
                return self.fusion(img_emb, txt_emb).squeeze(0)
        
        return None