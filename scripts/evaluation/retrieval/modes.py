"""
Mode evaluator - UPDATED to support concept KB
Replaces: scripts/evaluation/retrieval/modes.py
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
    """Evaluates retrieval modes - supports both flat and concept KBs"""
    
    def __init__(self, kb_dir: str, eval_dataset, device: str = "cpu"):
        self.device = device
        self.retriever = KBRetriever(kb_dir)
        self.kb_metadata = self.retriever.metadata
        self.kb_mode = self.retriever.kb_mode
        self.eval_dataset = eval_dataset
        
        print(f"Evaluating {self.kb_mode} KB")
        
        # Build lookup for exclusions (flat mode only)
        if self.kb_mode == "flat":
            self.image_to_kb_indices = defaultdict(list)
            for idx, entry in enumerate(self.kb_metadata):
                img_path = entry.get("image_path", "")
                self.image_to_kb_indices[img_path].append(idx)
        else:
            # For concept mode, use image_to_concept mapping
            self.image_to_concept = self.retriever.image_to_concept
        
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
        """Evaluate retrieval for a specific mode"""
        all_metrics = defaultdict(list)
        per_diagnosis_metrics = defaultdict(lambda: defaultdict(list))
        
        # For concept KB, change retriever mode
        if self.kb_mode == "concept":
            self.retriever = KBRetriever(self.retriever.kb_dir, mode=mode)
        
        for query in tqdm(self.eval_dataset, desc=f"Evaluating {mode}"):
            gt_label = query["diagnosis_label"]
            query_image_path = query["image_path"]
            
            # Encode query
            query_emb = self._encode_query(query, mode)
            if query_emb is None:
                continue
            
            # Get results based on KB mode
            if self.kb_mode == "flat":
                retrieved = self._retrieve_flat(
                    query_emb, query_image_path, top_k
                )
            else:
                retrieved = self._retrieve_concept(
                    query_emb, query_image_path, top_k
                )
            
            if not retrieved:
                continue
            
            # Compute metrics
            metrics = self.metrics_calc.compute_all_metrics(
                retrieved, gt_label
            )
            
            # Aggregate
            for k, v in metrics.items():
                all_metrics[k].append(v)
                per_diagnosis_metrics[gt_label][k].append(v)
        
        # Compute averages
        avg_metrics = {k: sum(v)/len(v) for k, v in all_metrics.items()}
        
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
    
    def _retrieve_flat(self, query_emb, query_image_path, top_k):
        """Retrieve from flat KB"""
        exclude_indices = self.image_to_kb_indices.get(query_image_path, [])
        
        query_np = query_emb.cpu().numpy().reshape(1, -1)
        scores, indices = self.retriever.search(
            query_np,
            top_k,
            exclude_indices=exclude_indices
        )
        
        valid_mask = indices[0] >= 0
        valid_indices = indices[0][valid_mask]
        
        if len(valid_indices) == 0:
            return []
        
        return [self.kb_metadata[int(idx)] for idx in valid_indices]
    
    def _retrieve_concept(self, query_emb, query_image_path, top_k):
        """Retrieve from concept KB"""
        query_np = query_emb.cpu().numpy()
        
        # Use search_images which expands concepts to images
        try:
            scores, results = self.retriever.search_images(
                query_np,
                top_k=top_k,
                concepts_to_retrieve=min(10, top_k),
                exclude_image_paths=[query_image_path]
            )
            return results
        except ValueError:
            # Fallback to concept-level search
            scores, indices = self.retriever.search(query_np, top_k)
            valid_mask = indices[0] >= 0
            valid_indices = indices[0][valid_mask]
            
            if len(valid_indices) == 0:
                return []
            
            return [self.kb_metadata[int(idx)] for idx in valid_indices]