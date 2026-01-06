import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
from tqdm import tqdm
from collections import defaultdict

from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.fusion.adaptive_fusion import AdaptiveFusion
from core.kb.image_loader import ImageLoader
from core.retrieval.retriever import KBRetriever
from scripts.evaluation.retrieval.metrics import MetricsCalculator


class ModeEvaluator:
    """Evaluates different retrieval modes using evaluation queries"""
    
    def __init__(self, kb_dir: str, eval_dataset, device: str = "cpu"):
        """
        Args:
            kb_dir: Path to knowledge base
            eval_dataset: EvaluationQueryDataset instance
            device: Device for computation
        """
        self.device = device
        self.retriever = KBRetriever(kb_dir)
        self.kb_metadata = self.retriever.metadata
        self.eval_dataset = eval_dataset
        
        # Load models
        self.encoder = BioMedCLIPEncoder(device=device)
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
        
        Args:
            mode: One of "text", "image", "fusion"
            top_k: Number of results to retrieve
            
        Returns:
            Dictionary with aggregated metrics
        """
        all_metrics = defaultdict(list)

        for query in tqdm(self.eval_dataset, desc=f"Evaluating {mode}"):
            gt_label = query["diagnosis_label"]

            # Encode query based on mode
            query_emb = self._encode_query(query, mode)
            if query_emb is None:
                continue

            # Retrieve
            query_np = query_emb.cpu().numpy().reshape(1, -1)
            _, indices = self.retriever.search(query_np, top_k)

            # Get retrieved metadata
            retrieved = [self.kb_metadata[idx] for idx in indices]

            # Compute metrics
            metrics = self.metrics_calc.compute_all_metrics(
                retrieved, gt_label
            )

            for k, v in metrics.items():
                all_metrics[k].append(v)

        return {
            "metrics": {k: sum(v)/len(v) for k, v in all_metrics.items()},
            "num_queries": len(self.eval_dataset)
        }
    
    def _encode_query(self, query: Dict, mode: str):
        """
        Encode query based on mode.
        
        Args:
            query: Query dict with keys: image_path, combined_text
            mode: One of "text", "image", "fusion"
        """
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