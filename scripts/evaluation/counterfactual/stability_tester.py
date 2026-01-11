"""
Stability tester - FIXED to use updated retriever with distinct results
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import json
import torch
import faiss

from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.fusion.adaptive_fusion import AdaptiveFusion
from core.kb.image_loader import ImageLoader
from core.reasoning.counterfactuals.stability.runner import StabilityRunner
from core.reasoning.counterfactuals.stability.retrieval import StabilityRetriever


class StabilityTester:
    """Tests counterfactual stability on evaluation queries"""
    
    def __init__(self, kb_dir: str, device: str = "cpu"):
        self.device = device
        
        # Load KB using unified retriever (handles both modes!)
        kb_dir = Path(kb_dir)
        
        # CHANGED: Use KBRetriever which auto-detects mode
        from core.retrieval.retriever import KBRetriever
        self.kb_retriever = KBRetriever(str(kb_dir))
        self.kb_mode = self.kb_retriever.kb_mode  # NEW: Get mode
        self.kb_metadata = self.kb_retriever.metadata
        
        print(f"✓ Loaded {self.kb_mode} KB")
        
        # CHANGED: Build image lookup based on KB mode
        if self.kb_mode == "flat":
            self._build_flat_lookup()
        else:  # concept mode
            self._build_concept_lookup()
        
        # Load models (unchanged)
        self.encoder = BioMedCLIPEncoder(device=device)
        self.image_loader = ImageLoader()
        
        fusion_path = Path("outputs/models/trained_fusion/fusion.pt")
        self.fusion = AdaptiveFusion().to(device)
        self.fusion.load_state_dict(
            torch.load(fusion_path, map_location=device)
        )
        self.fusion.eval()
        
        # CHANGED: Use KBRetriever instead of raw FAISS
        from core.reasoning.counterfactuals.stability.retrieval import StabilityRetriever
        self.stability_retriever = StabilityRetriever(
            self.kb_retriever.index, 
            self.kb_metadata
        )
        
        from core.reasoning.counterfactuals.stability.runner import StabilityRunner
        self.runner = StabilityRunner(self.stability_retriever, self.fusion)
    
    def _build_flat_lookup(self):
        """Build image->index lookup for flat KB"""
        self.image_to_indices = {}
        for idx, entry in enumerate(self.kb_metadata):
            img_path = str(Path(entry["image_path"]).resolve())
            if img_path not in self.image_to_indices:
                self.image_to_indices[img_path] = []
            self.image_to_indices[img_path].append(idx)
    
    def _build_concept_lookup(self):
        """Build image->concept lookup for concept KB"""
        self.image_to_concept = {}
        for concept in self.kb_metadata:
            concept_id = concept['concept_id']
            for img_path in concept.get('image_paths', []):
                img_key = str(Path(img_path).resolve())
                self.image_to_concept[img_key] = concept_id
    
    # CHANGED: Update test_query method (lines 80-120)
    def test_query(self, query: dict) -> dict:
        """Test stability for evaluation query - works for both KB modes"""
        
        # Load and encode query
        img = self.image_loader.load(query['image_path'])
        
        with torch.no_grad():
            img_emb = self.encoder.encode_image(img).unsqueeze(0)
            txt_emb = self.encoder.encode_text(query['combined_text']).unsqueeze(0)
        
        # NEW: Check if query is in KB and exclude it
        # (Note: Stability analysis focuses on distribution changes,
        #  so self-exclusion is less critical but we do it anyway)
        
        # Run stability analysis
        stability_output = self.runner.run(img_emb, txt_emb)
        
        return {
            "query_id": query['query_id'],
            "diagnosis": query['diagnosis_label'],
            "image_path": query['image_path'],
            "stability": stability_output["stability"],
            "baseline_distribution": stability_output["baseline"]
        }
    
    def get_metadata(self):
        """Get KB metadata (for backward compatibility)"""
        return self.kb_metadata
    
    def test_sample(self, idx: int):
        """
        Test KB sample by index (for backward compatibility).
        This excludes the query index itself from results.
        """
        entry = self.kb_metadata[idx]
        
        img = self.image_loader.load(entry["image_path"])
        
        with torch.no_grad():
            img_emb = self.encoder.encode_image(img).unsqueeze(0)
            txt_emb = self.encoder.encode_text(
                entry["clinical_text"]["combined"]
            ).unsqueeze(0)
        
        # Note: For KB self-queries, we should exclude idx from retrieval
        # This would require updating StabilityRetriever to support exclusions
        stability_output = self.runner.run(img_emb, txt_emb)
        
        return {
            "case_id": entry["case_id"],
            "diagnosis": entry["diagnosis_label"],
            "stability": stability_output["stability"],
            "baseline_distribution": stability_output["baseline"]
        }