"""
Stability tester - UPDATED to include constraints in output
Location: scripts/evaluation/counterfactual/stability_tester.py

Changes:
1. Return constraints in test results
2. Maintain backward compatibility
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
        kb_dir = Path(kb_dir)
        
        # Load KB using unified retriever
        from core.retrieval.retriever import KBRetriever
        self.kb_retriever = KBRetriever(str(kb_dir))
        self.kb_mode = self.kb_retriever.kb_mode
        self.kb_metadata = self.kb_retriever.metadata
        
        print(f"✓ Loaded {self.kb_mode} KB")
        
        # Build image lookup based on KB mode
        if self.kb_mode == "flat":
            self._build_flat_lookup()
        else:
            self._build_concept_lookup()
        
        # Load models
        self.encoder = BioMedCLIPEncoder(device=device)
        self.image_loader = ImageLoader()
        
        fusion_path = Path("outputs/models/trained_fusion/fusion.pt")
        self.fusion = AdaptiveFusion().to(device)
        self.fusion.load_state_dict(
            torch.load(fusion_path, map_location=device)
        )
        self.fusion.eval()
        
        # Setup retriever for stability
        from core.reasoning.counterfactuals.stability.retrieval import StabilityRetriever
        self.stability_retriever = StabilityRetriever(
            self.kb_retriever.index, 
            self.kb_metadata
        )
        
        # Create runner (will auto-detect constraints)
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
    
    def test_query(self, query: dict) -> dict:
        """
        Test stability for evaluation query.
        
        NEW: Returns constraints alongside stability
        """
        # Load and encode query
        img = self.image_loader.load(query['image_path'])
        
        with torch.no_grad():
            img_emb = self.encoder.encode_image(img).unsqueeze(0)
            txt_emb = self.encoder.encode_text(query['combined_text']).unsqueeze(0)
        
        # Run stability analysis (now includes constraints)
        stability_output = self.runner.run(img_emb, txt_emb)
        
        # Build result with constraints
        result = {
            "query_id": query['query_id'],
            "diagnosis": query['diagnosis_label'],
            "image_path": query['image_path'],
            "stability": stability_output["stability"],
            "baseline_distribution": stability_output["baseline"]
        }
        
        # NEW: Include constraints if available
        if "constraints" in stability_output:
            result["constraints"] = stability_output["constraints"]
        
        return result
    
    def get_metadata(self):
        """Get KB metadata"""
        return self.kb_metadata
    
    def test_sample(self, idx: int):
        """Test KB sample by index"""
        entry = self.kb_metadata[idx]
        
        img = self.image_loader.load(entry["image_path"])
        
        with torch.no_grad():
            img_emb = self.encoder.encode_image(img).unsqueeze(0)
            txt_emb = self.encoder.encode_text(
                entry["clinical_text"]["combined"]
            ).unsqueeze(0)
        
        stability_output = self.runner.run(img_emb, txt_emb)
        
        result = {
            "case_id": entry["case_id"],
            "diagnosis": entry["diagnosis_label"],
            "stability": stability_output["stability"],
            "baseline_distribution": stability_output["baseline"]
        }
        
        # NEW: Include constraints if available
        if "constraints" in stability_output:
            result["constraints"] = stability_output["constraints"]
        
        return result