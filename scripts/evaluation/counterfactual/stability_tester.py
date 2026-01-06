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
        
        # Load KB
        kb_dir = Path(kb_dir)
        self.index = faiss.read_index(str(kb_dir / "index.faiss"))
        
        with open(kb_dir / "metadata.json") as f:
            self.kb_metadata = json.load(f)
        
        # Load models
        self.encoder = BioMedCLIPEncoder(device=device)
        self.image_loader = ImageLoader()
        
        # Load fusion
        fusion_path = Path("outputs/models/trained_fusion/fusion.pt")
        self.fusion = AdaptiveFusion().to(device)
        self.fusion.load_state_dict(
            torch.load(fusion_path, map_location=device)
        )
        self.fusion.eval()
        
        # Setup stability runner
        retriever = StabilityRetriever(self.index, self.kb_metadata)
        self.runner = StabilityRunner(retriever, self.fusion)
    
    def test_query(self, query: dict) -> dict:
        """
        Test stability for an evaluation query.
        
        Args:
            query: Dict with keys: image_path, combined_text, diagnosis_label, query_id
            
        Returns:
            Stability test results
        """
        # Load and encode query
        img = self.image_loader.load(query['image_path'])
        
        with torch.no_grad():
            img_emb = self.encoder.encode_image(img).unsqueeze(0)
            txt_emb = self.encoder.encode_text(query['combined_text']).unsqueeze(0)
        
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
        """Test KB sample by index (for backward compatibility)"""
        entry = self.kb_metadata[idx]
        
        img = self.image_loader.load(entry["image_path"])
        
        with torch.no_grad():
            img_emb = self.encoder.encode_image(img).unsqueeze(0)
            txt_emb = self.encoder.encode_text(
                entry["clinical_text"]["combined"]
            ).unsqueeze(0)
        
        stability_output = self.runner.run(img_emb, txt_emb)
        
        return {
            "case_id": entry["case_id"],
            "diagnosis": entry["diagnosis_label"],
            "stability": stability_output["stability"],
            "baseline_distribution": stability_output["baseline"]
        }