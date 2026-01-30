

import torch
from peft import PeftModel
import open_clip
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.fusion.adaptive_fusion import AdaptiveFusion
from core.retrieval.retriever import KBRetriever
from core.kb.image_loader import ImageLoader
from core.embeddings.biomedclip import BioMedCLIPEncoder


def main():
    print("Attempting to load BioMedCLIP model...")
    try:
        encoder = BioMedCLIPEncoder(
            device="cpu",
            lora_path="outputs/models/trained_lora"
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    print("Attempting to load Fusion model...")
    try:
        fusion = AdaptiveFusion().to("cpu")
        fusion.load_state_dict(
            torch.load(
                "outputs/models/trained_fusion/fusion.pt",
                map_location="cpu"
            )
        )
        fusion.eval()
        print("Fusion model loaded successfully.")
    except Exception as e:
        print(f"An error occurred while loading Fusion model: {e}")
        return

    print("Attempting to load KBRetriever...")
    try:
        retriever = KBRetriever("outputs/kb/kb_final_v2")
        print("KBRetriever loaded successfully.")
    except Exception as e:
        print(f"An error occurred while loading KBRetriever: {e}")
        return

    print("Attempting to load ImageLoader...")
    try:
        image_loader = ImageLoader()
        print("ImageLoader loaded successfully.")
    except Exception as e:
        print(f"An error occurred while loading ImageLoader: {e}")
        return

    print("Attempting to encode query...")
    try:
        with torch.no_grad():
            txt_emb = encoder.encode_text("swollen eye").unsqueeze(0)
            img_emb = torch.zeros_like(txt_emb)
            query_emb = fusion(img_emb, txt_emb)
        print("Query encoded successfully.")
    except Exception as e:
        print(f"An error occurred while encoding query: {e}")
        return

    print("Attempting to search...")
    try:
        query_np = query_emb.cpu().numpy().astype("float32")
        scores, indices = retriever.search(query_np, 5)
        print("Search successful.")
    except Exception as e:
        print(f"An error occurred during search: {e}")
        return


if __name__ == "__main__":
    main()
