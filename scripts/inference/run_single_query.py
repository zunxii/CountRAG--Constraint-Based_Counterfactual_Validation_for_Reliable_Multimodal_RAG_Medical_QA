"""
Single query inference - FIXED to ensure distinct top-K results
"""

import json
import numpy as np
from pathlib import Path
import faiss
import torch
import sys
import argparse

# Fix imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.fusion.adaptive_fusion import AdaptiveFusion
from core.kb.image_loader import ImageLoader
from core.retrieval.retriever import KBRetriever
from configs.inference_config import INFERENCE_CONFIG


def main():
    parser = argparse.ArgumentParser(
        description="Single query inference with distinct top-K retrieval"
    )
    parser.add_argument("--query-text", required=True, help="Query text")
    parser.add_argument("--query-image", default=None, help="Query image path")
    parser.add_argument("--kb-dir", default=INFERENCE_CONFIG["kb_dir"], 
                       help="Knowledge base directory")
    parser.add_argument("--top-k", type=int, default=INFERENCE_CONFIG["top_k"],
                       help="Number of results to retrieve")
    args = parser.parse_args()

    print("="*70)
    print("SINGLE QUERY INFERENCE (Distinct Results)")
    print("="*70)
    
    print(f"\n📚 Loading KB from {args.kb_dir}...")
    retriever = KBRetriever(args.kb_dir)
    print(f"✓ KB loaded: {len(retriever.metadata)} entries")

    print(f"\n🤖 Loading encoder (LoRA) + trained fusion...")
    encoder = BioMedCLIPEncoder(
        device=INFERENCE_CONFIG["device"],
        lora_path=INFERENCE_CONFIG.get("lora_path")
    )

    fusion = AdaptiveFusion().to(INFERENCE_CONFIG["device"])
    fusion.load_state_dict(
        torch.load(
            INFERENCE_CONFIG["fusion_path"],
            map_location=INFERENCE_CONFIG["device"]
        )
    )
    fusion.eval()
    print("✓ Models loaded")

    image_loader = ImageLoader()

    # -------------------------
    # Encode query
    # -------------------------
    print("\n🔍 Encoding query...")

    with torch.no_grad():
        txt_emb = encoder.encode_text(args.query_text).unsqueeze(0)

        if args.query_image is not None:
            img = image_loader.load(args.query_image)
            img_emb = encoder.encode_image(img).unsqueeze(0)
        else:
            # text-only query → zero image vector
            img_emb = torch.zeros_like(txt_emb)

        query_emb = fusion(img_emb, txt_emb)

    print("✓ Query encoded")

    # -------------------------
    # Search with distinctness guarantee
    # -------------------------
    print(f"\n🔎 Retrieving top-{args.top_k} distinct results...")
    
    query_np = query_emb.cpu().numpy().astype("float32")
    
    # Check if query image matches any KB entry
    exclude_indices = []
    if args.query_image is not None:
        query_img_path = str(Path(args.query_image).resolve())
        for idx, entry in enumerate(retriever.metadata):
            kb_img_path = str(Path(entry["image_path"]).resolve())
            if kb_img_path == query_img_path:
                exclude_indices.append(idx)
        
        if exclude_indices:
            print(f"⚠ Excluding {len(exclude_indices)} self-match(es)")
    
    scores, indices = retriever.search(query_np, args.top_k, 
                                       exclude_indices=exclude_indices)

    # -------------------------
    # Display results
    # -------------------------
    print("\n" + "="*70)
    print(f"RETRIEVED CASES (Top-{args.top_k} Distinct)")
    print("="*70 + "\n")

    seen_cases = set()
    displayed = 0
    
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        idx = int(idx)
        
        # Skip invalid indices (padding)
        if idx < 0:
            break
        
        case = retriever.metadata[idx]
        case_id = case.get("case_id", f"case_{idx}")
        
        # Double-check distinctness (should already be handled)
        if case_id in seen_cases:
            print(f"⚠ Warning: Duplicate case_id detected: {case_id}")
            continue
        
        seen_cases.add(case_id)

        print(f"Rank {rank}")
        print(f"  Score: {score:.4f}")
        print(f"  Case ID: {case_id}")
        print(f"  Diagnosis: {case['diagnosis_label']}")
        print(f"  Image: {case['image_path']}")
        
        text_preview = case['clinical_text']['combined'][:200]
        print(f"  Text: {text_preview}{'...' if len(case['clinical_text']['combined']) > 200 else ''}")
        print("-" * 70)
        
        displayed += 1
    
    print(f"\n✓ Displayed {displayed} distinct results")
    print("="*70)


if __name__ == "__main__":
    main()