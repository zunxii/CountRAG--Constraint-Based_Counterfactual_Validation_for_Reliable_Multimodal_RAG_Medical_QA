"""
Main KB builder - 
Usage:
    python scripts/kb/build_kb.py --mode flat      # Original flat KB
    python scripts/kb/build_kb.py --mode concept   # NEW concept KB
"""
import torch
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.kb.builder import KBBuilder
from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.fusion.adaptive_fusion import AdaptiveFusion
from configs.kb_config import KB_BUILD_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Build Knowledge Base")
    parser.add_argument(
        "--mode",
        choices=["flat", "concept"],
        default="concept",
        help="KB architecture: flat (original) or concept (new, recommended)"
    )
    parser.add_argument(
        "--aggregation",
        choices=["mean", "max", "weighted"],
        default="mean",
        help="Image aggregation method for concept mode"
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix for output directory (e.g., '_v2')"
    )
    args = parser.parse_args()
    
    # Determine output directory
    if args.mode == "flat":
        output_dir = KB_BUILD_CONFIG["output_dir"]
    else:
        output_dir = KB_BUILD_CONFIG["output_dir"].replace("_v2", "_concept" + args.output_suffix)
    
    print("="*70)
    print(f"BUILDING KB - MODE: {args.mode.upper()}")
    print("="*70)
    print(f"Output: {output_dir}")
    if args.mode == "concept":
        print(f"Aggregation: {args.aggregation}")
    print()
    
    # Load trained fusion
    print("[1/3] Loading trained models...")
    fusion = AdaptiveFusion()
    fusion.load_state_dict(
        torch.load(
            KB_BUILD_CONFIG["fusion_path"],
            map_location=KB_BUILD_CONFIG["device"]
        )
    )
    fusion.eval()
    
    # Load encoders WITH LoRA
    image_encoder = BioMedCLIPEncoder(
        device=KB_BUILD_CONFIG["device"],
        lora_path=KB_BUILD_CONFIG["lora_path"]
    )
    
    text_encoder = BioMedCLIPEncoder(
        device=KB_BUILD_CONFIG["device"],
        lora_path=KB_BUILD_CONFIG["lora_path"]
    )
    
    print("✓ Models loaded")
    
    # Build KB
    print(f"\n[2/3] Building {args.mode} KB...")
    builder = KBBuilder(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        fusion_model=fusion,
        output_dir=output_dir,
        image_root=KB_BUILD_CONFIG["image_root"],
        device=KB_BUILD_CONFIG["device"],
        mode=args.mode,
        aggregation_method=args.aggregation,
    )
    
    builder.build(KB_BUILD_CONFIG["csv_path"])
    
    # Verify
    print("\n[3/3] Verifying KB...")
    from core.retrieval.retriever import KBRetriever
    
    retriever = KBRetriever(output_dir)
    stats = retriever.get_statistics()
    
    print("\nKB Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n✓ BUILD COMPLETE")
    print(f"✓ Saved to: {output_dir}")


if __name__ == "__main__":
    main()