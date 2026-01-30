"""
Counterfactual reasoning
"""

import json
import torch
import faiss
import numpy as np
from PIL import Image
import sys
from pathlib import Path
import os
import argparse

# Fix imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.fusion.adaptive_fusion import AdaptiveFusion
from core.reasoning.counterfactuals.stability.retrieval import StabilityRetriever
from core.reasoning.counterfactuals.stability.runner import StabilityRunner
from core.reasoning.counterfactuals.diagnostics.scorer import CounterfactualScorer
from core.reasoning.counterfactuals.orchestrator import CounterfactualReasoner
from core.reasoning.counterfactuals.explanation.gemini_explainer import GeminiCounterfactualExplainer
from configs.inference_config import INFERENCE_CONFIG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-text", required=True)
    parser.add_argument("--query-image", required=True)
    parser.add_argument("--kb-dir", default=INFERENCE_CONFIG["kb_dir"])
    parser.add_argument("--gemini-api-key", default=os.getenv("GEMINI_API_KEY"), help="Gemini API Key for explanation")
    parser.add_argument("--output", default="outputs/inference/counterfactual_results.json", help="Path to save JSON output")
    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    KB_DIR = args.kb_dir
    QUERY_TEXT = args.query_text
    QUERY_IMAGE_PATH = args.query_image
    DEVICE = INFERENCE_CONFIG["device"]

    print("  Loading KB...")
    index = faiss.read_index(f"{KB_DIR}/index.faiss")
    with open(f"{KB_DIR}/metadata.json") as f:
        metadata = json.load(f)

    print("  Loading encoders + trained fusion...")
    encoder = BioMedCLIPEncoder(
        device=DEVICE,
        lora_path=INFERENCE_CONFIG.get("lora_path")
    )

    fusion = AdaptiveFusion().to(DEVICE)
    fusion.load_state_dict(
        torch.load(
            INFERENCE_CONFIG["fusion_path"],
            map_location=DEVICE
        )
    )
    fusion.eval()

    retriever = StabilityRetriever(index, metadata)

    # ---------------- Encode query (original) ----------------
    print("\n✓ Encoding query...")

    img = Image.open(QUERY_IMAGE_PATH).convert("RGB")

    with torch.no_grad():
        img_emb = encoder.encode_image(img).unsqueeze(0)
        txt_emb = encoder.encode_text(QUERY_TEXT).unsqueeze(0)

    # ---------------- Level 1: Stability (original) ----------------
    # ---------------- Orchestrator Execution ----------------
    print("\n✓ Initializing Orchestrator...")
    
    explainer = None
    if args.gemini_api_key:
        print("  + Gemini Explainer enabled")
        explainer = GeminiCounterfactualExplainer(args.gemini_api_key)
        
    stability_runner = StabilityRunner(retriever, fusion)
    scorer = CounterfactualScorer()
    
    reasoner = CounterfactualReasoner(
        stability_runner=stability_runner,
        scorer=scorer,
        explainer=explainer
    )

    print("\n✓ Running Counterfactual Reasoning Chain...")
    output = reasoner.run(img_emb, txt_emb)

    print("\n=== STABILITY OUTPUT ===")
    print(json.dumps(output["stability"], indent=2))

    print("\n=== DIAGNOSTIC SCORES ===")
    for s in output["ranked_hypotheses"]:
        print(s)

    full_output = {
        "query": {
            "text": QUERY_TEXT,
            "image": QUERY_IMAGE_PATH
        },
        "stability": output["stability"],
        "ranked_hypotheses": output["ranked_hypotheses"],
        "explanation": output.get("explanation")
    }

    if "explanation" in output:
        print("\n=== GEMINI EXPLANATION ===")
        print(json.dumps(output["explanation"], indent=2))

    # Save to file
    with open(args.output, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
