import json
import torch
import faiss
import numpy as np
from PIL import Image
import sys
from pathlib import Path
import os
import argparse
from dataclasses import asdict, is_dataclass

# Fix imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.fusion.adaptive_fusion import AdaptiveFusion
from core.reasoning.counterfactuals.stability.retrieval import StabilityRetriever
from core.reasoning.counterfactuals.stability.runner import StabilityRunner
from core.reasoning.counterfactuals.diagnostics.scorer import CounterfactualScorer
from core.reasoning.counterfactuals.explanation.gemini_explainer import GeminiCounterfactualExplainer
from configs.inference_config import INFERENCE_CONFIG


# ---------------- Utility ----------------
def safe_serialize(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return str(obj)


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-text", required=True)
    parser.add_argument("--query-image", required=True)
    parser.add_argument("--kb-dir", default=INFERENCE_CONFIG["kb_dir"])
    parser.add_argument("--gemini-api-key", default=os.getenv("GEMINI_API_KEY"))
    parser.add_argument("--output", default="outputs/inference/final_reasoning.json")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    KB_DIR = Path(args.kb_dir)
    DEVICE = INFERENCE_CONFIG["device"]

    # ---------------- Load KB config ----------------
    kb_config_path = KB_DIR / "kb_config.json"
    if not kb_config_path.exists():
        raise RuntimeError("kb_config.json not found — cannot determine KB mode")

    with open(kb_config_path) as f:
        kb_config = json.load(f)

    kb_mode = kb_config.get("mode", "flat")
    print(f"\n✓ KB MODE DETECTED: {kb_mode.upper()}")

    # ---------------- Load metadata ----------------
    with open(KB_DIR / "metadata.json") as f:
        metadata = json.load(f)

    # ---------------- Load FAISS index ----------------
    if kb_mode == "flat":
        print("✓ Loading flat KB index")
        index = faiss.read_index(str(KB_DIR / "index.faiss"))

    elif kb_mode == "concept":
        print("✓ Loading concept KB index")
        index = faiss.read_index(str(KB_DIR / "concept_index.faiss"))

    else:
        raise ValueError(f"Unknown KB mode: {kb_mode}")

    # ---------------- Load encoders ----------------
    print("✓ Loading encoders and fusion model")

    encoder = BioMedCLIPEncoder(
        device=DEVICE,
        lora_path=INFERENCE_CONFIG.get("lora_path")
    )

    fusion = AdaptiveFusion().to(DEVICE)
    fusion.load_state_dict(
        torch.load(INFERENCE_CONFIG["fusion_path"], map_location=DEVICE)
    )
    fusion.eval()

    retriever = StabilityRetriever(index, metadata)

    # ---------------- Encode query ----------------
    print("\n✓ Encoding query")

    img = Image.open(args.query_image).convert("RGB")

    with torch.no_grad():
        img_emb = encoder.encode_image(img).unsqueeze(0).to(DEVICE)
        txt_emb = encoder.encode_text(args.query_text).unsqueeze(0).to(DEVICE)

    # ---------------- Explanation engine ----------------
    explainer = None
    if args.gemini_api_key:
        print("✓ Gemini explainer ENABLED")
        explainer = GeminiCounterfactualExplainer(args.gemini_api_key)
    else:
        print("! Gemini explainer DISABLED")

    # ---------------- Run reasoning ----------------
    print("\n✓ Running stability analysis")

    stability_runner = StabilityRunner(retriever, fusion)
    scorer = CounterfactualScorer()

    stability = stability_runner.run(img_emb, txt_emb)

    print("✓ Running diagnostic scoring")
    ranked = scorer.score(stability)

    explanation = None
    if explainer:
        print("✓ Generating Gemini explanation")
        try:
            explanation_obj = explainer.explain({
                "stability": stability,
                "ranked_hypotheses": [h.__dict__ for h in ranked]
            })
            explanation = asdict(explanation_obj)
        except Exception as e:
            print(f"❌ Gemini error: {e}")

    # ---------------- Final output ----------------
    final_output = {
        "kb_mode": kb_mode,
        "query": {
            "text": args.query_text,
            "image": args.query_image,
        },
        "stability": stability,
        "ranked_hypotheses": [h.__dict__ for h in ranked],
        "explanation": explanation,
    }

    with open(args.output, "w") as f:
        json.dump(final_output, f, indent=2, default=safe_serialize)

    print(f"\n✓ Results saved to {args.output}")

    # ---------------- Markdown report ----------------
    if explanation:
        md_path = output_path.with_suffix(".md")

        md = f"""# Medical Reasoning Report

## Primary Hypothesis
**{explanation['primary_hypothesis']}**

**Confidence**: {explanation['confidence_level'].upper()}

### Reasoning
{''.join(f'- {r}' for r in explanation['reasoning'])}

### Risk & Uncertainty
{''.join(f'- {r}' for r in explanation['uncertainty_notes'])}

### Rejected Hypotheses
{''.join(f'- {r}' for r in explanation['rejected_hypotheses'])}
"""
        with open(md_path, "w") as f:
            f.write(md)

        print(f"✓ Markdown report saved to {md_path}")


if __name__ == "__main__":
    main()
