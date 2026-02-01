# import numpy as np
# from tqdm import tqdm
# from collections import defaultdict

# from core.retrieval.retriever import KBRetriever
# from core.retrieval.evaluate import evaluate_retrieval

# # Load KB
# kb_dir = "kb_final"
# retriever = KBRetriever(kb_dir)

# embeddings = retriever.embeddings
# metadata = retriever.metadata

# metrics_sum = defaultdict(float)
# count = 0

# for i in tqdm(range(len(embeddings)), desc="Evaluating retrieval"):
#     query_emb = embeddings[i:i+1]
#     query_label = metadata[i]["diagnosis_label"]

#     _, indices = retriever.search(query_emb, top_k=11)

#     # remove self
#     indices = [idx for idx in indices if idx != i][:10]

#     retrieved = [metadata[idx] for idx in indices]

#     metrics = evaluate_retrieval(retrieved, query_label)

#     for k, v in metrics.items():
#         metrics_sum[k] += v

#     count += 1

# # Average metrics
# final_metrics = {k: v / count for k, v in metrics_sum.items()}

# print("\n=== RETRIEVAL RESULTS ===")
# for k, v in final_metrics.items():
#     print(f"{k}: {v:.4f}")


import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import sys

# Fix imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch

from core.embeddings.biomedclip import BioMedCLIPEncoder
# from core.embeddings.fusion import GatedFusion # Wrong class
from core.fusion.adaptive_fusion import AdaptiveFusion
from core.kb.image_loader import ImageLoader
from core.retrieval.retriever import KBRetriever
from configs.inference_config import INFERENCE_CONFIG


# =========================================================
# CONFIGURATION
# =========================================================
KB_DIR = INFERENCE_CONFIG["kb_dir"]
IMAGE_ROOT = ""
DEVICE = INFERENCE_CONFIG["device"]
TOP_K = 10

MODES = ["text", "image", "fusion"]
OUT_DIR = Path("experiments/retrieval")


# ... existing metric functions ... (skipped in replace content, keeping file structure)
# Wait, I can't skip content in replace_file_content unless I target specific lines.
# I will target the imports and then the loading section separately.



# =========================================================
# METRIC FUNCTIONS (NO CHEATING)
# =========================================================
def evaluate_retrieval(retrieved, target_label):
    hits = [1 if r["diagnosis_label"] == target_label else 0 for r in retrieved]

    metrics = {}

    for k in [1, 5, 10]:
        topk = hits[:k]
        metrics[f"R@{k}"] = 1.0 if sum(topk) > 0 else 0.0
        metrics[f"P@{k}"] = sum(topk) / k

    # MRR
    rr = 0.0
    for i, h in enumerate(hits):
        if h == 1:
            rr = 1.0 / (i + 1)
            break
    metrics["MRR"] = rr

    return metrics


def save_results(metrics, mode):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        "top_k": TOP_K,
        "num_queries": metrics["_count"],
        "metrics": {k: v for k, v in metrics.items() if k != "_count"}
    }

    out_path = OUT_DIR / f"{mode}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved results → {out_path}")


# =========================================================
# MAIN EVALUATION
# =========================================================
# ... (previous imports)

def run_evaluation():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA for ablation")
    args = parser.parse_args()

    print("Loading KB and encoders...")
    
    # Use config but override LoRA if requested
    lora_path = INFERENCE_CONFIG.get("lora_path")
    if args.no_lora:
        print("⚠ LoRA DISABLED (Ablation Mode)")
        lora_path = None
        
    retriever = KBRetriever(KB_DIR)
    metadata = retriever.metadata

    image_encoder = BioMedCLIPEncoder(device=DEVICE, lora_path=lora_path)
    text_encoder = BioMedCLIPEncoder(device=DEVICE, lora_path=lora_path)
    
    fusion_model = AdaptiveFusion().to(DEVICE)
    if "fusion_path" in INFERENCE_CONFIG:
        # Load fusion weights regardless of LoRA (unless we trained fusion specifically for no-lora, which we didn't, 
        # but typically fusion matches the encoder state. For strict ablation, maybe random fusion? 
        # But usually ablation is "what if base encoder + trained fusion". 
        # Let's keep fusion weights as they are the architecture choice.)
        fusion_path = INFERENCE_CONFIG["fusion_path"]
        print(f"Loading fusion weights from {fusion_path}")
        fusion_model.load_state_dict(
            torch.load(fusion_path, map_location=DEVICE)
        )
    fusion_model.eval()

    image_loader = ImageLoader(image_root=IMAGE_ROOT)

    for mode in MODES:
        print(f"\n=== Evaluating mode: {mode.upper()} ===")

        metric_sum = {}
        count = 0
        
        # Track per-disease performance
        per_label_correct = {}
        per_label_total = {}

        for i in tqdm(range(len(metadata)), desc=f"{mode} queries"):
            entry = metadata[i]
            label = entry["diagnosis_label"]
            
            # Init counters
            if label not in per_label_total:
                per_label_correct[label] = 0
                per_label_total[label] = 0

            # ... (test item generation) ...
            # Get all images for this concept to test individual image retrieval
            # For text mode, we just test the canonical text once
            test_items = []
            if mode == "text":
                test_items.append({"type": "text", "content": entry.get("canonical_text", "")})
            else:
                 # For image/fusion, test every image in the concept
                img_paths = entry.get("image_paths", [])
                img_paths = img_paths[:50] 
                for p in img_paths:
                    test_items.append({"type": "image", "content": p})

            for item in test_items:
                with torch.no_grad():
                    if mode == "text":
                        query_emb = text_encoder.encode_text(item["content"])
                    elif mode == "image":
                        image = image_loader.load(item["content"])
                        query_emb = image_encoder.encode_image(image)
                    elif mode == "fusion":
                        image = image_loader.load(item["content"])
                        img_emb = image_encoder.encode_image(image)
                        txt_emb = text_encoder.encode_text(entry.get("canonical_text", ""))
                        query_emb = fusion_model(
                            img_emb.unsqueeze(0),
                            txt_emb.unsqueeze(0)
                        ).squeeze(0)

                query_emb = query_emb.cpu().numpy().reshape(1, -1)
                
                _, indices = retriever.search(query_emb, top_k=TOP_K)
                retrieved = [metadata[idx] for idx in indices[0]]
                
                metrics = evaluate_retrieval(retrieved, label)
                
                # Check top-1 hit for per-label accuracy
                if metrics["R@1"] == 1.0:
                    per_label_correct[label] += 1
                per_label_total[label] += 1

                for k, v in metrics.items():
                    metric_sum[k] = metric_sum.get(k, 0.0) + v

                count += 1

        # Aggregate Global Metrics
        final_metrics = {k: v / count for k, v in metric_sum.items()}
        final_metrics["_count"] = count
        
        # Aggregate Per-Label Accuracy
        per_label_acc = {
            lbl: per_label_correct[lbl]/per_label_total[lbl] 
            for lbl in per_label_total if per_label_total[lbl] > 0
        }
        final_metrics["per_label_accuracy"] = per_label_acc

        print("\nRESULTS:")
        for k, v in final_metrics.items():
            if k not in ["_count", "per_label_accuracy"]:
                print(f"{k}: {v:.4f}")

        # Save with special naming for ablation
        suffix = "_no_lora" if args.no_lora else ""
        save_results(final_metrics, f"{mode}{suffix}")

if __name__ == "__main__":
    run_evaluation()
