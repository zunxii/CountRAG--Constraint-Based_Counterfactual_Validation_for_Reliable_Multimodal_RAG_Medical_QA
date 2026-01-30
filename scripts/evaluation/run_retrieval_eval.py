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

import torch

from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.embeddings.fusion import GatedFusion
from core.kb.image_loader import ImageLoader
from core.retrieval.retriever import KBRetriever


# =========================================================
# CONFIGURATION
# =========================================================
KB_DIR = "outputs/kb/kb_final"            # path to built KB
IMAGE_ROOT = ""               # images already have relative paths
DEVICE = "cpu"
TOP_K = 10

MODES = ["text", "image", "fusion"]
OUT_DIR = Path("experiments/retrieval")


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
def run_evaluation():
    print("Loading KB and encoders...")

    retriever = KBRetriever(KB_DIR)
    metadata = retriever.metadata

    image_encoder = BioMedCLIPEncoder(device=DEVICE)
    text_encoder = BioMedCLIPEncoder(device=DEVICE)
    fusion_model = GatedFusion(dim=image_encoder.dim).to(DEVICE)

    image_loader = ImageLoader(image_root=IMAGE_ROOT)

    for mode in MODES:
        print(f"\n=== Evaluating mode: {mode.upper()} ===")

        metric_sum = {}
        count = 0

        for i in tqdm(range(len(metadata)), desc=f"{mode} queries"):
            entry = metadata[i]
            label = entry["diagnosis_label"]

            # -------------------------------------------------
            # BUILD QUERY EMBEDDING (RE-ENCODED)
            # -------------------------------------------------
            with torch.no_grad():
                if mode == "text":
                    query_emb = text_encoder.encode_text(
                        entry["clinical_text"]["combined"]
                    )

                elif mode == "image":
                    image = image_loader.load(entry["image_path"])
                    query_emb = image_encoder.encode_image(image)

                elif mode == "fusion":
                    image = image_loader.load(entry["image_path"])
                    img_emb = image_encoder.encode_image(image)
                    txt_emb = text_encoder.encode_text(
                        entry["clinical_text"]["combined"]
                    )
                    query_emb = fusion_model(
                        img_emb.unsqueeze(0),
                        txt_emb.unsqueeze(0)
                    ).squeeze(0)

                else:
                    raise ValueError("Invalid mode")

            query_emb = query_emb.cpu().numpy().reshape(1, -1)

            # -------------------------------------------------
            # RETRIEVE (EXCLUDE SELF)
            # -------------------------------------------------
            _, indices = retriever.search(query_emb, top_k=TOP_K + 1)
            indices = [idx for idx in indices[0] if idx != i][:TOP_K]
            retrieved = [metadata[idx] for idx in indices]

            # -------------------------------------------------
            # EVALUATE
            # -------------------------------------------------
            metrics = evaluate_retrieval(retrieved, label)

            for k, v in metrics.items():
                metric_sum[k] = metric_sum.get(k, 0.0) + v

            count += 1

        # -------------------------------------------------
        # AGGREGATE
        # -------------------------------------------------
        final_metrics = {k: v / count for k, v in metric_sum.items()}
        final_metrics["_count"] = count

        print("\nRESULTS:")
        for k, v in final_metrics.items():
            if k != "_count":
                print(f"{k}: {v:.4f}")

        save_results(final_metrics, mode)


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    run_evaluation()
