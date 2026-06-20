#!/usr/bin/env python3
"""
generate_sheet_json.py
======================
Samples 50 random queries from data/processed/splits/eval.csv,
runs BOTH retrieval systems against the built KB, and emits a JSON
file structured exactly as the clinician evaluation Excel sheet expects.

System A  = Baseline    — top-3 unique diagnoses by cosine similarity (text-only query)
System B  = CountRAG    — top-3 unique diagnoses after multimodal fusion + counterfactual stability

Run from the repo root:
    python generate_sheet_json.py

Optional flags:
    --n            number of cases (default 50)
    --seed         random seed   (default 42)
    --kb-dir       KB directory  (default outputs/kb/kb_final_v2)
    --eval-csv     eval CSV path  (default data/processed/splits/eval.csv)
    --output       output JSON   (default sheet_eval_data.json)
    --device       cpu or cuda   (default cpu)
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import faiss

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.fusion.adaptive_fusion import AdaptiveFusion
from core.kb.image_loader import ImageLoader
from core.retrieval.retriever import KBRetriever
from configs.inference_config import INFERENCE_CONFIG
from core.reasoning.counterfactuals.stability.retrieval import StabilityRetriever
from core.reasoning.counterfactuals.stability.runner import StabilityRunner


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_eval_csv(csv_path: str) -> list:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): (v.strip() if v is not None else "") for k, v in row.items()})
    return rows


def sample_rows(rows: list, n: int, seed: int) -> list:
    rng = random.Random(seed)
    return rng.sample(rows, min(n, len(rows)))


def get_col(row: dict, *candidates, default="") -> str:
    for c in candidates:
        if c in row and str(row[c]).strip():
            return str(row[c]).strip()
    return default


# ─────────────────────────────────────────────────────────────────────────────
# Model / KB loading
# ─────────────────────────────────────────────────────────────────────────────

def load_models(kb_dir: str, device: str):
    print("[1/3] Loading encoder (LoRA)...")
    lora_path = INFERENCE_CONFIG.get("lora_path", "outputs/models/trained_lora")
    encoder = BioMedCLIPEncoder(device=device, lora_path=lora_path)

    print("[2/3] Loading fusion model...")
    fusion_pt = INFERENCE_CONFIG.get("fusion_path", "outputs/models/trained_fusion/fusion.pt")
    fusion = AdaptiveFusion().to(device)
    fusion.load_state_dict(torch.load(fusion_pt, map_location=device))
    fusion.eval()

    print("[3/3] Loading KB retriever...")
    retriever = KBRetriever(kb_dir)
    image_loader = ImageLoader()

    return encoder, fusion, retriever, image_loader


def _normalize_query_embedding(emb: torch.Tensor) -> np.ndarray:
    q_np = emb.detach().cpu().numpy().reshape(1, -1).astype("float32")
    faiss.normalize_L2(q_np)
    return q_np


def _collect_unique_diagnoses(
    q_np: np.ndarray,
    retriever,
    top_k: int = 3,
    exclude_indices=None,
    initial_fetch_k: int = 25,
) -> list:
    """
    Retrieve top_k DISTINCT diagnosis labels.
    Expands the search pool until enough unique diagnoses are found or KB is exhausted.
    """
    if exclude_indices is None:
        exclude_indices = []

    max_k = len(retriever.metadata)
    fetch_k = min(max(initial_fetch_k, top_k), max_k)

    results = []
    seen_diags = set()
    seen_doc_indices = set()

    while len(results) < top_k and fetch_k <= max_k:
        try:
            scores, indices = retriever.search(
                q_np,
                top_k=fetch_k,
                exclude_indices=exclude_indices
            )
        except TypeError:
            # In case the retriever.search signature doesn't accept exclude_indices
            scores, indices = retriever.search(q_np, top_k=fetch_k)

        for score, idx in zip(scores[0], indices[0]):
            idx = int(idx)

            if idx < 0 or idx >= len(retriever.metadata):
                continue
            if idx in seen_doc_indices:
                continue
            seen_doc_indices.add(idx)

            diag = retriever.metadata[idx].get("diagnosis_label", "Unknown")
            if not diag or diag in seen_diags:
                continue

            seen_diags.add(diag)
            results.append({
                "rank": len(results) + 1,
                "diagnosis": diag,
                "score": round(float(score), 4),
            })

            if len(results) >= top_k:
                break

        if len(results) < top_k:
            if fetch_k >= max_k:
                break
            fetch_k = min(max_k, fetch_k * 2)

    while len(results) < top_k:
        results.append({
            "rank": len(results) + 1,
            "diagnosis": "",
            "score": 0.0
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# System A: Baseline (text-only cosine similarity)
# ─────────────────────────────────────────────────────────────────────────────

def baseline_retrieve(query_text: str, encoder, retriever,
                      top_k: int = 3, exclude_image_path: str = None,
                      device: str = "cpu") -> list:
    """Text-only cosine similarity retrieval with unique diagnoses."""
    with torch.no_grad():
        txt_emb = encoder.encode_text(query_text)

    q_np = _normalize_query_embedding(txt_emb)

    exclude_indices = []
    if exclude_image_path and getattr(retriever, "kb_mode", None) == "flat":
        for idx, meta in enumerate(retriever.metadata):
            if str(meta.get("image_path", "")) == str(exclude_image_path):
                exclude_indices.append(idx)

    return _collect_unique_diagnoses(
        q_np=q_np,
        retriever=retriever,
        top_k=top_k,
        exclude_indices=exclude_indices,
        initial_fetch_k=25,
    )


# ─────────────────────────────────────────────────────────────────────────────
# System B: CountRAG (multimodal + counterfactual stability)
# ─────────────────────────────────────────────────────────────────────────────

def countrag_retrieve(query_text: str, image_path: str,
                      encoder, fusion, retriever, image_loader,
                      top_k: int = 3, device: str = "cpu"):
    """
    Multimodal fusion retrieval with counterfactual stability scoring.
    Returns (top_k_unique_results, reliability_dict).
    """
    img_path = Path(image_path) if image_path else None
    if img_path and not img_path.exists():
        img_path = Path("data/images") / img_path.name

    try:
        if img_path and img_path.exists():
            img = image_loader.load(str(img_path))
            with torch.no_grad():
                img_emb = encoder.encode_image(img).unsqueeze(0).to(device)
        else:
            raise FileNotFoundError("image path missing")
    except Exception as e:
        print(f"    [warn] image load failed ({e}), using zero embedding")
        img_emb = torch.zeros(1, 512, device=device)

    with torch.no_grad():
        txt_emb = encoder.encode_text(query_text).unsqueeze(0).to(device)

    with torch.no_grad():
        fused_emb = fusion(img_emb, txt_emb)

    # Counterfactual stability
    try:
        stab_retriever = StabilityRetriever(retriever.index, retriever.metadata)
        runner = StabilityRunner(stab_retriever, fusion, device=device, top_k=10)
        stab_out = runner.run(img_emb, txt_emb)

        js_no_text = stab_out["stability"]["js_divergence"].get("no_text", 0.5)
        js_no_image = stab_out["stability"]["js_divergence"].get("no_image", 0.5)
        robustness = stab_out["stability"].get("robustness_level", "medium")

        msi_t = round(max(0.0, min(1.0, 1.0 - js_no_text)), 2)
        msi_i = round(max(0.0, min(1.0, 1.0 - js_no_image)), 2)

        rob_w = {"high": 1.0, "medium": 0.75, "low": 0.5}.get(robustness, 0.6)
        r_score = round(min(1.0, 0.5 * (msi_t + msi_i) + 0.3 * rob_w), 2)

    except Exception as e:
        print(f"    [warn] stability failed ({e}), using defaults")
        msi_t = 0.0
        msi_i = 0.0
        r_score = 0.0

    verdict = "PASS" if r_score >= 0.5 else "REVIEW"

    reliability = {
        "msi_text": msi_t,
        "msi_image": msi_i,
        "reliability_score": r_score,
        "verdict": verdict,
        "raw": f"MSI-T:{msi_t} MSI-I:{msi_i} R:{r_score} [{verdict}]",
    }

    q_np = _normalize_query_embedding(fused_emb)
    results = _collect_unique_diagnoses(
        q_np=q_np,
        retriever=retriever,
        top_k=top_k,
        exclude_indices=None,
        initial_fetch_k=25,
    )

    return results, reliability


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def hit_at_k(results: list, gt: str, k: int) -> int:
    return int(any(r["diagnosis"] == gt for r in results[:k]))


def mrr_score(results: list, gt: str) -> float:
    for r in results:
        if r["diagnosis"] == gt:
            return round(1.0 / r["rank"], 4)
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kb-dir", type=str, default="outputs/kb/kb_final_v2")
    parser.add_argument("--eval-csv", type=str, default="data/processed/splits/eval.csv")
    parser.add_argument("--output", type=str, default="sheet_eval_data.json")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print(f"\nLoading: {args.eval_csv}")
    all_rows = load_eval_csv(args.eval_csv)
    if not all_rows:
        raise RuntimeError(f"No rows found in {args.eval_csv}")

    print(f"  Total rows: {len(all_rows)}")
    print(f"  Columns:    {list(all_rows[0].keys())}")

    sampled = sample_rows(all_rows, args.n, args.seed)
    print(f"  Sampled:    {len(sampled)} rows (seed={args.seed})\n")

    encoder, fusion, retriever, image_loader = load_models(args.kb_dir, args.device)
    print(f"  KB mode: {retriever.kb_mode}  |  entries: {len(retriever.metadata)}\n")

    cases = []

    for i, row in enumerate(sampled, 1):
        question = get_col(row, "Question", "question", "query", "text")
        image_path = get_col(row, "image_path", "image", "img_path")
        gt_label = get_col(row, "category", "label", "diagnosis", "ground_truth")
        context = get_col(row, "context", "Context")
        description = get_col(row, "description", "Description", "Question_summ", "question_summ")

        image_desc = description if description else context

        print(f"  [{i:02d}/{len(sampled)}] GT={gt_label:28s}  img={Path(image_path).name if image_path else 'N/A'}")

        # ── System A ─────────────────────────────────────────────────────────
        try:
            a_results = baseline_retrieve(
                query_text=question,
                encoder=encoder,
                retriever=retriever,
                top_k=3,
                exclude_image_path=image_path,
                device=args.device,
            )
            a_sim = round(float(a_results[0]["score"]) if a_results else 0.0, 3)
        except Exception as e:
            print(f"    [err] Baseline: {e}")
            a_results = [{"rank": k, "diagnosis": "", "score": 0.0} for k in range(1, 4)]
            a_sim = 0.0

        a_reliability_raw = f"Similarity: {a_sim}"

        # ── System B ─────────────────────────────────────────────────────────
        try:
            b_results, b_reliability = countrag_retrieve(
                query_text=question,
                image_path=image_path,
                encoder=encoder,
                fusion=fusion,
                retriever=retriever,
                image_loader=image_loader,
                top_k=3,
                device=args.device,
            )
        except Exception as e:
            print(f"    [err] CountRAG: {e}")
            b_results = [{"rank": k, "diagnosis": "", "score": 0.0} for k in range(1, 4)]
            b_reliability = {
                "msi_text": 0.0,
                "msi_image": 0.0,
                "reliability_score": 0.0,
                "verdict": "REVIEW",
                "raw": "MSI-T:0.00 MSI-I:0.00 R:0.00 [REVIEW]",
            }

        # ── Metrics ───────────────────────────────────────────────────────────
        a_metrics = {
            "hit@1": hit_at_k(a_results, gt_label, 1),
            "hit@3": hit_at_k(a_results, gt_label, 3),
            "mrr": mrr_score(a_results, gt_label),
        }
        b_metrics = {
            "hit@1": hit_at_k(b_results, gt_label, 1),
            "hit@3": hit_at_k(b_results, gt_label, 3),
            "mrr": mrr_score(b_results, gt_label),
        }

        cases.append({
            "case_id": i,
            "ground_truth": gt_label,
            "patient_query": question,
            "image_description": image_desc,
            "image_path": image_path,

            "sys_a_rank1": a_results[0]["diagnosis"],
            "sys_a_rank2": a_results[1]["diagnosis"],
            "sys_a_rank3": a_results[2]["diagnosis"],
            "sys_a_reliability": a_reliability_raw,

            "sys_b_rank1": b_results[0]["diagnosis"],
            "sys_b_rank2": b_results[1]["diagnosis"],
            "sys_b_rank3": b_results[2]["diagnosis"],
            "sys_b_reliability": b_reliability["raw"],

            "baseline": {
                "system": "System A (Baseline — text cosine similarity)",
                "top_k_results": a_results,
                "reliability": {"raw": a_reliability_raw, "similarity_score": a_sim},
                "metrics": a_metrics,
            },
            "countrag": {
                "system": "System B (CountRAG — multimodal + counterfactual)",
                "top_k_results": b_results,
                "reliability": b_reliability,
                "metrics": b_metrics,
            },
        })

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    def agg(sys_key):
        h1 = [c[sys_key]["metrics"]["hit@1"] for c in cases]
        h3 = [c[sys_key]["metrics"]["hit@3"] for c in cases]
        mr = [c[sys_key]["metrics"]["mrr"] for c in cases]
        n = len(cases)
        return {
            "recall@1": round(sum(h1) / n, 4),
            "recall@3": round(sum(h3) / n, 4),
            "mrr": round(sum(mr) / n, 4),
            "n_cases": n,
            "n_rank1_correct": sum(h1),
        }

    agg_a = agg("baseline")
    agg_b = agg("countrag")

    verdicts = [c["countrag"]["reliability"]["verdict"] for c in cases]
    r_vals = [c["countrag"]["reliability"]["reliability_score"] for c in cases]
    mt_vals = [c["countrag"]["reliability"]["msi_text"] for c in cases]
    mi_vals = [c["countrag"]["reliability"]["msi_image"] for c in cases]

    rel_summary = {
        "pass_count": verdicts.count("PASS"),
        "review_count": verdicts.count("REVIEW"),
        "pass_rate": round(verdicts.count("PASS") / len(cases), 4) if cases else 0.0,
        "avg_R": round(sum(r_vals) / len(r_vals), 4) if r_vals else 0.0,
        "avg_MSI_T": round(sum(mt_vals) / len(mt_vals), 4) if mt_vals else 0.0,
        "avg_MSI_I": round(sum(mi_vals) / len(mi_vals), 4) if mi_vals else 0.0,
    }

    bucket = defaultdict(lambda: {"a": 0, "b": 0, "n": 0})
    for c in cases:
        gt = c["ground_truth"]
        bucket[gt]["n"] += 1
        bucket[gt]["a"] += c["baseline"]["metrics"]["hit@1"]
        bucket[gt]["b"] += c["countrag"]["metrics"]["hit@1"]

    per_diag = {
        diag: {
            "n_cases": s["n"],
            "baseline_recall@1": round(s["a"] / s["n"], 4),
            "countrag_recall@1": round(s["b"] / s["n"], 4),
            "delta_recall@1": round((s["b"] - s["a"]) / s["n"], 4),
        }
        for diag, s in sorted(bucket.items())
        if s["n"] > 0
    }

    output = {
        "study": "CountRAG-Clinic — Real KB Evaluation",
        "eval_csv": args.eval_csv,
        "kb_dir": args.kb_dir,
        "n_cases": len(cases),
        "seed": args.seed,
        "systems": {
            "A": "Baseline — text-query cosine similarity over KB embeddings",
            "B": "CountRAG — LoRA-adapted BioMedCLIP + adaptive fusion + counterfactual stability",
        },
        "aggregate_metrics": {
            "baseline": agg_a,
            "countrag": agg_b,
            "delta": {
                "recall@1": round(agg_b["recall@1"] - agg_a["recall@1"], 4),
                "recall@3": round(agg_b["recall@3"] - agg_a["recall@3"], 4),
                "mrr": round(agg_b["mrr"] - agg_a["mrr"], 4),
            },
        },
        "countrag_reliability_summary": rel_summary,
        "per_diagnosis_breakdown": per_diag,
        "cases": cases,
    }

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✓ JSON written to: {out_path}")
    print("\n── Aggregate Metrics ────────────────────────────────────────")
    print(f"  Baseline  R@1={agg_a['recall@1']}  R@3={agg_a['recall@3']}  MRR={agg_a['mrr']}")
    print(f"  CountRAG  R@1={agg_b['recall@1']}  R@3={agg_b['recall@3']}  MRR={agg_b['mrr']}")
    d = output["aggregate_metrics"]["delta"]
    print(f"  Δ         R@1={d['recall@1']:+.4f}  R@3={d['recall@3']:+.4f}  MRR={d['mrr']:+.4f}")
    print(f"\n── CountRAG Reliability ─────────────────────────────────────")
    print(
        f"  PASS={rel_summary['pass_count']}  REVIEW={rel_summary['review_count']}"
        f"  Pass rate={rel_summary['pass_rate']}  Avg R={rel_summary['avg_R']}"
    )


if __name__ == "__main__":
    main()