#!/usr/bin/env python3
"""
Generate paper-ready reliability tables from the current repo outputs.

Primary input:
    outputs/evaluation/constraints/query_level_results.csv

This script creates two main result sets:
1) Counterfactual reliability table
   - robust vs unstable queries
   - JS divergence
   - top-1 change rate
   - top-5 overlap

2) Constraint validation table
   - flagged vs not-flagged queries
   - mean JS / max JS
   - robustness composition
   - aggregate score statistics

It is intentionally post-processing only: it does not run model inference.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "query_id",
    "diagnosis_label",
    "variant",
    "top_label",
    "js_divergence",
    "robustness_level",
    "aggregate_score",
    "overall_violation",
    "retrieved_labels",
}

VARIANT_ORDER = ["baseline", "text_neutral", "image_neutral", "noisy"]
ROBUSTNESS_ORDER = ["high", "medium", "low"]


def _norm_label(x: object) -> str:
    return str(x).strip().lower()


def _split_labels(s: object) -> List[str]:
    if s is None:
        return []
    txt = str(s).strip()
    if not txt or txt.lower() in {"nan", "none"}:
        return []
    parts = [p.strip().lower() for p in txt.split("|") if p and p.strip()]
    # Keep order but remove duplicates
    out: List[str] = []
    seen = set()
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip().lower() in {"", "nan", "none"}:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def _topk_overlap_rate(a: Sequence[str], b: Sequence[str], k: int = 5) -> float:
    """Overlap normalized by k (so 1.0 = identical top-k lists)."""
    if k <= 0:
        return 0.0
    sa = list(a)[:k]
    sb = list(b)[:k]
    if not sa and not sb:
        return 1.0
    return len(set(sa) & set(sb)) / float(min(k, max(len(sa), len(sb), 1)))


def load_results(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )
    return df


def baseline_frame(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["variant"].map(_norm_label) == "baseline"].copy()
    if base.empty:
        raise ValueError("No baseline rows found. Expected variant == 'baseline'.")
    base["query_id"] = base["query_id"].astype(str)
    base["robustness_level"] = base["robustness_level"].map(_norm_label)
    base["overall_violation"] = base["overall_violation"].astype(int)
    base["js_divergence"] = base["js_divergence"].map(_safe_float)
    base["aggregate_score"] = base["aggregate_score"].map(_safe_float)
    base["top_label"] = base["top_label"].map(_norm_label)
    base["retrieved_labels_list"] = base["retrieved_labels"].map(_split_labels)
    base["top5_list"] = base["retrieved_labels_list"].map(lambda x: list(x)[:5])
    return base


def compute_query_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create one row per query with baseline + per-perturbation comparisons."""
    base = baseline_frame(df)
    out_rows = []

    for qid, b in base.groupby("query_id", sort=False):
        q_rows = df[df["query_id"].astype(str) == str(qid)].copy()
        if q_rows.empty:
            continue

        b_row = b.iloc[0]
        base_top1 = b_row["top_label"]
        base_top5 = b_row["top5_list"]
        base_js = _safe_float(b_row["js_divergence"])
        base_flag = int(b_row["overall_violation"])
        base_agg = _safe_float(b_row["aggregate_score"])
        base_rob = b_row["robustness_level"]

        for _, r in q_rows.iterrows():
            variant = _norm_label(r["variant"])
            top_label = _norm_label(r["top_label"])
            retrieved = _split_labels(r.get("retrieved_labels", ""))
            top5 = retrieved[:5]
            js = _safe_float(r["js_divergence"])
            agg = _safe_float(r["aggregate_score"])
            flag = int(r["overall_violation"])
            out_rows.append(
                {
                    "query_id": str(qid),
                    "diagnosis_label": _norm_label(r["diagnosis_label"]),
                    "question": r.get("question", ""),
                    "image_path": r.get("image_path", ""),
                    "variant": variant,
                    "robustness_level": _norm_label(r["robustness_level"]),
                    "overall_violation": flag,
                    "aggregate_score": agg,
                    "js_divergence": js,
                    "top_label": top_label,
                    "top1_changed": int(top_label != base_top1),
                    "top5_overlap_rate": _topk_overlap_rate(base_top5, top5, k=5),
                    "top5_jaccard": _jaccard(base_top5, top5),
                    "retrieved_labels": "|".join(retrieved),
                    "baseline_top_label": base_top1,
                    "baseline_js": base_js,
                    "baseline_aggregate_score": base_agg,
                    "baseline_overall_violation": base_flag,
                    "baseline_robustness_level": base_rob,
                }
            )

    summary = pd.DataFrame(out_rows)
    if summary.empty:
        raise ValueError("No query-level rows could be constructed from the input CSV.")
    return summary


def add_query_metrics(query_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-query metrics across perturbations for validation tables."""
    pert = query_df[query_df["variant"].isin(["text_neutral", "image_neutral", "noisy"])].copy()
    if pert.empty:
        raise ValueError("No perturbation rows found (text_neutral/image_neutral/noisy).")

    agg = (
        pert.groupby("query_id", as_index=False)
        .agg(
            diagnosis_label=("diagnosis_label", "first"),
            baseline_robustness_level=("baseline_robustness_level", "first"),
            baseline_overall_violation=("baseline_overall_violation", "first"),
            baseline_aggregate_score=("baseline_aggregate_score", "first"),
            mean_js=("js_divergence", "mean"),
            max_js=("js_divergence", "max"),
            min_js=("js_divergence", "min"),
            mean_top1_change_rate=("top1_changed", "mean"),
            any_top1_changed=("top1_changed", lambda x: int(np.any(np.asarray(x, dtype=int) > 0))),
            mean_top5_overlap=("top5_overlap_rate", "mean"),
            mean_top5_jaccard=("top5_jaccard", "mean"),
            mean_aggregate_score=("aggregate_score", "mean"),
            mean_variant_violation=("overall_violation", "mean"),
        )
    )
    agg["baseline_robustness_level"] = agg["baseline_robustness_level"].map(_norm_label)
    agg["baseline_overall_violation"] = agg["baseline_overall_violation"].astype(int)
    agg["query_id"] = agg["query_id"].astype(str)
    return agg


def counterfactual_table(query_df: pd.DataFrame) -> pd.DataFrame:
    """Paper table: unstable queries behave differently."""
    rows = []
    for robustness_level in ROBUSTNESS_ORDER:
        subset = query_df[query_df["baseline_robustness_level"] == robustness_level]
        if subset.empty:
            continue
        for variant in ["text_neutral", "image_neutral", "noisy"]:
            v = subset[subset["variant"] == variant]
            if v.empty:
                continue
            rows.append(
                {
                    "robustness_level": robustness_level,
                    "perturbation": variant,
                    "n": int(len(v)),
                    "mean_js": float(v["js_divergence"].mean()),
                    "js_std": float(v["js_divergence"].std(ddof=0)),
                    "top1_change_rate": float(v["top1_changed"].mean()),
                    "top5_overlap_rate": float(v["top5_overlap_rate"].mean()),
                    "top5_jaccard": float(v["top5_jaccard"].mean()),
                    "mean_aggregate_delta": float((v["aggregate_score"] - v["baseline_aggregate_score"]).mean()),
                    "violation_rate": float(v["overall_violation"].mean()),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["robustness_level"] = pd.Categorical(out["robustness_level"], ROBUSTNESS_ORDER, ordered=True)
        out["perturbation"] = pd.Categorical(out["perturbation"], ["text_neutral", "image_neutral", "noisy"], ordered=True)
        out = out.sort_values(["robustness_level", "perturbation"]).reset_index(drop=True)
    return out


def constraint_validation_tables(query_level: pd.DataFrame, per_query: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paper table: flagged vs not flagged; and robustness vs violation."""
    # one row per query, using the baseline decision as the final reliability outcome
    q = per_query.copy()
    q["flagged"] = q["baseline_overall_violation"].astype(int)

    # flagged vs not flagged
    flag_rows = []
    for flag_val, grp in q.groupby("flagged", sort=True):
        flag_rows.append(
            {
                "flagged": "flagged" if int(flag_val) == 1 else "not_flagged",
                "n": int(len(grp)),
                "mean_js": float(grp["mean_js"].mean()),
                "max_js": float(grp["max_js"].mean()),
                "mean_top1_change_rate": float(grp["mean_top1_change_rate"].mean()),
                "mean_top5_overlap": float(grp["mean_top5_overlap"].mean()),
                "mean_aggregate_score": float(grp["mean_aggregate_score"].mean()),
                "baseline_aggregate_score": float(grp["baseline_aggregate_score"].mean()),
                "high_robustness_rate": float((grp["baseline_robustness_level"] == "high").mean()),
                "medium_robustness_rate": float((grp["baseline_robustness_level"] == "medium").mean()),
                "low_robustness_rate": float((grp["baseline_robustness_level"] == "low").mean()),
            }
        )
    flag_df = pd.DataFrame(flag_rows)
    if not flag_df.empty:
        flag_df["flagged"] = pd.Categorical(flag_df["flagged"], ["not_flagged", "flagged"], ordered=True)
        flag_df = flag_df.sort_values("flagged").reset_index(drop=True)

    # robustness vs violation cross-tab
    cross_rows = []
    for robustness_level in ROBUSTNESS_ORDER:
        grp = q[q["baseline_robustness_level"] == robustness_level]
        if grp.empty:
            continue
        cross_rows.append(
            {
                "robustness_level": robustness_level,
                "n": int(len(grp)),
                "flag_rate": float(grp["flagged"].mean()),
                "mean_js": float(grp["mean_js"].mean()),
                "max_js": float(grp["max_js"].mean()),
                "mean_aggregate_score": float(grp["mean_aggregate_score"].mean()),
                "top1_change_any_rate": float(grp["any_top1_changed"].mean()),
                "top5_overlap": float(grp["mean_top5_overlap"].mean()),
            }
        )
    cross_df = pd.DataFrame(cross_rows)
    if not cross_df.empty:
        cross_df["robustness_level"] = pd.Categorical(cross_df["robustness_level"], ROBUSTNESS_ORDER, ordered=True)
        cross_df = cross_df.sort_values("robustness_level").reset_index(drop=True)

    return flag_df, cross_df


def add_correlation_summary(per_query: pd.DataFrame) -> dict:
    """Simple convergent-validity summary for the paper text."""
    q = per_query.copy()
    robust_map = {"high": 2.0, "medium": 1.0, "low": 0.0}
    q["robustness_score"] = q["baseline_robustness_level"].map(robust_map).fillna(0.0)
    q["overall_violation"] = q["baseline_overall_violation"].astype(int)

    # Spearman in pure pandas/numpy style to avoid scipy hard dependency here
    def spearman(x: Sequence[float], y: Sequence[float]) -> float:
        xs = pd.Series(list(x)).rank(method="average")
        ys = pd.Series(list(y)).rank(method="average")
        if xs.std(ddof=0) == 0 or ys.std(ddof=0) == 0:
            return 0.0
        return float(xs.corr(ys, method="pearson"))

    return {
        "aggregate_vs_mean_js": spearman(q["mean_aggregate_score"], q["mean_js"]),
        "aggregate_vs_overall_violation": spearman(q["mean_aggregate_score"], q["overall_violation"]),
        "aggregate_vs_robustness_level": spearman(q["mean_aggregate_score"], q["robustness_score"]),
    }


def to_markdown(df: pd.DataFrame, digits: int = 3) -> str:
    if df.empty:
        return "(empty)"
    return df.to_markdown(index=False, floatfmt=f".{digits}f")


def save_outputs(outdir: Path, query_level: pd.DataFrame, cf_table: pd.DataFrame, flag_df: pd.DataFrame, cross_df: pd.DataFrame, correlations: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    query_level.to_csv(outdir / "paper_query_level_summary.csv", index=False)
    cf_table.to_csv(outdir / "paper_counterfactual_table.csv", index=False)
    flag_df.to_csv(outdir / "paper_constraint_flag_table.csv", index=False)
    cross_df.to_csv(outdir / "paper_constraint_robustness_table.csv", index=False)

    summary = {
        "num_queries": int(query_level["query_id"].nunique()),
        "num_rows": int(len(query_level)),
        "counterfactual": {
            "rows": int(len(cf_table)),
            "by_robustness": cf_table.groupby("robustness_level").size().to_dict() if not cf_table.empty else {},
        },
        "constraints": {
            "flagged_rate": float(flag_df.loc[flag_df["flagged"] == "flagged", "n"].sum() / max(query_level["query_id"].nunique(), 1)) if not flag_df.empty else 0.0,
            "cross_rows": int(len(cross_df)),
        },
        "correlations": correlations,
    }
    (outdir / "paper_results_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Also print a concise, paper-friendly report.
    print("\n=== COUNTERFACTUAL TABLE ===")
    print(to_markdown(cf_table))
    print("\n=== CONSTRAINT FLAG TABLE ===")
    print(to_markdown(flag_df))
    print("\n=== ROBUSTNESS VS VIOLATION TABLE ===")
    print(to_markdown(cross_df))
    print("\n=== CORRELATION SUMMARY ===")
    print(json.dumps(correlations, indent=2))
    print(f"\nSaved to: {outdir.resolve()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paper-ready reliability tables from repo outputs.")
    parser.add_argument(
        "--input-csv",
        default="outputs/evaluation/constraints/query_level_results.csv",
        help="Path to query_level_results.csv produced by eval-constraints",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation/paper_ready",
        help="Directory where paper-ready tables should be written",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)

    df = load_results(input_csv)
    query_level = compute_query_level_summary(df)
    per_query = add_query_metrics(query_level)
    cf_table = counterfactual_table(query_level)
    flag_df, cross_df = constraint_validation_tables(query_level, per_query)
    correlations = add_correlation_summary(per_query)
    save_outputs(output_dir, query_level, cf_table, flag_df, cross_df, correlations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())