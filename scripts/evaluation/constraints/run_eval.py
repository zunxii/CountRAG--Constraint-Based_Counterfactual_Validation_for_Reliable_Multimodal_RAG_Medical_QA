
"""
Evaluate constraint scores across baseline / text-neutral / image-neutral
(and optional noisy) variants for the locked evaluation split.

Outputs are written in long form plus aggregated radar-ready summaries so the
results can be plotted directly as a modality-violation radar chart.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.eval_contract import load_eval_contract
from scripts.evaluation.constraints.modality_constraint_utils import (
    RADAR_AXES,
    ModalityConstraintStudy,
    flatten_records,
    prepare_radar_rows,
    save_csv,
    save_json,
    summarize_by_variant,
)


def build_output_dir(contract: Dict[str, Any], output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    return Path(contract.get("paths", {}).get("constraint_eval_dir", "outputs/evaluation/constraints"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run modality-wise constraint evaluation")
    parser.add_argument("--contract", default="configs/evaluation_contract.yaml", help="Locked evaluation contract YAML")
    parser.add_argument("--kb-dir", default=None, help="Knowledge base directory (defaults to contract kb_concept_dir)")
    parser.add_argument("--output-dir", default=None, help="Output directory for constraint artifacts")
    parser.add_argument("--device", default=None, help="cpu or cuda")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of evaluation queries to sample")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for query sampling")
    parser.add_argument("--no-noisy", action="store_true", help="Skip the noisy variant")
    args = parser.parse_args()

    contract = load_eval_contract(args.contract)
    kb_dir = args.kb_dir or contract["paths"]["kb_concept_dir"]
    output_dir = build_output_dir(contract, args.output_dir)
    device = args.device or contract.get("environment", {}).get("device", "cpu")
    num_samples = args.num_samples if args.num_samples is not None else int(contract.get("counterfactual", {}).get("num_samples", 200))
    seed = args.seed if args.seed is not None else int(contract.get("environment", {}).get("seed", 42))

    study = ModalityConstraintStudy(
        contract=contract,
        kb_dir=kb_dir,
        device=device,
        num_samples=num_samples,
        seed=seed,
        include_noisy=not args.no_noisy,
    )

    print(f"Loaded {len(study._selected_queries)} evaluation queries", flush=True)
    print(f"Using KB: {kb_dir}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print("Running modality constraint study...\n", flush=True)

    results = study.run()
    long_rows = flatten_records(results)
    summary_rows = summarize_by_variant(long_rows)
    radar_rows = prepare_radar_rows(summary_rows)

    save_json(output_dir / "query_level_results.json", results)
    save_json(
        output_dir / "summary.json",
        {
            "num_queries": len(results),
            "num_rows": len(long_rows),
            "radar_axes": RADAR_AXES,
            "summary_by_variant": summary_rows,
            "radar_ready": radar_rows,
        },
    )

    save_csv(output_dir / "query_level_results.csv", long_rows)
    save_csv(output_dir / "summary_by_variant.csv", summary_rows)
    save_csv(output_dir / "radar_ready.csv", radar_rows)

    print("\nVariant summary:", flush=True)
    for row in summary_rows:
        print(
            f"- {row['variant']:<14} n={row['num_queries']:<3} "
            f"agg={row['aggregate_score_mean']:.3f} "
            f"vio={row['overall_violation_rate']:.3f} "
            f"js={row['mean_js']:.3f}",
            flush=True,
        )

    print(f"\nSaved: {output_dir / 'query_level_results.csv'}", flush=True)
    print(f"Saved: {output_dir / 'summary_by_variant.csv'}", flush=True)
    print(f"Saved: {output_dir / 'radar_ready.csv'}", flush=True)
    print(f"Saved: {output_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())