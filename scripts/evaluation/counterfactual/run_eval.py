import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from configs.eval_contract import load_eval_contract
from scripts.evaluation.counterfactual.evaluator import CounterfactualEvaluator


def main():
    parser = argparse.ArgumentParser(description="Counterfactual evaluation using locked contract")
    parser.add_argument("--contract", default="configs/evaluation_contract.yaml")
    parser.add_argument("--kb-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    contract = load_eval_contract(args.contract)

    kb_dir = args.kb_dir or contract["paths"]["kb_concept_dir"]
    output_dir = args.output_dir or contract["paths"]["counterfactual_eval_dir"]
    device = args.device or contract["environment"]["device"]
    num_samples = args.num_samples if args.num_samples is not None else contract["counterfactual"]["num_samples"]

    evaluator = CounterfactualEvaluator(
        contract=contract,
        kb_dir=kb_dir,
        output_dir=output_dir,
        device=device,
        num_samples=num_samples,
    )
    evaluator.run_evaluation()
    evaluator.save_results()


if __name__ == "__main__":
    main()