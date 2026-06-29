import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from configs.eval_contract import load_eval_contract
from scripts.evaluation.retrieval.evaluator import RetrievalEvaluator


def main():
    parser = argparse.ArgumentParser(description="Retrieval evaluation using locked contract")
    parser.add_argument("--contract", default="configs/evaluation_contract.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    contract = load_eval_contract(args.contract)
    output_dir = args.output_dir or contract["paths"]["retrieval_eval_dir"]
    device = args.device or contract["environment"]["device"]

    evaluator = RetrievalEvaluator(
        contract=contract,
        output_dir=output_dir,
        device=device,
    )
    evaluator.run_all_evaluations()
    evaluator.save_results()


if __name__ == "__main__":
    main()