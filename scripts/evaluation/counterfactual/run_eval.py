"""
Entry point for counterfactual evaluation - FIXED
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.evaluation.counterfactual.evaluator import CounterfactualEvaluator
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Counterfactual stability evaluation using eval queries"
    )
    parser.add_argument("--kb-dir", default="outputs/kb/kb_final_v2",
                       help="Path to knowledge base")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of queries to sample (default: all)")
    parser.add_argument("--output-dir", 
                       default="outputs/evaluation/counterfactual",
                       help="Output directory")
    parser.add_argument("--device", default="cpu",
                       help="Device for computation")
    args = parser.parse_args()
    
    evaluator = CounterfactualEvaluator(
        kb_dir=args.kb_dir,
        output_dir=args.output_dir,
        device=args.device,
        num_samples=args.num_samples
    )
    
    evaluator.run_evaluation()
    evaluator.save_results()


if __name__ == "__main__":
    main()