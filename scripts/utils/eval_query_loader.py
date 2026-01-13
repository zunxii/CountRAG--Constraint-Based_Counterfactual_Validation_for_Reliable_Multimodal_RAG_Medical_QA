"""
Unified evaluation query dataset loader
Used by ALL evaluation modules to ensure consistency
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict
import csv


class EvaluationQueryDataset:
    """
    Unified dataset loader for evaluation queries.
    Ensures all evaluation modules use the same 200 queries.
    """
    
    def __init__(self, csv_path: str = "data/processed/eval_queries.csv"):
        """
        Load evaluation queries from reserved dataset split.
        
        Args:
            csv_path: Path to evaluation queries CSV
        """
        self.csv_path = Path(csv_path)
        
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Evaluation queries not found at {csv_path}. "
                f"Run 'python scripts/utils/split_dataset.py' first!"
            )
        
        self.queries = self._load_queries()
        print(f"✓ Loaded {len(self.queries)} evaluation queries from {csv_path}")
    
    def _load_queries(self) -> List[Dict]:
        queries = []
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                question = row.get('Question', '').strip() or row.get('question', '').strip()
                query = {
                    'query_id': idx,
                    'image_path': row.get('image_path', '').strip(),
                    'diagnosis_label': row.get('category', '').strip(),
                    'question': question,
                    'combined_text': question,  # ✅ ONLY QUESTION!
                    'original_index': idx
                }
                queries.append(query)
        
        return queries
    
    @staticmethod
    def _combine_text(context: str, description: str) -> str:
        """Combine context and description"""
        context = context.strip()
        description = description.strip()
        
        if context and description:
            return f"{context} {description}"
        return context or description
    
    def __len__(self) -> int:
        return len(self.queries)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.queries[idx]
    
    def __iter__(self):
        return iter(self.queries)
    
    def get_by_diagnosis(self, diagnosis: str) -> List[Dict]:
        """Get all queries for a specific diagnosis"""
        return [q for q in self.queries if q['diagnosis_label'] == diagnosis]
    
    def sample(self, n: int, seed: int = 42) -> List[Dict]:
        """Sample n queries randomly"""
        import random
        random.seed(seed)
        return random.sample(self.queries, min(n, len(self.queries)))
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        from collections import Counter
        
        diagnoses = [q['diagnosis_label'] for q in self.queries]
        diagnosis_counts = Counter(diagnoses)
        
        return {
            'total_queries': len(self.queries),
            'num_unique_diagnoses': len(diagnosis_counts),
            'diagnosis_distribution': dict(diagnosis_counts),
            'avg_text_length': sum(len(q['combined_text']) for q in self.queries) / len(self.queries)
        }


# ========================================
# Convenience functions
# ========================================

def load_eval_queries(csv_path: str = "data/processed/eval_queries.csv") -> EvaluationQueryDataset:
    """Load evaluation queries - convenience function"""
    return EvaluationQueryDataset(csv_path)


def verify_eval_queries_exist() -> bool:
    """Check if evaluation queries exist"""
    return Path("data/processed/eval_queries.csv").exists()


def print_eval_dataset_info():
    """Print information about evaluation dataset"""
    try:
        dataset = load_eval_queries()
        stats = dataset.get_statistics()
        
        print("\n" + "="*70)
        print("EVALUATION DATASET INFORMATION")
        print("="*70)
        print(f"Total queries: {stats['total_queries']}")
        print(f"Unique diagnoses: {stats['num_unique_diagnoses']}")
        print(f"Avg text length: {stats['avg_text_length']:.1f} chars")
        print("\nTop 10 diagnoses:")
        
        sorted_diag = sorted(
            stats['diagnosis_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for diag, count in sorted_diag[:10]:
            print(f"  {diag}: {count}")
        print("="*70)
        
    except FileNotFoundError as e:
        print(f"\n⚠ {e}")
        print("\nRun this command first:")
        print("  python scripts/utils/split_dataset.py")


if __name__ == "__main__":
    print_eval_dataset_info()