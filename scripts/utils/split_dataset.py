"""
Dataset splitter - Reserve last 200 queries for evaluation
Usage: python scripts/utils/split_dataset.py
"""
import pandas as pd
from pathlib import Path
import json

def split_dataset(
    input_csv: str = "data/raw/clipsyntel.csv",
    output_dir: str = "data/processed",
    eval_size: int = 200
):
    """
    Split dataset into training (KB building) and evaluation sets.
    
    Args:
        input_csv: Path to full dataset
        output_dir: Output directory for splits
        eval_size: Number of samples for evaluation (taken from end)
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    total_samples = len(df)
    
    print(f"Total samples: {total_samples}")
    print(f"Evaluation samples: {eval_size}")
    print(f"Training samples: {total_samples - eval_size}")
    
    # Split dataset
    # Training: all except last eval_size
    # Evaluation: last eval_size samples
    train_df = df.iloc[:-eval_size]
    eval_df = df.iloc[-eval_size:]
    
    # Save splits
    train_path = output_dir / "train.csv"
    eval_path = output_dir / "eval_queries.csv"
    
    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)
    
    print(f"\n✓ Training set saved: {train_path} ({len(train_df)} samples)")
    print(f"✓ Evaluation set saved: {eval_path} ({len(eval_df)} samples)")
    
    # Save split metadata
    metadata = {
        "total_samples": total_samples,
        "train_samples": len(train_df),
        "eval_samples": len(eval_df),
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "split_method": "last_n_for_eval"
    }
    
    metadata_path = output_dir / "split_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved: {metadata_path}")
    
    # Verify no overlap
    train_ids = set(train_df.index)
    eval_ids = set(eval_df.index)
    overlap = train_ids & eval_ids
    
    assert len(overlap) == 0, "Found overlap between train and eval sets!"
    print("\n✓ Verified: No overlap between train and eval sets")
    
    return train_path, eval_path, metadata_path


if __name__ == "__main__":
    split_dataset()