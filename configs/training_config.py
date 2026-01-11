"""Configuration for training scripts - WITH CYCLE CONSISTENCY"""

# LoRA Training Config - WITH CYCLE CONSISTENCY
LORA_CONFIG = {
    "csv_path": "data/processed/splits/train.csv",
    "image_root": "data/images",
    "output_dir": "outputs/models/trained_lora",
    "device": "cpu",
    "batch_size": 16,
    "accum_steps": 2,
    "epochs": 10,
    "lr": 5e-5,
    "temperature": 0.07,
    
    # NEW: Cycle consistency settings
    "use_cycle_consistency": True,  # Enable cycle-consistent training
    "cycle_weight": 0.3,             # Weight for cycle loss (0.3 = 30% of total loss)
    "alignment_weight": 0.2,         # Weight for alignment loss (20% of total loss)
}

# Fusion Training Config (unchanged)
FUSION_CONFIG = {
    "csv_path": "data/processed/splits/train.csv",
    "image_root": "data/images",
    "output_path": "outputs/models/trained_fusion/fusion.pt",
    "device": "cpu",
    "batch_size": 2,
    "epochs": 5,
    "lr": 1e-4,
    "temperature": 0.07,
}