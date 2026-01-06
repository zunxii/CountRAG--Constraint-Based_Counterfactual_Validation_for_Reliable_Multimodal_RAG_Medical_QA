"""Configuration for training scripts - FIXED VERSION"""

# LoRA Training Config - FIXED
LORA_CONFIG = {
    "csv_path": "data/processed/train.csv",
    "image_root": "data/images",
    "output_dir": "outputs/models/trained_lora",
    "device": "cpu",
    "batch_size": 16,  # FIXED from 1
    "accum_steps": 2,  # FIXED from 8
    "epochs": 10,  # FIXED from 1
    "lr": 5e-5,
    "temperature": 0.07,
}

# Fusion Training Config
FUSION_CONFIG = {
    "csv_path": "data/processed/train.csv",
    "image_root": "data/images",
    "output_path": "outputs/models/trained_fusion/fusion.pt",
    "device": "cpu",
    "batch_size": 2,
    "epochs": 5,  # Increased from 3
    "lr": 1e-4,
    "temperature": 0.07,
}