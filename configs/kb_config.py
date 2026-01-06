# configs/kb_config.py
"""Configuration for KB building - UPDATED to use training split"""

KB_BUILD_CONFIG = {
    "csv_path": "data/processed/train.csv",  # CHANGED: Use training split only
    "image_root": "data/images",
    "output_dir": "outputs/kb/kb_final_v2",
    "device": "cpu",
    "lora_path": "outputs/models/trained_lora",
    "fusion_path": "outputs/models/trained_fusion/fusion.pt",
}

KB_SMOKE_CONFIG = {
    "csv_path": "data/raw/test.csv",
    "image_root": "data/images",
    "output_dir": "outputs/kb/kb_smoke",
    "device": "cpu",
}

# ========================================
# configs/evaluation_config.py (NEW FILE)
"""Configuration for all evaluation tasks"""

EVALUATION_CONFIG = {
    # Shared evaluation query set
    "eval_queries_csv": "data/processed/eval_queries.csv",
    "image_root": "data/images",
    "device": "cpu",
    
    # KB path for evaluations
    "kb_dir": "outputs/kb/kb_final_v2",
    
    # Model paths
    "lora_path": "outputs/models/trained_lora",
    "fusion_path": "outputs/models/trained_fusion/fusion.pt",
    
    # Retrieval evaluation settings
    "retrieval": {
        "top_k": 20,
        "modes": ["text", "image", "fusion"]
    },
    
    # Counterfactual evaluation settings
    "counterfactual": {
        "num_samples": 200,  # Use all eval queries
        "perturbation_scales": [0.01, 0.05, 0.1]
    },
    
    # Encoder evaluation settings
    "encoders": {
        "batch_size": 8
    },
    
    # LoRA evaluation settings
    "lora": {
        "comparison_samples": 50
    }
}