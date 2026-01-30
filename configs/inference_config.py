"""Configuration for inference scripts."""

INFERENCE_CONFIG = {
    "kb_dir": "outputs/kb/kb_final_concept",
    "device": "cpu",
    "top_k": 10,
    "lora_path": "outputs/models/trained_lora",
    "fusion_path": "outputs/models/trained_fusion/fusion.pt",
}