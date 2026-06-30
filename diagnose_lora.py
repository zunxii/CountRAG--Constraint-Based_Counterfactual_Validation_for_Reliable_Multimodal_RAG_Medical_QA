"""
diagnose_lora.py
=================
Run this in your actual training/eval environment (where torch + peft + open_clip work).
It isolates exactly where the LoRA signal is being lost.

Usage:
    cd test/   # repo root
    python diagnose_lora.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from core.embeddings.biomedclip import BioMedCLIPEncoder

LORA_PATH = "outputs/models/trained_lora"

print("=" * 70)
print("STEP 1: Load both encoders")
print("=" * 70)
base_encoder = BioMedCLIPEncoder(device="cpu")
print("✓ Base encoder loaded")
lora_encoder = BioMedCLIPEncoder(device="cpu", lora_path=LORA_PATH)
print("✓ LoRA encoder loaded")

print()
print("=" * 70)
print("STEP 2: Check if PeftModel wraps the model object differently")
print("=" * 70)
print(f"type(base_encoder.model)  = {type(base_encoder.model)}")
print(f"type(lora_encoder.model)  = {type(lora_encoder.model)}")
print(f"Are they the same object? {base_encoder.model is lora_encoder.model}")

print()
print("=" * 70)
print("STEP 3: Confirm LoRA layers actually exist in the wrapped model")
print("=" * 70)
lora_modules = [n for n, m in lora_encoder.model.named_modules() if "lora" in n.lower()]
print(f"Number of modules with 'lora' in name: {len(lora_modules)}")
print("Sample:", lora_modules[:5])

print()
print("=" * 70)
print("STEP 4: Encode identical text with both encoders, compare")
print("=" * 70)
test_text = "swelling around the ankle with redness"
with torch.no_grad():
    base_txt = base_encoder.encode_text(test_text)
    lora_txt = lora_encoder.encode_text(test_text)

l2_dist = torch.norm(base_txt - lora_txt).item()
cos_sim = F.cosine_similarity(base_txt.unsqueeze(0), lora_txt.unsqueeze(0)).item()
print(f"L2 distance (text):        {l2_dist:.6f}")
print(f"Cosine similarity (text):  {cos_sim:.6f}")
if l2_dist < 1e-5:
    print(">>> LoRA encoder produces IDENTICAL text embeddings to base. <<<")
    print(">>> This confirms LoRA is not being applied at inference.   <<<")

print()
print("=" * 70)
print("STEP 5: Manually force-apply LoRA delta and compare to encoder output")
print("=" * 70)
# Check if PEFT's adapter is even 'active'
if hasattr(lora_encoder.model, "active_adapters"):
    print("active_adapters:", lora_encoder.model.active_adapters)
if hasattr(lora_encoder.model, "peft_config"):
    print("peft_config keys:", list(lora_encoder.model.peft_config.keys()))

# Check disable_adapter context manager behavior — if output is IDENTICAL
# with adapter disabled, that further confirms adapter is inactive in forward.
if hasattr(lora_encoder.model, "disable_adapter"):
    with torch.no_grad():
        with lora_encoder.model.disable_adapter():
            disabled_txt = lora_encoder.encode_text(test_text)
    dist_disabled_vs_enabled = torch.norm(disabled_txt - lora_txt).item()
    print(f"L2 distance (adapter ON vs disable_adapter() context): {dist_disabled_vs_enabled:.6f}")
    if dist_disabled_vs_enabled < 1e-5:
        print(">>> Confirmed: enabling/disabling the adapter makes NO difference.")
        print(">>> The adapter is loaded but never participates in forward().")
    else:
        print(">>> Adapter DOES affect forward() when toggled directly.")
        print(">>> => bug is likely in BioMedCLIPEncoder wiring, not PEFT itself.")

print()
print("=" * 70)
print("DONE — paste this output back to debug further")
print("=" * 70)

print()
print("=" * 70)
print("STEP 6: Trace through StabilityRunner.run() exactly as evaluator does")
print("=" * 70)
from configs.eval_contract import load_eval_contract
from core.fusion.adaptive_fusion import AdaptiveFusion
from core.retrieval.retriever import KBRetriever
from core.reasoning.counterfactuals.stability.retrieval import StabilityRetriever
from core.reasoning.counterfactuals.stability.runner import StabilityRunner

contract = load_eval_contract("configs/evaluation_contract.yaml")
kb_dir = contract["paths"]["kb_concept_dir"]

kb_retriever = KBRetriever(kb_dir)
stability_retriever = StabilityRetriever(kb_retriever.index, kb_retriever.metadata)

fusion = AdaptiveFusion()
fusion.load_state_dict(torch.load(contract["paths"]["models"]["fusion_model_file"], map_location="cpu"))
fusion.eval()

runner_base = StabilityRunner(stability_retriever, fusion, contract=contract, device="cpu", top_k=10)
runner_lora = StabilityRunner(stability_retriever, fusion, contract=contract, device="cpu", top_k=10)

with torch.no_grad():
    img_emb_base = base_encoder.encode_image(__import__("PIL.Image", fromlist=["Image"]).new("RGB", (224,224))).unsqueeze(0)
    txt_emb_base = base_encoder.encode_text(test_text).unsqueeze(0)

    img_emb_lora = lora_encoder.encode_image(__import__("PIL.Image", fromlist=["Image"]).new("RGB", (224,224))).unsqueeze(0)
    txt_emb_lora = lora_encoder.encode_text(test_text).unsqueeze(0)

print(f"img_emb diff (base vs lora): {torch.norm(img_emb_base - img_emb_lora).item():.6f}")
print(f"txt_emb diff (base vs lora): {torch.norm(txt_emb_base - txt_emb_lora).item():.6f}")

out_base = runner_base.run(img_emb_base, txt_emb_base)
out_lora = runner_lora.run(img_emb_lora, txt_emb_lora)

print()
print("constraints (base) keys:", list(out_base.get("constraints", {}).keys()))
print("constraints (lora) keys:", list(out_lora.get("constraints", {}).keys()))

c_base = out_base.get("constraints", {})
c_lora = out_lora.get("constraints", {})

if "error" in c_base or "error" in c_lora:
    print()
    print("!!! CONSTRAINT EXTRACTION ERRORED !!!")
    print("base error:", c_base.get("error"))
    print("base traceback:", c_base.get("traceback"))
    print("lora error:", c_lora.get("error"))
    print("lora traceback:", c_lora.get("traceback"))
else:
    print()
    print("scores (base):", c_base.get("scores"))
    print("scores (lora):", c_lora.get("scores"))
    mc_base = c_base.get("modality_consistency", {})
    mc_lora = c_lora.get("modality_consistency", {})
    print()
    print("modality_consistency (base):", mc_base)
    print("modality_consistency (lora):", mc_lora)