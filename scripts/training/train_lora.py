#!/usr/bin/env python3
"""
Robust LoRA training with cycle-consistency for patient-question <-> clinical-description alignment.
Fixes: consistent keys, no hidden exceptions, clear loss definitions, safe tokenization & device handling.
"""

import os
import json
import sys
import random
import csv
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# PEFT / CLIP
from peft import LoraConfig, get_peft_model, TaskType
import open_clip
from PIL import Image


sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.embeddings.cycle_consistency import CycleConsistencyLoss, AlignmentLoss
# --------------------
# CONFIG
# --------------------
CSV_PATH = "data/processed/splits/train.csv"
IMAGE_ROOT = "data/images"
OUTPUT_DIR = "outputs/models/trained_lora"
PLOTS_DIR = "outputs/plots/lora"

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 15
LR = 5e-5
PATIENCE = 3
VAL_SPLIT = 0.1

# Cycle consistency controls
USE_CYCLE_CONSISTENCY = True
CYCLE_WEIGHT = 0.2      # start small
ALIGNMENT_WEIGHT = 0.05 # keep alignment weak by default
TEMPERATURE = 0.07

SEED = 42
NUM_WORKERS = 2
DROP_LAST = True       # keep for stable batch-size behavior

# Safety flag: if False, we DON'T use clinical text in training (only question+image)
USE_CLINICAL_AS_KB = True

# --------------------
# UTIL / SEED
# --------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if DEVICE == "mps":
    print(f"Device: {DEVICE} (Apple Silicon GPU)")
else:
    print(f"Device: {DEVICE}")
print(f"Model: {MODEL_NAME}")
print(f"Cycle consistency: {'ON' if USE_CYCLE_CONSISTENCY and USE_CLINICAL_AS_KB else 'OFF'}")
print(f"Using clinical text as KB: {'YES' if USE_CLINICAL_AS_KB else 'NO'}")

# --------------------
# DATASET
# --------------------
class CycleTrainingDataset(Dataset):
    """
    Expects CSV with columns (case-insensitive):
      - Question or question
      - Question_summ or question_summ (optional; used as clinical if USE_CLINICAL_AS_KB=True)
      - image_path (relative to IMAGE_ROOT)
      - category (kept only for diagnostics; NOT used as supervision)
      - context, description (optional)
    """
    def __init__(self, csv_path: str, image_root: str, preprocess):
        self.image_root = Path(image_root)
        self.preprocess = preprocess
        self.samples: List[Dict] = []

        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # normalize keys (lowercase)
                row_l = {k.strip().lower(): (v or "").strip() for k, v in row.items()}

                img_name = row_l.get("image_path", "")
                # Accept multiple possible question field names
                question = row_l.get("question", "") or row_l.get("question_text", "") or row_l.get("question_text".lower(), "")
                # clinical / kb text (optional)
                clinical = (
                    row_l.get("question_summ", "") or
                    (row_l.get("context", "") + " " + row_l.get("description", "")).strip()
                )

                category = row_l.get("category", "")

                if not img_name or not question:
                    continue

                if USE_CLINICAL_AS_KB and not clinical:
                    # if configured to use clinical but it's missing, skip sample
                    continue

                img_path = self.image_root / img_name
                if not img_path.exists():
                    # skip missing image but log occasionally
                    continue

                # length filters to remove very short strings
                if len(question) < 6:
                    continue
                if USE_CLINICAL_AS_KB and len(clinical) < 6:
                    continue

                self.samples.append({
                    "image_path": str(img_path),
                    "question": question,
                    "clinical": clinical,
                    "category": category
                })

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found in {csv_path}")

        print(f"Loaded {len(self.samples)} samples from {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # load image (guarded)
        try:
            img = Image.open(s["image_path"]).convert("RGB")
            img = self.preprocess(img)
        except Exception as e:
            # If image read fails, return a zero-tensor (but still keep text)
            print(f"Warning: failed to open {s['image_path']}: {e}")
            img = torch.zeros(3, 224, 224)

        return {
            "image": img,
            "question": s["question"],
            "clinical": s["clinical"],
            "category": s["category"]
        }

def cycle_collate_fn(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)
    questions = [item["question"] for item in batch]
    clinicals = [item["clinical"] for item in batch]
    categories = [item["category"] for item in batch]
    return {
        "images": images,
        "questions": questions,
        "clinicals": clinicals,
        "categories": categories
    }

# --------------------
# CLIP contrastive (standard)
# --------------------
def contrastive_loss(img_emb, txt_emb, temperature=TEMPERATURE):
    img = F.normalize(img_emb, dim=-1)
    txt = F.normalize(txt_emb, dim=-1)
    logits = (img @ txt.T) / temperature
    labels = torch.arange(img.size(0), device=img.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_i2t + loss_t2i)

# --------------------
# TRAIN / VAL EPOCH
# --------------------
def train_epoch(model, loader, optimizer, device, tokenizer,
                cycle_criterion, alignment_criterion):
    model.train()
    total = 0.0
    total_contrastive = 0.0
    total_cycle = 0.0
    total_align = 0.0
    count = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        images = batch["images"].to(device)
        questions = batch["questions"]
        clinicals = batch["clinicals"]

        # Tokenize + move to device
        q_tokens = tokenizer(questions)
        c_tokens = tokenizer(clinicals)
        # tokenizer usually returns torch tensor already
        if isinstance(q_tokens, torch.Tensor):
            q_tokens = q_tokens.to(device)
            c_tokens = c_tokens.to(device)
        else:
            # if tokenizer returns a dict (some tokenizers), handle sensibly:
            q_tokens = {k: v.to(device) for k, v in q_tokens.items()}
            c_tokens = {k: v.to(device) for k, v in c_tokens.items()}

        # Encode
        img_emb = model.encode_image(images)      # (B, D)
        q_txt_emb = model.encode_text(q_tokens)  # (B, D)
        c_txt_emb = model.encode_text(c_tokens)  # (B, D)

        # Losses
        loss_q_i = contrastive_loss(img_emb, q_txt_emb)
        loss_c_i = contrastive_loss(img_emb, c_txt_emb) if USE_CLINICAL_AS_KB else torch.tensor(0.0, device=device)
        contrastive = 0.5 * (loss_q_i + loss_c_i) if USE_CLINICAL_AS_KB else loss_q_i

        cycle_loss = cycle_criterion(q_txt_emb, c_txt_emb, img_emb) if (USE_CYCLE_CONSISTENCY and USE_CLINICAL_AS_KB) else torch.tensor(0.0, device=device)
        align_loss = alignment_criterion(q_txt_emb, c_txt_emb) if (USE_CLINICAL_AS_KB) else torch.tensor(0.0, device=device)

        loss = contrastive + CYCLE_WEIGHT * cycle_loss + ALIGNMENT_WEIGHT * align_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += loss.item()
        total_contrastive += float(contrastive.detach().cpu().item())
        total_cycle += float(cycle_loss.detach().cpu().item())
        total_align += float(align_loss.detach().cpu().item())
        count += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "contr": f"{(contrastive.item() if isinstance(contrastive, torch.Tensor) else contrastive):.4f}",
            "cycle": f"{total_cycle/count:.6f}"
        })

    if count == 0:
        raise RuntimeError("No training batches processed.")
    return total/count, total_contrastive/count, total_cycle/count, total_align/count

def validate_epoch(model, loader, device, tokenizer, cycle_criterion, alignment_criterion):
    model.eval()
    total = 0.0
    total_contrastive = 0.0
    total_cycle = 0.0
    total_align = 0.0
    count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            images = batch["images"].to(device)
            questions = batch["questions"]
            clinicals = batch["clinicals"]

            q_tokens = tokenizer(questions)
            c_tokens = tokenizer(clinicals)
            if isinstance(q_tokens, torch.Tensor):
                q_tokens = q_tokens.to(device)
                c_tokens = c_tokens.to(device)
            else:
                q_tokens = {k: v.to(device) for k, v in q_tokens.items()}
                c_tokens = {k: v.to(device) for k, v in c_tokens.items()}

            img_emb = model.encode_image(images)
            q_txt_emb = model.encode_text(q_tokens)
            c_txt_emb = model.encode_text(c_tokens)

            loss_q_i = contrastive_loss(img_emb, q_txt_emb)
            loss_c_i = contrastive_loss(img_emb, c_txt_emb) if USE_CLINICAL_AS_KB else torch.tensor(0.0, device=device)
            contrastive = 0.5 * (loss_q_i + loss_c_i) if USE_CLINICAL_AS_KB else loss_q_i

            cycle_loss = cycle_criterion(q_txt_emb, c_txt_emb, img_emb) if (USE_CYCLE_CONSISTENCY and USE_CLINICAL_AS_KB) else torch.tensor(0.0, device=device)
            align_loss = alignment_criterion(q_txt_emb, c_txt_emb) if USE_CLINICAL_AS_KB else torch.tensor(0.0, device=device)
            loss = contrastive + CYCLE_WEIGHT * cycle_loss + ALIGNMENT_WEIGHT * align_loss

            total += loss.item()
            total_contrastive += float(contrastive.detach().cpu().item())
            total_cycle += float(cycle_loss.detach().cpu().item())
            total_align += float(align_loss.detach().cpu().item())
            count += 1

    if count == 0:
        raise RuntimeError("No validation batches processed.")
    return total/count, total_contrastive/count, total_cycle/count, total_align/count

# --------------------
# TRAINING ENTRYPOINT
# --------------------
def plot_history(history: List[dict], save_path: str):
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    train_cycle = [h["train_cycle"] for h in history]
    val_cycle = [h["val_cycle"] for h in history]

    plt.figure(figsize=(10,4))
    plt.plot(epochs, train_losses, label="train_total")
    plt.plot(epochs, val_losses, label="val_total")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_total.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10,4))
    plt.plot(epochs, train_cycle, label="train_cycle")
    plt.plot(epochs, val_cycle, label="val_cycle")
    plt.xlabel("epoch"); plt.ylabel("cycle_loss"); plt.legend()
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_cycle.png"), dpi=200)
    plt.close()

def train():
    # Load model & tokenizer
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model = model.to(DEVICE)

    # Apply LoRA (PEFT)
    lora_conf = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                           target_modules="all-linear", bias="none",
                           task_type=TaskType.FEATURE_EXTRACTION)
    model = get_peft_model(model, lora_conf)
    model.print_trainable_parameters()

    # Dataset / loaders
    full_ds = CycleTrainingDataset(CSV_PATH, IMAGE_ROOT, preprocess)
    val_size = int(VAL_SPLIT * len(full_ds))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=cycle_collate_fn, num_workers=NUM_WORKERS,
                              pin_memory=(DEVICE=="cuda"), drop_last=DROP_LAST)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=cycle_collate_fn, num_workers=NUM_WORKERS,
                            pin_memory=(DEVICE=="cuda"))

    print(f"Train samples: {train_size}, Val samples: {val_size}")

    # Loss modules
    cycle_criterion = CycleConsistencyLoss().to(DEVICE)
    alignment_criterion = AlignmentLoss(temperature=TEMPERATURE).to(DEVICE)

    # Optimizer/scheduler
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val = float('inf')
    patience = 0
    history = []

    for epoch in range(1, EPOCHS+1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        train_loss, train_contr, train_cycle, train_align = train_epoch(
            model, train_loader, optimizer, DEVICE, tokenizer, cycle_criterion, alignment_criterion
        )

        val_loss, val_contr, val_cycle, val_align = validate_epoch(
            model, val_loader, DEVICE, tokenizer, cycle_criterion, alignment_criterion
        )

        scheduler.step()

        print(f"Epoch {epoch} - train_total: {train_loss:.4f}, val_total: {val_loss:.4f}")
        print(f"  train_contrastive: {train_contr:.4f}, train_cycle: {train_cycle:.4f}, train_align: {train_align:.4f}")
        print(f"  val_contrastive: {val_contr:.4f}, val_cycle: {val_cycle:.4f}, val_align: {val_align:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "val_loss": val_loss,
            "train_contrastive": train_contr, "val_contrastive": val_contr,
            "train_cycle": train_cycle, "val_cycle": val_cycle,
            "train_alignment": train_align, "val_alignment": val_align
        })

        # Save if improved
        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            model.save_pretrained(OUTPUT_DIR)
            print(f"Saved best model to {OUTPUT_DIR}")
        else:
            patience += 1
            print(f"Patience: {patience}/{PATIENCE}")
            if patience >= PATIENCE:
                print("Early stopping triggered.")
                break

    # Save history & plots
    with open(Path(OUTPUT_DIR)/"training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_history(history, str(Path(PLOTS_DIR)/"lora_cycle_training.png"))
    print("Training done.")

if __name__ == "__main__":
    train()
