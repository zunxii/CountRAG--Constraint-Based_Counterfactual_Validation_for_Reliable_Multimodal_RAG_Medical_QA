"""
LoRA Training - FIXED: Uses ONLY question + image (NO LEAKAGE)
"""
import os
import csv
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType
import open_clip
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================
CSV_PATH = "data/processed/splits/train.csv"
IMAGE_ROOT = "data/images"
OUTPUT_DIR = "outputs/models/trained_lora"
PLOTS_DIR = "outputs/plots/lora"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 15
LR = 5e-5
PATIENCE = 3
VAL_SPLIT = 0.1

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print("✓ FIXED: Using ONLY question + image (no description/context)")

# ============================================================================
# DATASET - FIXED TO USE ONLY QUESTION
# ============================================================================
class TrainingDataset(Dataset):
    def __init__(self, csv_path, image_root, preprocess):
        self.samples = []
        self.preprocess = preprocess
        self.image_root = Path(image_root)
        
        print(f"Loading training data from {csv_path}...")
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row.get("image_path", "").strip()
                question = row.get("Question", "").strip()  # ONLY question!
                category = row.get("category", "").strip()
                
                if not img_name or not question or not category:
                    continue
                
                if len(question) < 10:
                    continue
                
                img_path = self.image_root / img_name
                if img_path.exists():
                    self.samples.append((img_path, question, category))
        
        print(f"✓ Loaded {len(self.samples)} samples (question + image only)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, text, category = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.preprocess(image)
            return image, text, category
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            dummy_img = torch.zeros(3, 224, 224)
            return dummy_img, text, category

# ============================================================================
# LOSS
# ============================================================================
def contrastive_loss(img_emb, txt_emb, temperature=0.07):
    img_emb = F.normalize(img_emb, dim=-1)
    txt_emb = F.normalize(txt_emb, dim=-1)
    
    logits = img_emb @ txt_emb.T / temperature
    labels = torch.arange(len(img_emb), device=img_emb.device)
    
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    
    return (loss_i2t + loss_t2i) / 2

# ============================================================================
# TRAINING
# ============================================================================
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, texts, _ in pbar:
        try:
            images = images.to(device)
            tokenizer = open_clip.get_tokenizer(MODEL_NAME)
            tokens = tokenizer(list(texts)).to(device)
            
            img_emb = model.encode_image(images)
            txt_emb = model.encode_text(tokens)
            
            loss = contrastive_loss(img_emb, txt_emb)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        except Exception as e:
            print(f"\nBatch error: {e}")
            continue
    
    return total_loss / num_batches if num_batches > 0 else 0.0

def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, texts, _ in tqdm(loader, desc="Validating"):
            try:
                images = images.to(device)
                tokenizer = open_clip.get_tokenizer(MODEL_NAME)
                tokens = tokenizer(list(texts)).to(device)
                
                img_emb = model.encode_image(images)
                txt_emb = model.encode_text(tokens)
                
                loss = contrastive_loss(img_emb, txt_emb)
                total_loss += loss.item()
                num_batches += 1
                
            except:
                continue
    
    return total_loss / num_batches if num_batches > 0 else 0.0

def plot_training_curves(history, save_path):
    epochs = [h['epoch'] for h in history]
    train_losses = [h['train_loss'] for h in history]
    val_losses = [h['val_loss'] for h in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('LoRA Training (Question+Image Only)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# MAIN
# ============================================================================
def train():
    print("="*70)
    print("LoRA TRAINING - FIXED VERSION")
    print("Uses ONLY: question + image (NO context/description)")
    print("="*70)
    
    # Load model
    print("\n[1/4] Loading BioMedCLIP...")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
    model = model.to(DEVICE)
    
    # Apply LoRA
    print("\n[2/4] Applying LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print("\n[3/4] Loading dataset...")
    full_dataset = TrainingDataset(CSV_PATH, IMAGE_ROOT, preprocess)
    
    val_size = int(VAL_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    print(f"✓ Train: {train_size}, Validation: {val_size}")
    
    # Train
    print("\n[4/4] Starting training...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = []
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*70}")
        
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        val_loss = validate(model, val_loader, DEVICE)
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train: {train_loss:.4f}")
        print(f"  Val:   {val_loss:.4f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"  ✓ Best! Saving...")
            model.save_pretrained(OUTPUT_DIR)
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"\n⚠ Early stopping")
            break
    
    # Save history
    with open(Path(OUTPUT_DIR) / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot
    plot_training_curves(history, Path(PLOTS_DIR) / "lora_training_curves.png")
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE")
    print(f"✓ Best val loss: {best_val_loss:.4f}")
    print(f"✓ Model: {OUTPUT_DIR}")
    print("="*70)

if __name__ == "__main__":
    train()