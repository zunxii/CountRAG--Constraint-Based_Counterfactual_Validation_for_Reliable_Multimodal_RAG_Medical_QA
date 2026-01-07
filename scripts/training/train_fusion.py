"""
Fusion Training with Loss Tracking and Validation
LOCAL VERSION
"""

import os
import csv
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.fusion.adaptive_fusion import AdaptiveFusion

# ============================================================================
# CONFIGURATION
# ============================================================================
CSV_PATH = "data/processed/train.csv"
IMAGE_ROOT = "data/images"
OUTPUT_PATH = "outputs/models/trained_fusion/fusion.pt"
PLOTS_DIR = "outputs/plots/fusion"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
TEMPERATURE = 0.07
VAL_SPLIT = 0.1
PATIENCE = 3

os.makedirs(Path(OUTPUT_PATH).parent, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print(f"Device: {DEVICE}")

# ============================================================================
# DATASET
# ============================================================================
class FusionDataset(Dataset):
    def __init__(self, csv_path, image_root):
        self.samples = []
        self.image_root = Path(image_root)
        
        print(f"Loading training data from {csv_path}...")
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row.get("image_path", "").strip()
                context = row.get("context", "")
                description = row.get("description", "")
                text = f"{context} {description}".strip()
                
                if not img_name or len(text) < 10:
                    continue
                
                img_path = self.image_root / img_name
                if img_path.exists():
                    self.samples.append((img_path, text))
        
        print(f"✓ Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def fusion_collate_fn(batch):
    image_paths, texts = zip(*batch)
    return list(image_paths), list(texts)

# ============================================================================
# LOSS FUNCTION
# ============================================================================
def contrastive_loss(a, b, temperature):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    
    logits = a @ b.T / temperature
    labels = torch.arange(len(a), device=a.device)
    
    loss_ab = F.cross_entropy(logits, labels)
    loss_ba = F.cross_entropy(logits.T, labels)
    
    return (loss_ab + loss_ba) / 2

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_epoch(fusion, encoder, loader, optimizer, device):
    fusion.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(loader, desc="Training")
    for image_paths, texts in pbar:
        try:
            images = [Image.open(p).convert("RGB") for p in image_paths]
            
            with torch.no_grad():
                img_emb = encoder.encode_image_batch(images)
                txt_emb = encoder.encode_text_batch(texts)
            
            fused = fusion(img_emb, txt_emb)
            loss = contrastive_loss(fused, txt_emb, TEMPERATURE)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        except Exception as e:
            print(f"\nBatch error: {e}")
            continue
    
    return total_loss / num_batches if num_batches > 0 else 0.0

def validate(fusion, encoder, loader, device):
    fusion.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for image_paths, texts in tqdm(loader, desc="Validating"):
            try:
                images = [Image.open(p).convert("RGB") for p in image_paths]
                
                img_emb = encoder.encode_image_batch(images)
                txt_emb = encoder.encode_text_batch(texts)
                
                fused = fusion(img_emb, txt_emb)
                loss = contrastive_loss(fused, txt_emb, TEMPERATURE)
                
                total_loss += loss.item()
                num_batches += 1
                
            except:
                continue
    
    return total_loss / num_batches if num_batches > 0 else 0.0

# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_training_curves(history, save_path):
    epochs = [h['epoch'] for h in history]
    train_losses = [h['train_loss'] for h in history]
    val_losses = [h['val_loss'] for h in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Fusion Training: Loss Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Loss curves saved to {save_path}")

# ============================================================================
# MAIN TRAINING
# ============================================================================
def train():
    print("="*70)
    print("FUSION TRAINING WITH VALIDATION")
    print("="*70)
    
    # Load frozen encoder
    print("\n[1/4] Loading frozen encoder...")
    encoder = BioMedCLIPEncoder(device=DEVICE)
    encoder.model.eval()
    for p in encoder.model.parameters():
        p.requires_grad = False
    print("✓ Encoder loaded and frozen")
    
    # Initialize fusion
    print("\n[2/4] Initializing fusion module...")
    fusion = AdaptiveFusion().to(DEVICE)
    print(f"✓ Fusion parameters: {sum(p.numel() for p in fusion.parameters()):,}")
    
    # Load dataset
    print("\n[3/4] Loading dataset...")
    full_dataset = FusionDataset(CSV_PATH, IMAGE_ROOT)
    
    val_size = int(VAL_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=fusion_collate_fn, num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=fusion_collate_fn, num_workers=2
    )
    
    print(f"✓ Train: {train_size}, Validation: {val_size}")
    
    # Setup optimizer
    print("\n[4/4] Starting training...")
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = []
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*70}")
        
        train_loss = train_epoch(fusion, encoder, train_loader, optimizer, DEVICE)
        val_loss = validate(fusion, encoder, val_loader, DEVICE)
        scheduler.step()
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"  ✓ New best model! Saving...")
            torch.save(fusion.state_dict(), OUTPUT_PATH)
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⚠ Early stopping at epoch {epoch+1}")
            break
    
    # Save history
    history_path = Path(OUTPUT_PATH).parent / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ Training history saved to {history_path}")
    
    # Plot curves
    plot_path = Path(PLOTS_DIR) / "fusion_training_curves.png"
    plot_training_curves(history, plot_path)
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE")
    print(f"✓ Best validation loss: {best_val_loss:.4f}")
    print(f"✓ Model saved to: {OUTPUT_PATH}")
    print(f"✓ Plots saved to: {PLOTS_DIR}")
    print("="*70)

if __name__ == "__main__":
    train()