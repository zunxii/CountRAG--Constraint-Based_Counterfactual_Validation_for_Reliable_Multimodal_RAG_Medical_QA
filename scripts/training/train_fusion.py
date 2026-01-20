"""
Fusion Training - FIXED: Uses ONLY question + image (NO LEAKAGE)
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
CSV_PATH = "data/processed/splits/train.csv"
IMAGE_ROOT = "data/images"
OUTPUT_PATH = "outputs/models/trained_fusion/fusion.pt"
PLOTS_DIR = "outputs/plots/fusion"

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
TEMPERATURE = 0.07
VAL_SPLIT = 0.1
PATIENCE = 3

os.makedirs(Path(OUTPUT_PATH).parent, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

if DEVICE == "mps":
    print(f"Device: {DEVICE} (Apple Silicon GPU)")
else:
    print(f"Device: {DEVICE}")
print("✓ FIXED: Using ONLY question + image (no context/description)")

# ============================================================================
# DATASET - FIXED TO USE ONLY QUESTION
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
                question = row.get("Question", "").strip()  # ONLY question!
                
                if not img_name or len(question) < 10:
                    continue
                
                img_path = self.image_root / img_name
                if img_path.exists():
                    self.samples.append((img_path, question))
        
        print(f"✓ Loaded {len(self.samples)} samples (question + image only)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def fusion_collate_fn(batch):
    image_paths, texts = zip(*batch)
    return list(image_paths), list(texts)

# ============================================================================
# LOSS
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
# TRAINING
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

def plot_training_curves(history, save_path):
    epochs = [h['epoch'] for h in history]
    train_losses = [h['train_loss'] for h in history]
    val_losses = [h['val_loss'] for h in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Fusion Training (Question+Image Only)', fontsize=14, fontweight='bold')
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
    print("FUSION TRAINING - FIXED VERSION")
    print("Uses ONLY: question + image (NO context/description)")
    print("="*70)
    
    # Load encoder
    print("\n[1/4] Loading frozen encoder...")
    encoder = BioMedCLIPEncoder(device=DEVICE)
    encoder.model.eval()
    for p in encoder.model.parameters():
        p.requires_grad = False
    
    # Init fusion
    print("\n[2/4] Initializing fusion...")
    fusion = AdaptiveFusion().to(DEVICE)
    print(f"✓ Params: {sum(p.numel() for p in fusion.parameters()):,}")
    
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
    
    print(f"✓ Train: {train_size}, Val: {val_size}")
    
    # Train
    print("\n[4/4] Starting training...")
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
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
            torch.save(fusion.state_dict(), OUTPUT_PATH)
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"\n⚠ Early stopping")
            break
    
    # Save history
    with open(Path(OUTPUT_PATH).parent / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot
    plot_training_curves(history, Path(PLOTS_DIR) / "fusion_training_curves.png")
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE")
    print(f"✓ Best val: {best_val_loss:.4f}")
    print(f"✓ Model: {OUTPUT_PATH}")
    print("="*70)

if __name__ == "__main__":
    train()