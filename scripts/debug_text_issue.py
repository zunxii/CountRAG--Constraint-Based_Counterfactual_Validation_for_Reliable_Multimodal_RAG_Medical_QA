import json
import torch
import sys
import csv
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path.cwd()))

from core.embeddings.biomedclip import BioMedCLIPEncoder

def debug_text_mismatch():
    print("DEBUG: Investigating Text Mismatch")
    
    # 1. Load one eval query
    eval_csv = Path("data/processed/splits/eval.csv")
    with open(eval_csv, 'r') as f:
        reader = csv.DictReader(f)
        query_row = next(reader) # Get first row
    
    query_text = query_row.get('Question', '').strip() or query_row.get('question', '').strip()
    diagnosis = query_row.get('category', '').strip()
    
    print(f"\n[QUERY] Diagnosis: {diagnosis}")
    print(f"[QUERY] Text: '{query_text}'")
    
    # 2. Load KB Metadata
    kb_dir = Path("outputs/kb/kb_final_concept")
    with open(kb_dir / "metadata.json", 'r') as f:
        kb_meta = json.load(f)
        
    # Find matching concept
    matching_concepts = [c for c in kb_meta if c['diagnosis_label'] == diagnosis]
    
    if not matching_concepts:
        print(f"ERROR: No concept found for diagnosis '{diagnosis}'")
        return

    concept = matching_concepts[0]
    print(f"\n[KB CONCEPT] ID: {concept['concept_id']}")
    print(f"[KB CONCEPT] Canonical Text: '{concept.get('canonical_text')}'")
    
    # 3. Compute Embeddings & similarity
    print("\nLoading Encoder...")
    device = "cpu" # Debug on CPU
    encoder = BioMedCLIPEncoder(device=device)
    
    print("Encoding...")
    with torch.no_grad():
        q_emb = encoder.encode_text(query_text)
        c_emb = encoder.encode_text(concept['canonical_text'])
        
    sim = torch.dot(q_emb, c_emb).item()
    print(f"\n[SIMILARITY] Dot Product: {sim:.4f}")
    
    if sim < 0.2:
        print("⚠ VERY LOW SIMILARITY! The text is semantically different.")

if __name__ == "__main__":
    debug_text_mismatch()
