"""
KB Builder - UPDATED with concept-based architecture support
Replaces: core/kb/builder.py
"""
import csv
import json
import hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm

from core.kb.schema import KBEntry
from core.kb.text_processor import TextProcessor
from core.kb.image_loader import ImageLoader
from core.kb.index import KBIndex
from core.kb.storage import KBStorage
from core.kb.report import BuildReport
from core.fusion.adaptive_fusion import AdaptiveFusion


class KBBuilder:
    """
    KB Builder with TWO modes:
    - mode='flat': Original one-entry-per-image
    - mode='concept': Groups images by canonical description (NEW)
    """

    def __init__(
        self,
        image_encoder,
        text_encoder,
        fusion_model,
        output_dir: str,
        image_root: str,
        device: str = "cpu",
        mode: str = "concept",  # 'flat' or 'concept'
        aggregation_method: str = "mean",  # for concept mode
    ):
        self.device = device
        self.mode = mode
        self.aggregation_method = aggregation_method

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.fusion_model = fusion_model.to(device)
        self.fusion_model.eval()

        self.text_processor = TextProcessor()
        self.image_loader = ImageLoader()
        self.storage = KBStorage()
        self.report = BuildReport()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.image_root = Path(image_root)
        if not self.image_root.exists():
            raise FileNotFoundError(f"Image root not found: {self.image_root}")

        # Storage
        self.entries: list[KBEntry] = []
        self.embeddings: list[np.ndarray] = []
        
        # Concept mode storage
        self.concepts: dict = {}  # concept_id -> concept data

    def build(self, csv_path: str):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        print("\n" + "="*70)
        print(f"BUILDING KB - MODE: {self.mode.upper()}")
        if self.mode == "concept":
            print("Grouping images by canonical descriptions")
        print("="*70 + "\n")

        if self.mode == "flat":
            self._build_flat(csv_path)
        else:
            self._build_concept(csv_path)

    def _build_flat(self, csv_path: Path):
        """Original flat KB building (one entry per image)"""
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for idx, row in enumerate(
                tqdm(reader, desc="Building KB", unit="rows")
            ):
                self.report.log_row_seen()

                try:
                    entry = self._process_row_flat(row, idx)
                    self.entries.append(entry)

                    self.report.log_success(
                        category=entry.diagnosis_label,
                        anatomy_region=entry.anatomy["normalized_region"],
                    )

                except Exception as e:
                    self.report.log_failure(idx, str(e))
                    continue

        self._finalize_flat()

    def _build_concept(self, csv_path: Path):
        """NEW concept-based KB building (groups by description)"""
        
        # PHASE 1: Group by description
        print("[1/4] Grouping by canonical description...")
        description_groups = self._group_by_description(csv_path)
        
        print(f"✓ Found {len(description_groups)} unique concepts")
        print(f"✓ Total images: {sum(len(g) for g in description_groups.values())}")
        
        # PHASE 2: Create concepts and encode images
        print("\n[2/4] Creating concepts and encoding images...")
        self._create_concepts(description_groups)
        
        print(f"✓ Created {len(self.concepts)} concepts")
        
        # PHASE 3: Aggregate and fuse
        print("\n[3/4] Aggregating images and creating concept embeddings...")
        self._aggregate_concepts()
        
        # PHASE 4: Finalize
        print("\n[4/4] Building indices...")
        self._finalize_concept()

    def _group_by_description(self, csv_path: Path) -> dict:
        """Group rows by canonical description"""
        groups = defaultdict(list)
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.report.log_row_seen()
                
                context = row.get("context", "").strip()
                description = row.get("description", "").strip()
                canonical_text = self.text_processor.combine_text(context, description)
                
                if len(canonical_text) < 10:
                    continue
                
                # Hash for grouping
                desc_hash = hashlib.md5(canonical_text.encode()).hexdigest()[:12]
                
                groups[desc_hash].append({
                    'image_path': row.get("image_path", "").strip(),
                    'canonical_text': canonical_text,
                    'diagnosis': row.get("category", "").strip(),
                    'context': context,
                    'description': description,
                })
        
        return groups

    def _create_concepts(self, description_groups: dict):
        """Create concepts from grouped descriptions"""
        
        for desc_hash, rows in tqdm(description_groups.items(), 
                                     desc="Processing concepts"):
            if not rows:
                continue
            
            first_row = rows[0]
            diagnosis = first_row['diagnosis']
            canonical_text = first_row['canonical_text']
            
            concept_id = f"{diagnosis}_{desc_hash}"
            
            # Initialize concept
            self.concepts[concept_id] = {
                'concept_id': concept_id,
                'canonical_text': canonical_text,
                'diagnosis_label': diagnosis,
                'image_paths': [],
                'image_embeddings': [],
                'context': first_row['context'],
                'description': first_row['description'],
            }
            
            # Encode all images for this concept
            for row in rows:
                img_path = self.image_root / row['image_path']
                
                if not img_path.exists():
                    continue
                
                try:
                    image = self.image_loader.load(img_path)
                    
                    with torch.no_grad():
                        img_emb = self.image_encoder.encode_image(image)
                        img_emb_np = img_emb.cpu().numpy()
                    
                    self.concepts[concept_id]['image_paths'].append(str(img_path))
                    self.concepts[concept_id]['image_embeddings'].append(img_emb_np)
                    
                except Exception as e:
                    self.report.log_failure(-1, f"Image error: {e}")
                    continue
            
            # Only keep concepts with images
            if len(self.concepts[concept_id]['image_paths']) == 0:
                del self.concepts[concept_id]
            else:
                self.report.log_success(
                    category=diagnosis,
                    anatomy_region="concept"
                )

    def _aggregate_concepts(self):
        """Encode text and aggregate image embeddings"""
        
        for concept_id, concept in tqdm(self.concepts.items(),
                                        desc="Aggregating"):
            
            # Encode canonical text ONCE
            with torch.no_grad():
                txt_emb = self.text_encoder.encode_text(concept['canonical_text'])
                concept['text_embedding'] = txt_emb.cpu().numpy()
            
            # Aggregate multiple image embeddings
            img_embeddings = np.array(concept['image_embeddings'])
            
            if self.aggregation_method == "mean":
                agg_img = img_embeddings.mean(axis=0)
            elif self.aggregation_method == "max":
                agg_img = img_embeddings.max(axis=0)
            elif self.aggregation_method == "weighted":
                norms = np.linalg.norm(img_embeddings, axis=1, keepdims=True)
                weights = norms / norms.sum()
                agg_img = (img_embeddings * weights).sum(axis=0)
            else:
                agg_img = img_embeddings.mean(axis=0)
            
            # Normalize
            norm = np.linalg.norm(agg_img)
            if norm > 0:
                agg_img = agg_img / norm
            
            concept['aggregated_image_embedding'] = agg_img
            concept['num_images'] = len(concept['image_paths'])
            concept['image_variance'] = float(img_embeddings.var())
            
            # Create concept embedding via fusion
            with torch.no_grad():
                img_tensor = torch.from_numpy(agg_img).unsqueeze(0).to(self.device)
                txt_tensor = torch.from_numpy(concept['text_embedding']).unsqueeze(0).to(self.device)
                
                concept_emb = self.fusion_model(img_tensor, txt_tensor)
                concept['concept_embedding'] = concept_emb.squeeze(0).cpu().numpy()

    def _process_row_flat(self, row: dict, idx: int) -> KBEntry:
        """Process single row for flat mode (original logic)"""
        # Resolve image
        image_name = row.get("image_path", "").strip()
        if not image_name:
            raise ValueError("Missing image_path")

        image_path = self.image_root / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Label
        category = row.get("category", "").strip()
        if not category:
            raise ValueError("Missing category")

        # Text: Use context+description for KB reference
        context = row.get("context", "")
        description = row.get("description", "")
        combined_text = self.text_processor.combine_text(context, description)

        if len(combined_text) < 10:
            raise ValueError("Clinical text too short")

        anatomy_info = self.text_processor.extract_anatomy(combined_text)
        normalized_region = self.text_processor.normalize_region(
            anatomy_info["semantic_types"]
        )

        anatomy = {
            "raw_mentions": anatomy_info["raw_mentions"],
            "semantic_types": anatomy_info["semantic_types"],
            "normalized_region": normalized_region,
        }

        # Load image
        image = self.image_loader.load(image_path)

        # Encode & fuse
        with torch.no_grad():
            img_emb = self.image_encoder.encode_image(image).to(self.device)
            txt_emb = self.text_encoder.encode_text(combined_text).to(self.device)

            assert img_emb.shape == txt_emb.shape, "Embedding dim mismatch"

            fused = self.fusion_model(
                img_emb.unsqueeze(0),
                txt_emb.unsqueeze(0),
            )

        fused_np = fused.squeeze(0).cpu().numpy()

        if np.isnan(fused_np).any():
            raise ValueError("NaN in fused embedding")

        embedding_id = len(self.embeddings)
        self.embeddings.append(fused_np)

        return KBEntry(
            case_id=f"case_{idx:06d}",
            image_path=str(image_path),
            diagnosis_label=category,
            clinical_text={
                "context": context,
                "description": description,
                "combined": combined_text,
            },
            anatomy=anatomy,
            embedding_id=embedding_id,
        )

    def _finalize_flat(self):
        """Finalize flat KB (original logic)"""
        if not self.embeddings:
            raise RuntimeError("No valid entries")

        embeddings = np.vstack(self.embeddings).astype("float32")

        # FAISS index
        index = KBIndex(dim=embeddings.shape[1])
        index.add(embeddings)

        # Save
        self.storage.save_embeddings(
            embeddings,
            self.output_dir / "embeddings.npy"
        )

        self.storage.save_metadata(
            [e.__dict__ for e in self.entries],
            self.output_dir / "metadata.json"
        )

        index.save(self.output_dir / "index.faiss")

        self.storage.save_metadata(
            {
                "num_entries": len(self.entries),
                "embedding_dim": embeddings.shape[1],
                "device": self.device,
                "mode": "flat",
            },
            self.output_dir / "kb_config.json"
        )

        self.report.finalize()
        self.report.save(self.output_dir / "build_report.json")
        
        print("\n" + "="*70)
        print("✓ FLAT KB BUILD COMPLETE")
        print(f"✓ Entries: {len(self.entries)}")
        print("="*70)

    def _finalize_concept(self):
        """Finalize concept KB"""
        import faiss
        
        if not self.concepts:
            raise RuntimeError("No valid concepts")
        
        concepts = list(self.concepts.values())
        
        # Extract embeddings
        text_embeddings = np.array([c['text_embedding'] for c in concepts]).astype('float32')
        image_embeddings = np.array([c['aggregated_image_embedding'] for c in concepts]).astype('float32')
        concept_embeddings = np.array([c['concept_embedding'] for c in concepts]).astype('float32')
        
        # Normalize
        faiss.normalize_L2(text_embeddings)
        faiss.normalize_L2(image_embeddings)
        faiss.normalize_L2(concept_embeddings)
        
        # Build indices
        text_index = faiss.IndexFlatIP(512)
        text_index.add(text_embeddings)
        
        image_index = faiss.IndexFlatIP(512)
        image_index.add(image_embeddings)
        
        concept_index = faiss.IndexFlatIP(512)
        concept_index.add(concept_embeddings)
        
        # Save embeddings
        np.save(self.output_dir / "text_embeddings.npy", text_embeddings)
        np.save(self.output_dir / "image_embeddings.npy", image_embeddings)
        np.save(self.output_dir / "concept_embeddings.npy", concept_embeddings)
        
        # Save indices
        faiss.write_index(text_index, str(self.output_dir / "text_index.faiss"))
        faiss.write_index(image_index, str(self.output_dir / "image_index.faiss"))
        faiss.write_index(concept_index, str(self.output_dir / "concept_index.faiss"))
        
        # Save metadata (clean for JSON)
        metadata = []
        for c in concepts:
            metadata.append({
                'concept_id': c['concept_id'],
                'canonical_text': c['canonical_text'],
                'diagnosis_label': c['diagnosis_label'],
                'image_paths': c['image_paths'],
                'num_images': c['num_images'],
                'image_variance': c['image_variance'],
            })
        
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Build image->concept mapping
        image_to_concept = {}
        for c in concepts:
            for img_path in c['image_paths']:
                image_to_concept[img_path] = c['concept_id']
        
        with open(self.output_dir / "image_to_concept.json", 'w') as f:
            json.dump(image_to_concept, f, indent=2)
        
        # Save config
        config = {
            'num_concepts': len(concepts),
            'num_images': sum(c['num_images'] for c in concepts),
            'embedding_dim': 512,
            'mode': 'concept',
            'aggregation_method': self.aggregation_method,
        }
        
        with open(self.output_dir / "kb_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        self.report.finalize()
        self.report.save(self.output_dir / "build_report.json")
        
        print("\n" + "="*70)
        print("✓ CONCEPT KB BUILD COMPLETE")
        print(f"✓ Concepts: {len(concepts)}")
        print(f"✓ Images: {config['num_images']}")
        print(f"✓ Avg images/concept: {config['num_images']/len(concepts):.1f}")
        print("="*70)