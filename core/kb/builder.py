"""
KB Builder - FIXED: Uses context+description for KB text
(NOT used in training, only for retrieval reference)
"""
import csv
from pathlib import Path
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
    KB builder using context+description for reference text.
    
    IMPORTANT: This is ONLY for the KB (retrieval reference).
    Training uses ONLY question+image.
    """

    def __init__(
        self,
        image_encoder,
        text_encoder,
        fusion_model,
        output_dir: str,
        image_root: str,
        device: str = "cpu",
    ):
        self.device = device

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

        self.entries: list[KBEntry] = []
        self.embeddings: list[np.ndarray] = []

    def build(self, csv_path: str):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        print("\n" + "="*70)
        print("BUILDING KB - Uses context+description for reference")
        print("(Training uses question+image only)")
        print("="*70 + "\n")

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for idx, row in enumerate(
                tqdm(reader, desc="Building KB", unit="rows")
            ):
                self.report.log_row_seen()

                try:
                    entry = self._process_row(row, idx)
                    self.entries.append(entry)

                    self.report.log_success(
                        category=entry.diagnosis_label,
                        anatomy_region=entry.anatomy["normalized_region"],
                    )

                except Exception as e:
                    self.report.log_failure(idx, str(e))
                    continue

        self._finalize()

    def _process_row(self, row: dict, idx: int) -> KBEntry:
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
            case_id=f"clipsyntel_{idx:06d}",
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

    def _finalize(self):
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
                "kb_text": "context+description",
                "training_text": "question_only"
            },
            self.output_dir / "kb_config.json"
        )

        self.report.finalize()
        self.report.save(self.output_dir / "build_report.json")
        
        print("\n" + "="*70)
        print("✓ KB BUILD COMPLETE")
        print(f"✓ Entries: {len(self.entries)}")
        print(f"✓ KB text: context+description (for reference)")
        print(f"✓ Training used: question+image only")
        print("="*70)