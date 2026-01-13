import torch
import torch.nn.functional as F
import open_clip
from typing import List, Optional
from PIL import Image
from pathlib import Path
from peft import PeftModel


class BioMedCLIPEncoder:
    """
    Production-grade BioMedCLIP encoder (OpenCLIP-based).

    ✔ Correct OpenCLIP loading
    ✔ Correct PEFT / LoRA loading
    ✔ No silent failures
    """

    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        lora_path: Optional[str] = None,
    ):
        self.device = device

        # ---- Load base OpenCLIP model ----
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # ---- Load LoRA adapters (CORRECT WAY) ----
        if lora_path is not None:
            lora_path = Path(lora_path)
            if not lora_path.exists():
                raise FileNotFoundError(f"LoRA path not found: {lora_path}")

            self.model = PeftModel.from_pretrained(
                self.model,
                lora_path,
                is_trainable=False
            )

        self.model = self.model.to(self.device)
        self.model.eval()

        self.dim = 512  # BioMedCLIP fixed embedding size

        # ---- HARD ASSERTION: LoRA must be active if provided ----
        if lora_path is not None:
            lora_norm = sum(
                p.abs().sum().item()
                for n, p in self.model.named_parameters()
                if "lora" in n
            )
            if lora_norm == 0.0:
                raise RuntimeError(
                    "LoRA adapters loaded but inactive (zero norm). "
                    "Check target_modules mismatch between training and inference."
                )

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------
    @staticmethod
    def _l2norm(x: torch.Tensor):
        return F.normalize(x, dim=-1)

    # --------------------------------------------------
    # IMAGE EMBEDDINGS
    # --------------------------------------------------
    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Returns:
            Tensor of shape (512,)
        """
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        emb = self.model.encode_image(image_tensor)
        return self._l2norm(emb).squeeze(0)

    @torch.no_grad()
    def encode_image_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Returns:
            Tensor of shape (B, 512)
        """
        image_tensors = torch.stack(
            [self.preprocess(img) for img in images]
        ).to(self.device)

        emb = self.model.encode_image(image_tensors)
        return self._l2norm(emb)

    # --------------------------------------------------
    # TEXT EMBEDDINGS
    # --------------------------------------------------
    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """
        Returns:
            Tensor of shape (512,)
        """
        tokens = self.tokenizer([text]).to(self.device)
        emb = self.model.encode_text(tokens)
        return self._l2norm(emb).squeeze(0)

    @torch.no_grad()
    def encode_text_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Returns:
            Tensor of shape (B, 512)
        """
        tokens = self.tokenizer(texts).to(self.device)
        emb = self.model.encode_text(tokens)
        return self._l2norm(emb)
