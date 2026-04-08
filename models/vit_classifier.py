"""
MakeAIPrep - Model 2: Vision Transformer (ViT) Fine-Tune
==========================================================
Classification fine de styles avec ViT-B/16 pre-entraine (Google).

Pourquoi ce modele:
- Les ViT capturent mieux les relations globales dans l'image
- Excellents pour la classification fine (intensite maquillage, type de coiffure)
- Performants sur des datasets moyens (10k-50k images) avec fine-tuning
- Attention maps interpretables (on peut voir OU le modele regarde)

Architecture:
- ViT-B/16 backbone (pre-entraine ImageNet-21k via HuggingFace)
- [CLS] token -> MLP head pour N classes de styles
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from typing import Optional, Tuple


class ViTStyleClassifier(nn.Module):
    """Vision Transformer fine-tune pour la classification de styles."""

    def __init__(
        self,
        num_classes: int = 8,
        model_name: str = "google/vit-base-patch16-224",
        pretrained: bool = True,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.model_name = "ViT-B/16"
        self.num_classes = num_classes

        # Charger le ViT (avec fallback si pas de connexion)
        if pretrained:
            try:
                self.vit = ViTModel.from_pretrained(model_name)
                print("[ViT] Poids pre-entraines charges avec succes")
            except Exception as e:
                print(f"[ViT] Impossible de telecharger les poids: {e}")
                print("[ViT] Utilisation du modele sans pre-entrainement")
                config = ViTConfig(
                    hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                    intermediate_size=3072, image_size=224, patch_size=16,
                )
                self.vit = ViTModel(config)
        else:
            config = ViTConfig(
                hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                intermediate_size=3072, image_size=224, patch_size=16,
            )
            self.vit = ViTModel(config)

        hidden_size = self.vit.config.hidden_size  # 768 pour ViT-B

        # Classificateur MLP
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Gele le backbone ViT."""
        for param in self.vit.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, num_layers: Optional[int] = None):
        """Degele le backbone ViT (tout ou les N derniers layers)."""
        if num_layers is None:
            for param in self.vit.parameters():
                param.requires_grad = True
        else:
            # Degeler les N derniers encoder layers
            total_layers = len(self.vit.encoder.layer)
            for i, layer in enumerate(self.vit.encoder.layer):
                if i >= total_layers - num_layers:
                    for param in layer.parameters():
                        param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ViT attend pixel_values, pas besoin de tokenizer pour les images
        outputs = self.vit(pixel_values=x)
        # Utiliser le [CLS] token (premier token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne les attention maps (utile pour l'interpretabilite).

        Permet de visualiser OU le modele regarde pour classifier:
        - Regarde-t-il le visage? les vetements? les accessoires?
        """
        outputs = self.vit(pixel_values=x, output_attentions=True)
        # Moyenner sur toutes les tetes et couches
        attentions = torch.stack(outputs.attentions)  # (layers, batch, heads, seq, seq)
        avg_attention = attentions.mean(dim=(0, 2))  # (batch, seq, seq)
        # Attention du CLS token vers les patches
        cls_attention = avg_attention[:, 0, 1:]  # (batch, num_patches)
        return cls_attention

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extrait le CLS embedding."""
        outputs = self.vit(pixel_values=x)
        return outputs.last_hidden_state[:, 0, :]


def create_vit_model(
    num_classes: int = 8,
    model_name: str = "google/vit-base-patch16-224",
    pretrained: bool = True,
    dropout: float = 0.1,
    freeze_backbone: bool = True,
) -> ViTStyleClassifier:
    """Factory function pour creer le modele ViT."""
    return ViTStyleClassifier(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
