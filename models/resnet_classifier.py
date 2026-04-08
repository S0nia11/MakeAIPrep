"""
MakeAIPrep - Model 1: ResNet50 Baseline
========================================
Classification de styles vestimentaires avec ResNet50 pre-entraine (ImageNet).

Pourquoi ce modele:
- Baseline solide et bien compris dans la communaute
- Rapide a entrainer et leger en inference
- Bon point de comparaison pour evaluer les modeles plus complexes

Architecture:
- ResNet50 backbone (pre-entraine ImageNet)
- Global Average Pooling
- Dropout + FC head pour N classes de styles
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class ResNetStyleClassifier(nn.Module):
    """ResNet50 fine-tune pour la classification de styles."""

    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.model_name = "ResNet50"
        self.num_classes = num_classes

        # Charger ResNet50 (avec fallback si pas de connexion)
        if pretrained:
            try:
                weights = models.ResNet50_Weights.DEFAULT
                self.backbone = models.resnet50(weights=weights)
                print("[ResNet50] Poids pre-entraines charges avec succes")
            except Exception as e:
                print(f"[ResNet50] Impossible de telecharger les poids: {e}")
                print("[ResNet50] Utilisation du modele sans pre-entrainement")
                self.backbone = models.resnet50(weights=None)
        else:
            self.backbone = models.resnet50(weights=None)

        # Recuperer la dimension du dernier layer
        in_features = self.backbone.fc.in_features  # 2048

        # Remplacer le classificateur final
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Gele toutes les couches sauf le classificateur."""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Degele toutes les couches pour le fine-tuning complet."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extrait les features avant le classificateur (pour visualisation)."""
        modules = list(self.backbone.children())[:-1]
        feature_extractor = nn.Sequential(*modules)
        features = feature_extractor(x)
        return features.flatten(1)


def create_resnet_model(
    num_classes: int = 8,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = True,
) -> ResNetStyleClassifier:
    """Factory function pour creer le modele ResNet."""
    return ResNetStyleClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
