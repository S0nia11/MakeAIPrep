"""
MakeAIPrep - Model 4: CLIP + MLP Recommandation Multimodale
=============================================================
Modele de recommandation combinant image (visage/tenue) et texte (evenement).

Pourquoi ce modele:
- Combine les informations visuelles ET textuelles en un seul modele
- Peut recommander des styles en fonction du contexte (type d'evenement)
- CLIP fournit des embeddings riches pre-entraines
- Le MLP apprend a fusionner et adapter ces embeddings pour notre tache
- Architecture simple et efficace (pas besoin d'un gros modele)

Architecture:
- CLIP ViT-B/32 (gele) -> image embedding (512d)
- CLIP Text Encoder (gele) -> text embedding (512d)
- Concatenation [image; text] -> 1024d
- MLP: 1024 -> 512 -> 256 -> num_classes
- Sorties: probabilites par style + embedding commun pour recommandation

C'est le modele le plus adapte a MakeAIPrep car il permet:
1. "Je vais a un entretien" + [photo] -> recommandation de style
2. Matching entre l'apparence actuelle et les styles recommandes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from typing import List, Dict, Optional, Tuple


class FusionMLP(nn.Module):
    """MLP de fusion multimodale image + texte."""

    def __init__(
        self,
        input_dim: int = 1024,  # 512 image + 512 text
        hidden_dims: List[int] = [512, 256],
        num_classes: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
            ])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (B, num_classes) pour la classification
            features: (B, last_hidden_dim) embedding fusionne pour la recommandation
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features


class CLIPMLPRecommender(nn.Module):
    """Modele CLIP + MLP pour la recommandation de styles multimodale."""

    def __init__(
        self,
        num_classes: int = 8,
        clip_model_name: str = "ViT-B-32",
        pretrained_dataset: str = "openai",
        mlp_hidden_dims: List[int] = [512, 256],
        dropout: float = 0.2,
        freeze_clip: bool = True,
    ):
        super().__init__()
        self.model_name = "CLIP+MLP"
        self.num_classes = num_classes

        # Charger CLIP (avec fallback si pas de connexion)
        try:
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                clip_model_name, pretrained=pretrained_dataset
            )
            print("[CLIP+MLP] Poids pre-entraines charges avec succes")
        except Exception as e:
            print(f"[CLIP+MLP] Impossible de telecharger les poids: {e}")
            print("[CLIP+MLP] Utilisation du modele sans pre-entrainement")
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                clip_model_name, pretrained=""
            )
        self.tokenizer = open_clip.get_tokenizer(clip_model_name)

        # Dimension des embeddings CLIP
        embed_dim = self.clip_model.visual.output_dim  # 512 pour ViT-B-32

        # Projection optionnelle des embeddings
        self.image_proj = nn.Linear(embed_dim, embed_dim)
        self.text_proj = nn.Linear(embed_dim, embed_dim)

        # MLP de fusion
        self.fusion_mlp = FusionMLP(
            input_dim=embed_dim * 2,  # image + text concatenes
            hidden_dims=mlp_hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
        )

        # Geler CLIP si demande
        if freeze_clip:
            self._freeze_clip()

    def _freeze_clip(self):
        """Gele les poids CLIP (on n'entraine que le MLP)."""
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def unfreeze_clip(self):
        """Degele CLIP pour le fine-tuning complet (optionnel, epoch tardive)."""
        for param in self.clip_model.parameters():
            param.requires_grad = True

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode les images avec CLIP + projection."""
        with torch.set_grad_enabled(self.image_proj.weight.requires_grad):
            clip_features = self.clip_model.encode_image(images)
            clip_features = clip_features.float()
            clip_features = F.normalize(clip_features, dim=-1)
        return self.image_proj(clip_features)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode les textes avec CLIP + projection."""
        device = next(self.parameters()).device
        tokens = self.tokenizer(texts).to(device)
        with torch.set_grad_enabled(self.text_proj.weight.requires_grad):
            clip_features = self.clip_model.encode_text(tokens)
            clip_features = clip_features.float()
            clip_features = F.normalize(clip_features, dim=-1)
        return self.text_proj(clip_features)

    def forward(
        self,
        images: torch.Tensor,
        texts: Optional[List[str]] = None,
        text_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (B, 3, 224, 224) images
            texts: Liste de B descriptions textuelles (ex: "entretien d'embauche")
            text_features: (B, embed_dim) features textuelles pre-calculees (optionnel)

        Returns:
            logits: (B, num_classes)
            fused_features: (B, 256) embedding fusionne
        """
        # Encoder les images
        img_features = self.encode_image(images)

        # Encoder les textes
        if text_features is not None:
            txt_features = text_features
        elif texts is not None:
            txt_features = self.encode_text(texts)
        else:
            # Si pas de texte, utiliser un vecteur nul (mode image-only)
            txt_features = torch.zeros_like(img_features)

        # Fusion par concatenation
        fused = torch.cat([img_features, txt_features], dim=-1)

        # MLP
        logits, features = self.fusion_mlp(fused)

        return logits, features

    def recommend(
        self,
        image: torch.Tensor,
        event_description: str,
        style_names: List[str],
        top_k: int = 3,
    ) -> Dict:
        """Mode recommandation: donne les top-K styles pour un evenement donne.

        Args:
            image: (1, 3, 224, 224) photo de l'utilisateur
            event_description: "Entretien d'alternance chez Google"
            style_names: Liste des noms de styles
            top_k: Nombre de recommandations

        Returns:
            Dict avec les recommandations classees
        """
        self.eval()
        with torch.no_grad():
            logits, features = self.forward(image, texts=[event_description])
            probs = F.softmax(logits, dim=-1).squeeze()

        # Trier par probabilite
        sorted_indices = torch.argsort(probs, descending=True)

        recommendations = []
        for i in range(min(top_k, len(style_names))):
            idx = sorted_indices[i].item()
            recommendations.append({
                "rank": i + 1,
                "style": style_names[idx],
                "confidence": probs[idx].item(),
            })

        return {
            "event": event_description,
            "recommendations": recommendations,
            "fused_embedding": features.squeeze().cpu().numpy(),
        }


def create_clip_mlp_model(
    num_classes: int = 8,
    clip_model_name: str = "ViT-B-32",
    pretrained_dataset: str = "openai",
    mlp_hidden_dims: List[int] = [512, 256],
    dropout: float = 0.2,
    freeze_clip: bool = True,
) -> CLIPMLPRecommender:
    """Factory function pour creer le modele CLIP + MLP."""
    return CLIPMLPRecommender(
        num_classes=num_classes,
        clip_model_name=clip_model_name,
        pretrained_dataset=pretrained_dataset,
        mlp_hidden_dims=mlp_hidden_dims,
        dropout=dropout,
        freeze_clip=freeze_clip,
    )
