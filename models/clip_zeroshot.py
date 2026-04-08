"""
MakeAIPrep - Model 3: CLIP Zero-Shot Classification
=====================================================
Classification de styles SANS entrainement avec CLIP d'OpenAI.

Pourquoi ce modele:
- Aucun entrainement necessaire = pas besoin de dataset labellise
- CLIP comprend le langage naturel -> on decrit les styles en texte
- Excellent pour le prototypage rapide et la validation du concept
- Peut etre utilise comme "upper baseline" pour valider les labels

Architecture:
- CLIP ViT-B/32 pre-entraine (OpenAI)
- Encode l'image ET des descriptions textuelles des styles
- Cosine similarity entre l'embedding image et chaque style textuel
- Le style avec la plus haute similarite est la prediction
"""

import torch
import torch.nn as nn
import open_clip
from PIL import Image
from typing import List, Dict, Tuple, Optional
import numpy as np


class CLIPZeroShotClassifier(nn.Module):
    """CLIP zero-shot pour classifier les styles sans entrainement."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained_dataset: str = "openai",
        style_categories: Optional[List[str]] = None,
    ):
        super().__init__()
        self.model_name_id = "CLIP-ZeroShot"

        # Charger CLIP (avec fallback si pas de connexion)
        try:
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained_dataset
            )
            print("[CLIP-ZeroShot] Poids pre-entraines charges avec succes")
        except Exception as e:
            print(f"[CLIP-ZeroShot] Impossible de telecharger les poids: {e}")
            print("[CLIP-ZeroShot] Utilisation du modele sans pre-entrainement")
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=""
            )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Categories de styles par defaut
        if style_categories is None:
            style_categories = [
                "casual", "business", "elegant", "streetwear",
                "sportswear", "boheme", "chic", "minimaliste",
            ]
        self.style_categories = style_categories

        # Templates de prompts pour chaque style (en francais et anglais)
        self.text_prompts = self._build_prompts()

        # Pre-calculer les embeddings textuels
        self._text_features = None

        # Geler tous les parametres (zero-shot = pas d'entrainement)
        for param in self.parameters():
            param.requires_grad = False

    def _build_prompts(self) -> Dict[str, List[str]]:
        """Construit des prompts descriptifs pour chaque style."""
        templates = {
            "casual": [
                "a person wearing casual everyday clothes",
                "someone in a relaxed casual outfit with jeans and t-shirt",
                "casual comfortable daily wear clothing",
            ],
            "business": [
                "a person wearing professional business attire",
                "someone in a formal suit for a job interview",
                "professional office wear with blazer and dress pants",
            ],
            "elegant": [
                "a person wearing elegant formal evening wear",
                "someone in a sophisticated elegant dress or suit",
                "refined elegant outfit for a gala or formal event",
            ],
            "streetwear": [
                "a person wearing trendy urban streetwear fashion",
                "someone in modern streetwear with sneakers and hoodie",
                "urban street style fashion outfit",
            ],
            "sportswear": [
                "a person wearing athletic sportswear",
                "someone in sporty activewear clothing",
                "athletic and sporty outfit for exercise",
            ],
            "boheme": [
                "a person wearing bohemian free-spirited clothing",
                "someone in boho style with flowing fabrics and patterns",
                "bohemian artistic fashion with natural textures",
            ],
            "chic": [
                "a person wearing fashionable chic clothing",
                "someone in a trendy and stylish modern outfit",
                "chic and fashionable contemporary wear",
            ],
            "minimaliste": [
                "a person wearing minimalist clean clothing",
                "someone in a simple minimalist outfit with neutral colors",
                "clean minimalist fashion with essential pieces only",
            ],
        }
        # Ne garder que les categories definies
        return {k: v for k, v in templates.items() if k in self.style_categories}

    @torch.no_grad()
    def _encode_text_prompts(self, device: torch.device) -> torch.Tensor:
        """Encode tous les prompts textuels et moyenne par categorie."""
        all_features = []
        for style in self.style_categories:
            prompts = self.text_prompts.get(style, [f"a photo of {style} style clothing"])
            tokens = self.tokenizer(prompts).to(device)
            features = self.clip_model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            # Moyenne des prompts pour ce style
            mean_feature = features.mean(dim=0)
            mean_feature = mean_feature / mean_feature.norm()
            all_features.append(mean_feature)

        return torch.stack(all_features)  # (num_classes, embed_dim)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classifie les images par similarite cosinus avec les styles textuels.

        Args:
            x: Batch d'images (B, 3, 224, 224)

        Returns:
            Similarites normalisees (B, num_classes) - interpretables comme probabilites
        """
        device = x.device

        # Encoder les images
        image_features = self.clip_model.encode_image(x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Encoder les textes (cache pour eviter de re-calculer)
        if self._text_features is None or self._text_features.device != device:
            self._text_features = self._encode_text_prompts(device)

        # Similarite cosinus
        similarity = image_features @ self._text_features.T

        # Mise a l'echelle (temperature CLIP)
        logit_scale = self.clip_model.logit_scale.exp()
        similarity = similarity * logit_scale

        return similarity

    @torch.no_grad()
    def predict_with_details(self, image: Image.Image) -> Dict:
        """Prediction detaillee pour une seule image (mode inference)."""
        device = next(self.parameters()).device
        img_tensor = self.preprocess(image).unsqueeze(0).to(device)

        similarity = self.forward(img_tensor)
        probs = torch.softmax(similarity, dim=-1).squeeze()

        results = {}
        for i, style in enumerate(self.style_categories):
            results[style] = {
                "probability": probs[i].item(),
                "rank": 0,  # sera rempli apres
            }

        # Ajouter les rangs
        sorted_styles = sorted(results.keys(), key=lambda s: results[s]["probability"], reverse=True)
        for rank, style in enumerate(sorted_styles, 1):
            results[style]["rank"] = rank

        return {
            "predictions": results,
            "top_style": sorted_styles[0],
            "top_probability": results[sorted_styles[0]]["probability"],
            "top3": sorted_styles[:3],
        }


def create_clip_zeroshot(
    model_name: str = "ViT-B-32",
    pretrained_dataset: str = "openai",
    style_categories: Optional[List[str]] = None,
) -> CLIPZeroShotClassifier:
    """Factory function pour creer le modele CLIP zero-shot."""
    return CLIPZeroShotClassifier(
        model_name=model_name,
        pretrained_dataset=pretrained_dataset,
        style_categories=style_categories,
    )
