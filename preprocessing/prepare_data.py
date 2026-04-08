"""
MakeAIPrep - Preparation des donnees
======================================

Ce script:
1. Telecharge un dataset de vetements/styles depuis HuggingFace ou cree un dataset de demo
2. Organise les images par categorie de style
3. Genere le fichier metadata.csv

Usage:
    python prepare_data.py --mode demo      # Dataset de demo (petites images generees)
    python prepare_data.py --mode download   # Telecharge un vrai dataset
"""

import os
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import pandas as pd


STYLE_CATEGORIES = [
    "casual", "business", "elegant", "streetwear",
    "sportswear", "boheme", "chic", "minimaliste",
]

DATA_DIR = Path("./data")


def create_demo_dataset(num_per_class: int = 100, image_size: int = 224):
    """Cree un dataset synthetique de demo pour tester le pipeline.

    Genere des images colorees avec des patterns differents par style.
    Ce n'est PAS un vrai dataset - c'est uniquement pour valider
    que le code fonctionne avant d'utiliser des vraies images.
    """
    print("Creation du dataset de demo...")
    images_dir = DATA_DIR / "images"

    # Couleurs dominantes par style (pour differentiation)
    style_colors = {
        "casual": [(100, 130, 180), (180, 160, 130)],      # Bleu jean, beige
        "business": [(40, 40, 50), (255, 255, 255)],        # Noir, blanc
        "elegant": [(150, 50, 50), (200, 170, 100)],        # Rouge, or
        "streetwear": [(80, 80, 80), (200, 50, 50)],        # Gris, rouge vif
        "sportswear": [(50, 150, 50), (50, 50, 200)],       # Vert, bleu sport
        "boheme": [(180, 120, 80), (100, 150, 100)],        # Terre, vert nature
        "chic": [(230, 200, 220), (100, 100, 100)],         # Rose pale, gris
        "minimaliste": [(240, 240, 240), (200, 200, 200)],  # Blanc, gris clair
    }

    metadata_rows = []

    for style in STYLE_CATEGORIES:
        style_dir = images_dir / style
        style_dir.mkdir(parents=True, exist_ok=True)
        colors = style_colors[style]

        for i in range(num_per_class):
            # Generer une image avec des patterns
            img = np.zeros((image_size, image_size, 3), dtype=np.uint8)

            # Couleur de base avec variation
            base_color = random.choice(colors)
            noise = np.random.randint(-20, 20, 3)
            color = np.clip(np.array(base_color) + noise, 0, 255).astype(np.uint8)
            img[:] = color

            # Ajouter du bruit pour la texture
            texture = np.random.randint(-15, 15, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + texture, 0, 255).astype(np.uint8)

            # Sauvegarder
            img_pil = Image.fromarray(img)
            filename = f"{style}_{i:04d}.jpg"
            img_pil.save(style_dir / filename, quality=85)

            metadata_rows.append({
                "image_path": f"images/{style}/{filename}",
                "style": style,
                "event_type": random.choice([
                    "entretien_embauche", "soiree_networking",
                    "journee_campus", "presentation_projet",
                ]),
                "description": f"Tenue {style} pour un contexte professionnel etudiant",
            })

    # Sauvegarder les metadonnees
    df = pd.DataFrame(metadata_rows)
    df.to_csv(DATA_DIR / "metadata.csv", index=False)

    print(f"Dataset de demo cree: {len(metadata_rows)} images")
    print(f"  - {num_per_class} images par classe")
    print(f"  - {len(STYLE_CATEGORIES)} classes: {', '.join(STYLE_CATEGORIES)}")
    print(f"  - Dossier: {DATA_DIR.resolve()}")


def download_fashion_dataset():
    """Telecharge un dataset de mode reel.

    Options recommandees:
    1. HuggingFace: "ashraq/fashion-product-images-small"
    2. Kaggle: "paramaggarwal/fashion-product-images-dataset"
    3. DeepFashion (recherche academique)

    Note: Vous devrez mapper les categories du dataset vers les 8 styles MakeAIPrep.
    """
    print("Telechargement du dataset reel...")

    try:
        from datasets import load_dataset

        print("Chargement depuis HuggingFace: 'ashraq/fashion-product-images-small'")
        ds = load_dataset("ashraq/fashion-product-images-small", split="train")

        # Mapping des categories du dataset vers nos styles
        category_mapping = {
            "Topwear": "casual",
            "Bottomwear": "casual",
            "Dress": "elegant",
            "Innerwear": "casual",
            "Shoes": "casual",
            "Accessories": "chic",
            "Bags": "chic",
            "Watches": "business",
            "Jewellery": "elegant",
            "Belts": "business",
        }

        images_dir = DATA_DIR / "images"
        metadata_rows = []
        counts = {style: 0 for style in STYLE_CATEGORIES}
        max_per_class = 500  # Limiter pour equilibrer

        for item in ds:
            if item.get("image") is None:
                continue

            # Mapper la categorie
            cat = item.get("masterCategory", "")
            style = category_mapping.get(cat, None)
            if style is None or counts[style] >= max_per_class:
                continue

            # Sauvegarder l'image
            style_dir = images_dir / style
            style_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{style}_{counts[style]:04d}.jpg"
            img = item["image"].convert("RGB").resize((224, 224))
            img.save(style_dir / filename, quality=90)

            metadata_rows.append({
                "image_path": f"images/{style}/{filename}",
                "style": style,
                "event_type": "general",
                "description": item.get("productDisplayName", f"Tenue {style}"),
            })
            counts[style] += 1

        df = pd.DataFrame(metadata_rows)
        df.to_csv(DATA_DIR / "metadata.csv", index=False)

        print(f"Dataset telecharge: {len(metadata_rows)} images")
        for style, count in counts.items():
            print(f"  {style}: {count} images")

    except ImportError:
        print("Le package 'datasets' n'est pas installe.")
        print("Installez-le: pip install datasets")
        print("Ou utilisez --mode demo pour un dataset synthetique")


def main():
    parser = argparse.ArgumentParser(description="Preparation des donnees MakeAIPrep")
    parser.add_argument(
        "--mode", choices=["demo", "download"], default="demo",
        help="'demo' pour un dataset synthetique, 'download' pour un vrai dataset"
    )
    parser.add_argument("--num-per-class", type=int, default=100)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "demo":
        create_demo_dataset(num_per_class=args.num_per_class)
    elif args.mode == "download":
        download_fashion_dataset()

    # Verifier la structure
    print("\nStructure du dataset:")
    for style_dir in sorted((DATA_DIR / "images").glob("*")):
        if style_dir.is_dir():
            count = len(list(style_dir.glob("*.jpg")))
            print(f"  {style_dir.name}: {count} images")


if __name__ == "__main__":
    main()
