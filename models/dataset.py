"""
MakeAIPrep - Dataset & DataLoaders pour le pipeline de relooking.

Gere:
- Chargement des images de styles vestimentaires
- Augmentation des donnees
- Creation des DataLoaders train/val/test
"""

import os
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd


class StyleDataset(Dataset):
    """Dataset pour la classification de styles vestimentaires.

    Structure attendue du dossier data/:
        data/
        ├── images/
        │   ├── casual/
        │   │   ├── img001.jpg
        │   │   └── ...
        │   ├── business/
        │   ├── elegant/
        │   └── ...
        └── metadata.csv  (optionnel: image_path, style, event_type, description)
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        split: str = "train",
        metadata_file: Optional[str] = None,
    ):
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.transform = transform
        self.split = split

        # Charger les categories depuis les sous-dossiers
        self.classes = sorted([
            d.name for d in self.images_dir.iterdir()
            if d.is_dir()
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # Charger les chemins d'images et labels
        self.samples = []
        for class_name in self.classes:
            class_dir = self.images_dir / class_name
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))

        # Charger les metadonnees si disponibles
        self.metadata = None
        meta_path = metadata_file or (self.root_dir / "metadata.csv")
        if Path(meta_path).exists():
            self.metadata = pd.read_csv(meta_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_metadata(self, idx: int) -> Optional[Dict]:
        """Retourne les metadonnees associees a une image."""
        if self.metadata is not None:
            img_path = self.samples[idx][0]
            filename = Path(img_path).name
            row = self.metadata[self.metadata["image_path"].str.contains(filename)]
            if not row.empty:
                return row.iloc[0].to_dict()
        return None


class StyleDatasetWithText(StyleDataset):
    """Dataset etendu avec description textuelle (pour CLIP + MLP).

    Chaque sample retourne (image, texte, label).
    Le texte combine le type d'evenement et la description du style.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Descriptions par defaut par categorie de style
        self.default_descriptions = {
            "casual": "Tenue decontractee et confortable pour le quotidien",
            "business": "Tenue professionnelle formelle pour le bureau",
            "elegant": "Tenue sophistiquee pour les occasions speciales",
            "streetwear": "Style urbain moderne et tendance",
            "sportswear": "Tenue sportive et dynamique",
            "boheme": "Style libre et artistique avec des motifs naturels",
            "chic": "Tenue raffinee et a la mode",
            "minimaliste": "Style epure avec des pieces essentielles",
        }

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Recuperer la description textuelle
        meta = self.get_metadata(idx)
        if meta and "description" in meta:
            text = meta["description"]
        else:
            class_name = self.idx_to_class[label]
            text = self.default_descriptions.get(class_name, f"Style {class_name}")

        return image, text, label


def get_transforms(image_size: int = 224, augment: bool = True) -> Dict[str, transforms.Compose]:
    """Retourne les transformations pour train/val/test."""

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    return {
        "train": train_transform,
        "val": eval_transform,
        "test": eval_transform,
    }


def create_dataloaders(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.8,
    val_split: float = 0.1,
    with_text: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Cree les DataLoaders train/val/test.

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    tfms = get_transforms(image_size, augment=True)

    DatasetClass = StyleDatasetWithText if with_text else StyleDataset

    full_dataset = DatasetClass(
        root_dir=data_dir,
        transform=tfms["train"],
        split="train",
    )

    # Split train/val/test
    total = len(full_dataset)
    train_size = int(total * train_split)
    val_size = int(total * val_split)
    test_size = total - train_size - val_size

    train_set, val_set, test_set = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Appliquer les bonnes transformations aux splits val/test
    # (on utilise un wrapper pour ca)
    val_set.dataset = DatasetClass(
        root_dir=data_dir,
        transform=tfms["val"],
        split="val",
    )
    test_set.dataset = DatasetClass(
        root_dir=data_dir,
        transform=tfms["test"],
        split="test",
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, full_dataset.classes
