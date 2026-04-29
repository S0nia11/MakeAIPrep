"""
MakeAIPrep - Script principal d'entrainement et benchmark
===========================================================

Ce script entraine les 4 modeles, les compare et genere un rapport
pour determiner quel(s) modele(s) garder.

Usage:
    # Entrainer tous les modeles et comparer
    python train.py --all

    # Entrainer un seul modele
    python train.py --model resnet
    python train.py --model vit
    python train.py --model clip_zeroshot
    python train.py --model clip_mlp

    # Benchmark uniquement (apres entrainement)
    python train.py --benchmark-only
"""

import argparse
import json
import os
import sys
import ssl

# Fix SSL pour Windows (erreur de connexion lors du telechargement des poids)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
from pathlib import Path

import torch
import yaml

from models.resnet_classifier import create_resnet_model
from models.vit_classifier import create_vit_model
from models.clip_zeroshot import create_clip_zeroshot
from models.clip_mlp_recommender import create_clip_mlp_model
from models.dataset import create_dataloaders
from models.trainer import Trainer
from models.metrics import compute_metrics, count_parameters, measure_inference_time
from models.visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_benchmark_comparison,
    generate_recommendation_report,
)


def load_config(config_path: str = "./models/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: Apple MPS")
    else:
        device = torch.device("cpu")
        print("Device: CPU (entrainement sera plus lent)")
    return device


def train_resnet(config: dict, device: torch.device, data_dir: str, save_dir: str) -> dict:
    """Entraine le modele ResNet50 baseline."""
    print("\n" + "=" * 60)
    print("MODEL 1: ResNet50 - Baseline CNN")
    print("=" * 60)

    cfg = config["models"]["resnet50"]
    num_classes = len(config["style_categories"])

    model = create_resnet_model(
        num_classes=num_classes,
        pretrained=cfg["pretrained"],
        dropout=cfg["dropout"],
        freeze_backbone=False,  # On gele manuellement apres si pretrained a marche
    )
    # Ne geler que si les poids pre-entraines sont charges
    has_pretrained = any("weight" in k for k in model.backbone.state_dict().keys())
    if cfg["pretrained"] and has_pretrained:
        model._freeze_backbone()

    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_dir=data_dir,
        image_size=config["data"]["image_size"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        lr=cfg.get("learning_rate", config["training"]["learning_rate"]),
        epochs=config["training"]["epochs"],
        patience=config["training"]["early_stopping_patience"],
        save_dir=save_dir,
        model_name="ResNet50",
    )

    results = trainer.train(
        unfreeze_epoch=cfg["freeze_backbone_epochs"],
        unfreeze_callback=model.unfreeze_backbone,
    )

    plot_training_curves(results, save_dir)
    plot_confusion_matrix(results, class_names, save_dir)

    return results


def train_vit(config: dict, device: torch.device, data_dir: str, save_dir: str) -> dict:
    """Entraine le modele ViT-B/16."""
    print("\n" + "=" * 60)
    print("MODEL 2: ViT-B/16 - Vision Transformer")
    print("=" * 60)

    cfg = config["models"]["vit"]
    num_classes = len(config["style_categories"])

    model = create_vit_model(
        num_classes=num_classes,
        model_name=cfg["model_name"],
        pretrained=cfg["pretrained"],
        dropout=cfg["dropout"],
        freeze_backbone=True,
    )

    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_dir=data_dir,
        image_size=config["data"]["image_size"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        lr=cfg.get("learning_rate", config["training"]["learning_rate"]),
        epochs=config["training"]["epochs"],
        patience=config["training"]["early_stopping_patience"],
        save_dir=save_dir,
        model_name="ViT-B16",
    )

    results = trainer.train(
        unfreeze_epoch=cfg["freeze_backbone_epochs"],
        unfreeze_callback=lambda: model.unfreeze_backbone(num_layers=4),
    )

    plot_training_curves(results, save_dir)
    plot_confusion_matrix(results, class_names, save_dir)

    return results


def evaluate_clip_zeroshot(config: dict, device: torch.device, data_dir: str, save_dir: str) -> dict:
    """Evalue CLIP en zero-shot (pas d'entrainement)."""
    print("\n" + "=" * 60)
    print("MODEL 3: CLIP Zero-Shot (pas d'entrainement)")
    print("=" * 60)

    cfg = config["models"]["clip_zeroshot"]
    style_categories = config["style_categories"]

    model = create_clip_zeroshot(
        model_name=cfg["model_name"],
        pretrained_dataset=cfg["pretrained_dataset"],
        style_categories=style_categories,
    ).to(device)

    _, _, test_loader, class_names = create_dataloaders(
        data_dir=data_dir,
        image_size=config["data"]["image_size"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    # Evaluer directement (pas d'entrainement)
    import numpy as np
    from torch.utils.data import DataLoader

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            similarity = model(images)
            probs = torch.softmax(similarity, dim=-1)
            _, predicted = similarity.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probs)

    test_metrics = compute_metrics(y_true, y_pred, y_proba, class_names)

    # Temps d'inference
    sample = next(iter(test_loader))[0][:1].to(device)
    inference_time = measure_inference_time(model, sample, device)

    results = {
        "model_name": "CLIP-ZeroShot",
        "parameters": count_parameters(model),
        "training_time_s": 0,  # Pas d'entrainement
        "best_val_accuracy": test_metrics["accuracy"],
        "test_metrics": test_metrics,
        "inference_time": inference_time,
        "history": {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []},
    }

    # Sauvegarder
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(save_path / "CLIP-ZeroShot_results.json", "w") as f:
        json.dump(json.loads(json.dumps(results, default=convert)), f, indent=2)

    print(f"\nResultats CLIP Zero-Shot:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {test_metrics['f1_macro']:.4f}")
    print(f"  Inference: {inference_time['mean_ms']:.2f}ms")

    return results


def train_clip_mlp(config: dict, device: torch.device, data_dir: str, save_dir: str) -> dict:
    """Entraine le modele CLIP + MLP multimodal."""
    print("\n" + "=" * 60)
    print("MODEL 4: CLIP + MLP Recommandation Multimodale")
    print("=" * 60)

    cfg = config["models"]["clip_mlp"]
    num_classes = len(config["style_categories"])

    model = create_clip_mlp_model(
        num_classes=num_classes,
        clip_model_name=cfg["model_name"],
        pretrained_dataset=cfg["pretrained_dataset"],
        mlp_hidden_dims=cfg["mlp_hidden_dims"],
        dropout=cfg["dropout"],
        freeze_clip=True,
    )

    # Utiliser le dataset avec texte pour CLIP+MLP
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_dir=data_dir,
        image_size=config["data"]["image_size"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        with_text=True,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        lr=cfg.get("learning_rate", config["training"]["learning_rate"]),
        epochs=config["training"]["epochs"],
        patience=config["training"]["early_stopping_patience"],
        save_dir=save_dir,
        model_name="CLIP+MLP",
    )

    results = trainer.train(
        unfreeze_epoch=cfg["freeze_clip_epochs"],
        unfreeze_callback=model.unfreeze_clip,
    )

    plot_training_curves(results, save_dir)
    plot_confusion_matrix(results, class_names, save_dir)

    # Demo: recommandation pour un evenement
    print("\n--- Demo Recommandation ---")
    model.eval()
    sample_image = next(iter(test_loader))[0][:1].to(device)

    for event in ["Entretien d'alternance", "Soiree de gala", "Journee sur le campus"]:
        result = model.recommend(sample_image, event, class_names, top_k=3)
        print(f"\nEvenement: {event}")
        for rec in result["recommendations"]:
            print(f"  #{rec['rank']} {rec['style']} ({rec['confidence']:.2%})")

    return results


def run_benchmark(save_dir: str):
    """Charge les resultats sauvegardes et genere le benchmark comparatif."""
    save_path = Path(save_dir)
    all_results = []

    for json_file in save_path.glob("*_results.json"):
        with open(json_file, "r") as f:
            results = json.load(f)
            all_results.append(results)

    if not all_results:
        print("Aucun resultat trouve. Lancez d'abord l'entrainement.")
        return

    print(f"\n{len(all_results)} modeles trouves pour le benchmark")

    plot_benchmark_comparison(all_results, save_dir)
    generate_recommendation_report(all_results, save_dir)


def main():
    parser = argparse.ArgumentParser(description="MakeAIPrep - Entrainement des modeles IA")
    parser.add_argument("--all", action="store_true", help="Entrainer tous les modeles")
    parser.add_argument("--model", choices=["resnet", "vit", "clip_zeroshot", "clip_mlp"],
                        help="Entrainer un modele specifique")
    parser.add_argument("--benchmark-only", action="store_true", help="Benchmark uniquement")
    parser.add_argument("--config", default="./models/config.yaml", help="Chemin config")
    parser.add_argument("--data-dir", default="./datasets", help="Dossier des donnees")
    parser.add_argument("--save-dir", default="./results", help="Dossier de sauvegarde")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    if args.benchmark_only:
        run_benchmark(args.save_dir)
        return

    all_results = []

    if args.all or args.model == "resnet":
        results = train_resnet(config, device, args.data_dir, args.save_dir)
        all_results.append(results)

    if args.all or args.model == "vit":
        results = train_vit(config, device, args.data_dir, args.save_dir)
        all_results.append(results)

    if args.all or args.model == "clip_zeroshot":
        results = evaluate_clip_zeroshot(config, device, args.data_dir, args.save_dir)
        all_results.append(results)

    if args.all or args.model == "clip_mlp":
        results = train_clip_mlp(config, device, args.data_dir, args.save_dir)
        all_results.append(results)

    # Benchmark si plusieurs modeles
    if len(all_results) > 1:
        plot_benchmark_comparison(all_results, args.save_dir)
        generate_recommendation_report(all_results, args.save_dir)
    elif args.all:
        run_benchmark(args.save_dir)

    print("\nTermine ! Resultats sauvegardes dans:", args.save_dir)
    print("Fichiers generes:")
    for f in Path(args.save_dir).glob("*"):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
