"""
MakeAIPrep - Trainer generique pour entrainer et evaluer les modeles.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm

from .metrics import compute_metrics, measure_inference_time, count_parameters


class Trainer:
    """Trainer generique pour tous les modeles de classification."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        class_names: List[str],
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        epochs: int = 20,
        patience: int = 5,
        save_dir: str = "./results",
        model_name: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.save_dir = Path(save_dir)
        self.model_name = model_name or getattr(model, "model_name", "Unknown")

        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Optimiseur et scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)

        # Historique
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
        }
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

    def train_epoch(self) -> Dict[str, float]:
        """Entraine le modele sur une epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            # Supporter les datasets avec et sans texte
            if len(batch) == 3:
                images, texts, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
                # Pour CLIP+MLP
                if hasattr(self.model, 'encode_text'):
                    outputs = self.model(images, texts=texts)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                else:
                    outputs = self.model(images)
            else:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=loss.item(), acc=correct / total)

        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict:
        """Evalue le modele sur un DataLoader."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        for batch in loader:
            if len(batch) == 3:
                images, texts, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
                if hasattr(self.model, 'encode_text'):
                    outputs = self.model(images, texts=texts)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                else:
                    outputs = self.model(images)
            else:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

            loss = self.criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            probs = torch.softmax(outputs, dim=-1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_proba = np.array(all_probs)

        metrics = compute_metrics(y_true, y_pred, y_proba, self.class_names)
        metrics["loss"] = total_loss / len(y_true)

        return metrics

    def train(
        self,
        unfreeze_epoch: Optional[int] = None,
        unfreeze_callback: Optional[Callable] = None,
    ) -> Dict:
        """Boucle d'entrainement complete avec early stopping.

        Args:
            unfreeze_epoch: Epoch a laquelle degeler le backbone
            unfreeze_callback: Fonction appelee pour degeler (ex: model.unfreeze_backbone)
        """
        print(f"\n{'='*60}")
        print(f"Entrainement: {self.model_name}")
        print(f"{'='*60}")

        params = count_parameters(self.model)
        print(f"Parametres: {params['total_params_M']}M total, {params['trainable_params_M']}M entrainables")
        print(f"Epochs: {self.epochs}, LR: {self.optimizer.param_groups[0]['lr']}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(self.epochs):
            # Degeler le backbone a l'epoch specifiee
            if unfreeze_epoch and epoch == unfreeze_epoch and unfreeze_callback:
                print(f"\n>> Degelage du backbone a l'epoch {epoch}")
                unfreeze_callback()
                # Mettre a jour l'optimiseur
                self.optimizer = AdamW(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=self.optimizer.param_groups[0]["lr"] * 0.1,
                    weight_decay=0.01,
                )
                params = count_parameters(self.model)
                print(f">> Nouveaux params entrainables: {params['trainable_params_M']}M")

            # Train
            train_metrics = self.train_epoch()
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])

            # Validation
            val_metrics = self.evaluate(self.val_loader)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])

            self.scheduler.step()

            print(
                f"Epoch [{epoch+1}/{self.epochs}] "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
                f"F1: {val_metrics['f1_macro']:.4f}"
            )

            # Early stopping
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_metrics)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    print(f"\nEarly stopping a l'epoch {epoch+1}")
                    break

        training_time = time.time() - start_time

        # Charger le meilleur modele et evaluer sur test
        self._load_best_checkpoint()
        test_metrics = self.evaluate(self.test_loader)

        # Mesurer le temps d'inference
        sample = next(iter(self.test_loader))
        sample_input = sample[0][:1].to(self.device)
        inference_metrics = measure_inference_time(self.model, sample_input, self.device)

        # Resultats finaux
        results = {
            "model_name": self.model_name,
            "parameters": count_parameters(self.model),
            "training_time_s": round(training_time, 2),
            "best_val_accuracy": self.best_val_acc,
            "test_metrics": test_metrics,
            "inference_time": inference_metrics,
            "history": self.history,
        }

        # Sauvegarder les resultats
        self._save_results(results)

        print(f"\n{'='*60}")
        print(f"RESULTATS FINAUX - {self.model_name}")
        print(f"{'='*60}")
        print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")
        print(f"Top-3 Accuracy: {test_metrics.get('top3_accuracy', 'N/A')}")
        print(f"Inference:      {inference_metrics['mean_ms']:.2f}ms")
        print(f"Training time:  {training_time:.1f}s")
        print(f"{'='*60}\n")

        return results

    def _save_checkpoint(self, epoch: int, metrics: Dict):
        """Sauvegarde le meilleur modele."""
        path = self.save_dir / f"{self.model_name}_best.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "metrics": {k: v for k, v in metrics.items() if k != "classification_report"},
        }, path)

    def _load_best_checkpoint(self):
        """Charge le meilleur modele."""
        path = self.save_dir / f"{self.model_name}_best.pth"
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])

    def _save_results(self, results: Dict):
        """Sauvegarde les resultats en JSON."""
        # Convertir les objets non-serialisables
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        path = self.save_dir / f"{self.model_name}_results.json"
        serializable = json.loads(json.dumps(results, default=convert))
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
