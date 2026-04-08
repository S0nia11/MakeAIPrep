"""
MakeAIPrep - Metriques d'evaluation pour comparer les modeles.
"""

import time
from typing import Dict, List, Tuple

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
) -> Dict:
    """Calcule toutes les metriques de classification."""

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    # Top-3 accuracy
    if y_proba is not None and y_proba.shape[1] >= 3:
        top3_preds = np.argsort(y_proba, axis=1)[:, -3:]
        top3_correct = sum(1 for i, t in enumerate(y_true) if t in top3_preds[i])
        metrics["top3_accuracy"] = top3_correct / len(y_true)

    # Rapport par classe
    metrics["classification_report"] = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    # Matrice de confusion
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    return metrics


def measure_inference_time(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    device: torch.device,
    n_runs: int = 100,
) -> Dict[str, float]:
    """Mesure le temps d'inference moyen d'un modele."""
    model.eval()
    sample_input = sample_input.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(sample_input)

    # Mesure
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            model(sample_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "median_ms": np.median(times) * 1000,
        "min_ms": np.min(times) * 1000,
    }


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Compte les parametres d'un modele."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
        "total_params_M": round(total / 1e6, 2),
        "trainable_params_M": round(trainable / 1e6, 2),
    }
