"""
MakeAIPrep - Visualisation des resultats du benchmark.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curves(results: Dict, save_path: str = "./results"):
    """Trace les courbes loss et accuracy pour un modele."""
    save_path = Path(save_path)
    name = results["model_name"]
    history = results["history"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-", label="Train")
    ax1.plot(epochs, history["val_loss"], "r-", label="Validation")
    ax1.set_title(f"{name} - Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "b-", label="Train")
    ax2.plot(epochs, history["val_acc"], "r-", label="Validation")
    ax2.set_title(f"{name} - Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / f"{name}_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(results: Dict, class_names: List[str], save_path: str = "./results"):
    """Trace la matrice de confusion."""
    save_path = Path(save_path)
    name = results["model_name"]
    cm = np.array(results["test_metrics"]["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_title(f"{name} - Matrice de Confusion")
    ax.set_xlabel("Predit")
    ax.set_ylabel("Reel")
    plt.tight_layout()
    plt.savefig(save_path / f"{name}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_benchmark_comparison(all_results: List[Dict], save_path: str = "./results"):
    """Compare tous les modeles cote a cote.

    Genere:
    1. Bar chart des metriques principales
    2. Radar chart multi-criteres
    3. Tableau recapitulatif
    """
    save_path = Path(save_path)
    model_names = [r["model_name"] for r in all_results]

    # --- 1. Bar Chart des metriques ---
    metrics_to_compare = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ["#C1FF72", "#5CE1E6", "#6E6E6E", "#000000"]

    for idx, metric in enumerate(metrics_to_compare):
        ax = axes[idx // 2][idx % 2]
        values = [r["test_metrics"].get(metric, 0) for r in all_results]
        bars = ax.bar(model_names, values, color=colors[:len(model_names)], edgecolor="black")
        ax.set_title(metric.replace("_", " ").title(), fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

        # Annoter les valeurs
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")

    plt.suptitle("Comparaison des Modeles - MakeAIPrep", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path / "benchmark_comparison_bars.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- 2. Radar Chart ---
    criteria = ["Accuracy", "F1 Score", "Top-3 Acc", "Vitesse", "Legerete"]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(criteria), endpoint=False).tolist()
    angles += angles[:1]

    for i, r in enumerate(all_results):
        # Normaliser les valeurs entre 0 et 1
        acc = r["test_metrics"].get("accuracy", 0)
        f1 = r["test_metrics"].get("f1_macro", 0)
        top3 = r["test_metrics"].get("top3_accuracy", 0)

        # Vitesse: inverser (plus rapide = mieux)
        inf_time = r["inference_time"]["mean_ms"]
        max_time = max(rr["inference_time"]["mean_ms"] for rr in all_results)
        speed = 1 - (inf_time / max_time) if max_time > 0 else 1

        # Legerete: inverser (moins de params = mieux)
        params = r["parameters"]["total_params_M"]
        max_params = max(rr["parameters"]["total_params_M"] for rr in all_results)
        lightness = 1 - (params / max_params) if max_params > 0 else 1

        values = [acc, f1, top3, speed, lightness]
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=2, label=r["model_name"], color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(criteria, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title("Comparaison Multi-Criteres", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(save_path / "benchmark_radar_chart.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- 3. Tableau recapitulatif ---
    print("\n" + "=" * 90)
    print("TABLEAU RECAPITULATIF - BENCHMARK MakeAIPrep")
    print("=" * 90)
    header = f"{'Modele':<16} {'Acc':>7} {'F1':>7} {'Top3':>7} {'Inf(ms)':>8} {'Params(M)':>10} {'Train(s)':>9}"
    print(header)
    print("-" * 90)

    for r in all_results:
        tm = r["test_metrics"]
        print(
            f"{r['model_name']:<16} "
            f"{tm.get('accuracy', 0):>7.4f} "
            f"{tm.get('f1_macro', 0):>7.4f} "
            f"{tm.get('top3_accuracy', 0):>7.4f} "
            f"{r['inference_time']['mean_ms']:>8.2f} "
            f"{r['parameters']['total_params_M']:>10.2f} "
            f"{r.get('training_time_s', 0):>9.1f}"
        )
    print("=" * 90)

    # Sauvegarder le tableau en texte
    with open(save_path / "benchmark_summary.txt", "w") as f:
        f.write(header + "\n")
        f.write("-" * 90 + "\n")
        for r in all_results:
            tm = r["test_metrics"]
            f.write(
                f"{r['model_name']:<16} "
                f"{tm.get('accuracy', 0):>7.4f} "
                f"{tm.get('f1_macro', 0):>7.4f} "
                f"{tm.get('top3_accuracy', 0):>7.4f} "
                f"{r['inference_time']['mean_ms']:>8.2f} "
                f"{r['parameters']['total_params_M']:>10.2f} "
                f"{r.get('training_time_s', 0):>9.1f}\n"
            )


def generate_recommendation_report(all_results: List[Dict], save_path: str = "./results"):
    """Genere un rapport de recommandation: quel(s) modele(s) garder et pourquoi."""
    save_path = Path(save_path)

    # Scoring multi-criteres
    scores = {}
    for r in all_results:
        name = r["model_name"]
        tm = r["test_metrics"]

        # Ponderation selon les priorites MakeAIPrep
        score = (
            tm.get("accuracy", 0) * 0.25 +          # Precision globale
            tm.get("f1_macro", 0) * 0.25 +           # Equilibre entre classes
            tm.get("top3_accuracy", 0) * 0.20 +      # Recommandations pertinentes
            (1 - r["inference_time"]["mean_ms"] / 100) * 0.15 +  # Rapidite
            (1 - r["parameters"]["total_params_M"] / 200) * 0.15  # Deploiement mobile
        )
        scores[name] = score

    # Trier par score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    report = []
    report.append("=" * 70)
    report.append("RAPPORT DE RECOMMANDATION - MakeAIPrep")
    report.append("Quel(s) modele(s) garder et pourquoi ?")
    report.append("=" * 70)
    report.append("")

    for rank, (name, score) in enumerate(ranked, 1):
        r = next(rr for rr in all_results if rr["model_name"] == name)
        tm = r["test_metrics"]

        report.append(f"#{rank} {name} (Score: {score:.4f})")
        report.append(f"   Accuracy: {tm.get('accuracy', 0):.4f}")
        report.append(f"   F1 Score: {tm.get('f1_macro', 0):.4f}")
        report.append(f"   Inference: {r['inference_time']['mean_ms']:.2f}ms")
        report.append(f"   Params: {r['parameters']['total_params_M']}M")
        report.append("")

    report.append("")
    report.append("RECOMMANDATION FINALE:")
    report.append("-" * 40)

    best = ranked[0][0]
    if "CLIP+MLP" in best or "CLIP" in best:
        report.append(f">> GARDER: {best}")
        report.append("   Raison: Meilleur score global, capacite multimodale")
        report.append("   (combine image + contexte textuel de l'evenement)")
        report.append("")
        report.append(">> GARDER AUSSI: CLIP-ZeroShot (en complement)")
        report.append("   Raison: Pas besoin d'entrainement, ideal pour le prototypage")
        report.append("   et comme fallback quand les donnees sont limitees")
    else:
        report.append(f">> GARDER: {best}")
        report.append("   Raison: Meilleur score global sur les criteres combines")

    report_text = "\n".join(report)
    print(report_text)

    with open(save_path / "recommendation_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
