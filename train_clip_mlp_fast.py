"""
MakeAIPrep - Entrainement rapide CLIP+MLP avec embeddings precalcules
======================================================================

CLIP est gele dans clip_mlp_recommender.py, donc ses sorties sont identiques
a chaque epoch. Au lieu de refaire passer les images dans CLIP a chaque batch
(tres lent), on precalcule les embeddings UNE seule fois et on entraine
uniquement le MLP par-dessus.

Gain typique: ~50-100x plus rapide pour le meme resultat.

Usage:
    python train_clip_mlp_fast.py --data-dir ./datasets/final
    python train_clip_mlp_fast.py --data-dir ./datasets/final --epochs 30 --no-cache
"""

import argparse
import json
import os
import ssl
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from models.clip_mlp_recommender import FusionMLP
from models.metrics import compute_metrics, count_parameters, measure_inference_time
from models.visualization import plot_training_curves, plot_confusion_matrix


def collect_samples(data_dir: Path):
    """Liste (image_path, label, class_name) trouve sous data_dir/images/."""
    images_dir = data_dir / "images"
    classes = sorted([d.name for d in images_dir.iterdir() if d.is_dir()])
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    samples = []
    for c in classes:
        for p in (images_dir / c).iterdir():
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                samples.append((p, cls_to_idx[c]))
    return samples, classes


def precompute_embeddings(samples, classes, device, cache_path: Path):
    """Calcule les embeddings image+text CLIP une fois et les cache sur disque."""
    if cache_path.exists():
        print(f"Cache trouve: {cache_path}")
        data = torch.load(cache_path, weights_only=False)
        return data["img_emb"], data["txt_emb"], data["labels"]

    import open_clip
    print("Chargement de CLIP ViT-B-32...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()

    # Embeddings textuels: une description par classe (mode label-only)
    class_texts = [f"a portrait of a person with a {c.replace('_', ' ')} look"
                   for c in classes]
    with torch.no_grad():
        tokens = tokenizer(class_texts).to(device)
        class_txt = model.encode_text(tokens).float()
        class_txt = F.normalize(class_txt, dim=-1).cpu()

    img_embs = torch.zeros(len(samples), 512)
    labels = torch.zeros(len(samples), dtype=torch.long)
    txt_embs = torch.zeros(len(samples), 512)

    with torch.no_grad():
        for i, (path, lab) in enumerate(tqdm(samples, desc="Embeddings CLIP")):
            img = Image.open(path).convert("RGB")
            x = preprocess(img).unsqueeze(0).to(device)
            f = model.encode_image(x).float()
            f = F.normalize(f, dim=-1).cpu().squeeze()
            img_embs[i] = f
            labels[i] = lab
            txt_embs[i] = class_txt[lab]  # texte conditionne sur la classe

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"img_emb": img_embs, "txt_emb": txt_embs,
                "labels": labels, "classes": classes}, cache_path)
    print(f"Cache sauvegarde: {cache_path}")
    return img_embs, txt_embs, labels


def make_loaders(img, txt, lab, batch_size, seed=42):
    full = TensorDataset(img, txt, lab)
    n = len(full)
    n_tr = int(0.8 * n)
    n_va = int(0.1 * n)
    n_te = n - n_tr - n_va
    tr, va, te = random_split(full, [n_tr, n_va, n_te],
                               generator=torch.Generator().manual_seed(seed))
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True),
        DataLoader(va, batch_size=batch_size),
        DataLoader(te, batch_size=batch_size),
    )


def evaluate(mlp, loader, device, criterion):
    mlp.eval()
    total_loss, all_p, all_y, all_pr = 0.0, [], [], []
    with torch.no_grad():
        for img, txt, y in loader:
            img, txt, y = img.to(device), txt.to(device), y.to(device)
            x = torch.cat([img, txt], dim=-1)
            logits, _ = mlp(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            p = logits.softmax(-1)
            all_pr.append(p.cpu().numpy())
            all_p.append(p.argmax(-1).cpu().numpy())
            all_y.append(y.cpu().numpy())
    return (np.concatenate(all_y), np.concatenate(all_p),
            np.concatenate(all_pr), total_loss / len(loader.dataset))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./datasets/final")
    ap.add_argument("--save-dir", default="./results")
    ap.add_argument("--config", default="./models/config.yaml")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / "_clip_cache.pt"
    if args.no_cache and cache_path.exists():
        cache_path.unlink()

    cfg = yaml.safe_load(open(args.config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    samples, classes = collect_samples(data_dir)
    print(f"{len(samples)} images / {len(classes)} classes")

    img, txt, lab = precompute_embeddings(samples, classes, device, cache_path)

    train_loader, val_loader, test_loader = make_loaders(
        img, txt, lab, args.batch_size
    )

    mlp = FusionMLP(
        input_dim=1024,
        hidden_dims=cfg["models"]["clip_mlp"]["mlp_hidden_dims"],
        num_classes=len(classes),
        dropout=cfg["models"]["clip_mlp"]["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc, best_state, patience, no_improve = 0.0, None, 5, 0
    t0 = time.time()

    for epoch in range(args.epochs):
        mlp.train()
        loss_sum, n_ok, n_tot = 0.0, 0, 0
        for img_b, txt_b, y in train_loader:
            img_b, txt_b, y = img_b.to(device), txt_b.to(device), y.to(device)
            x = torch.cat([img_b, txt_b], dim=-1)
            logits, _ = mlp(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
            n_ok += (logits.argmax(-1) == y).sum().item()
            n_tot += y.size(0)
        scheduler.step()
        tr_loss, tr_acc = loss_sum / n_tot, n_ok / n_tot

        y_v, p_v, pr_v, va_loss = evaluate(mlp, val_loader, device, criterion)
        va_acc = (y_v == p_v).mean()
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss);   history["val_acc"].append(va_acc)
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train {tr_loss:.4f}/{tr_acc:.4f} | Val {va_loss:.4f}/{va_acc:.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.clone() for k, v in mlp.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping epoch {epoch+1}")
                break

    if best_state:
        mlp.load_state_dict(best_state)
    train_time = time.time() - t0

    # Test
    y_t, p_t, pr_t, _ = evaluate(mlp, test_loader, device, criterion)
    test_metrics = compute_metrics(y_t, p_t, pr_t, classes)

    # Inference time (juste sur le MLP, sans CLIP)
    sample = next(iter(test_loader))
    sample_x = torch.cat([sample[0][:1], sample[1][:1]], dim=-1).to(device)
    inf = measure_inference_time(mlp, sample_x, device)

    results = {
        "model_name": "CLIP+MLP",
        "parameters": count_parameters(mlp),
        "training_time_s": round(train_time, 2),
        "best_val_accuracy": float(best_acc),
        "test_metrics": test_metrics,
        "inference_time": inf,
        "history": history,
    }

    def conv(o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    # Sauvegarder le MLP entraine (pour predict.py)
    ckpt_path = save_dir / "CLIP+MLP_fast.pth"
    torch.save({"model_state_dict": mlp.state_dict(), "classes": classes}, ckpt_path)
    print(f"Checkpoint MLP: {ckpt_path}")

    out = save_dir / "CLIP+MLP_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(json.loads(json.dumps(results, default=conv)), f,
                  indent=2, ensure_ascii=False)

    plot_training_curves(results, save_dir)
    plot_confusion_matrix({**results, "test_metrics": test_metrics,
                           "y_true": y_t, "y_pred": p_t}, classes, save_dir)

    print(f"\n=== CLIP+MLP (fast) ===")
    print(f"Test acc: {test_metrics['accuracy']:.4f}  F1: {test_metrics['f1_macro']:.4f}")
    print(f"Train time: {train_time:.1f}s  (vs ~1400s avant)")
    print(f"Sauvegarde: {out}")


if __name__ == "__main__":
    main()
