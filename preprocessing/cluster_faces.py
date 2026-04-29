"""
MakeAIPrep - Pseudo-labellisation des visages par clustering (v2)
==================================================================

Pipeline ameliore:
1. Extraction de features avec CLIP ViT-B/32 (concepts semantiques riches,
   plus pertinents pour le style/look qu'un ResNet ImageNet).
2. Normalisation L2 (distance cosinus implicite avec KMeans euclidien).
3. Reduction PCA (debruite + accelere).
4. KMeans avec selection du meilleur K via score silhouette (si --auto-k).
5. Copie des images dans datasets_faces/images/cluster_X/ + metadata.csv.

Usage:
    python preprocessing/cluster_faces.py                  # K=8 fixe
    python preprocessing/cluster_faces.py --auto-k         # cherche K optimal entre 5 et 10
    python preprocessing/cluster_faces.py --k 8 --pca 64
"""

import argparse
import os
import shutil
import stat
import time
from pathlib import Path


def _rm_robust(path: Path, retries: int = 5):
    """Supprime un dossier en gerant les verrous Windows / OneDrive."""
    def on_error(func, p, exc_info):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass
    for i in range(retries):
        try:
            if path.exists():
                shutil.rmtree(path, onerror=on_error)
            return
        except PermissionError:
            time.sleep(1.0 + i)
    # Dernier recours: vider fichier par fichier
    if path.exists():
        for p in sorted(path.rglob("*"), reverse=True):
            try:
                if p.is_file():
                    os.chmod(p, stat.S_IWRITE)
                    p.unlink()
                elif p.is_dir():
                    p.rmdir()
            except Exception:
                pass
        try:
            path.rmdir()
        except Exception:
            pass

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tqdm import tqdm


def load_clip(device):
    """Charge CLIP ViT-B/32 (open_clip)."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.to(device).eval()
    return model, preprocess


def extract_clip_features(image_paths, device):
    """Extrait les features image CLIP (512-d) pour chaque image."""
    model, preprocess = load_clip(device)
    feats = []
    with torch.no_grad():
        for p in tqdm(image_paths, desc="Extraction CLIP"):
            img = Image.open(p).convert("RGB")
            x = preprocess(img).unsqueeze(0).to(device)
            f = model.encode_image(x).cpu().numpy().squeeze()
            feats.append(f)
    return np.stack(feats).astype(np.float32)


def l2_normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def select_best_k(feats, k_range=(5, 11), seed=42):
    """Selectionne le meilleur K via score silhouette."""
    print("\nRecherche du meilleur K (silhouette)...")
    best_k, best_score = None, -1.0
    for k in range(*k_range):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(feats)
        score = silhouette_score(feats, labels, sample_size=min(1000, len(feats)))
        print(f"  K={k}: silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score = k, score
    print(f"  -> meilleur K = {best_k} (silhouette={best_score:.4f})")
    return best_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="datasets/images_nettoyées_manuellement")
    parser.add_argument("--dst", default="datasets_faces")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--auto-k", action="store_true",
                        help="Cherche K optimal entre 5 et 10 via silhouette")
    parser.add_argument("--pca", type=int, default=64,
                        help="Dimension PCA (0 pour desactiver)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    images_dst = dst / "images"

    if not src.exists():
        raise SystemExit(f"Dossier source introuvable: {src.resolve()}")

    image_paths = sorted([p for p in src.iterdir()
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    print(f"Images trouvees: {len(image_paths)}")
    if not image_paths:
        raise SystemExit("Aucune image trouvee.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Features CLIP
    feats = extract_clip_features(image_paths, device)
    print(f"Features brutes: {feats.shape}")

    # 2. L2 normalize -> distances cosinus stables
    feats = l2_normalize(feats)

    # 3. PCA pour debruiter
    if args.pca > 0 and args.pca < feats.shape[1]:
        pca = PCA(n_components=args.pca, random_state=args.seed)
        feats = pca.fit_transform(feats)
        print(f"PCA -> {feats.shape}, variance expliquee: "
              f"{pca.explained_variance_ratio_.sum():.3f}")
        feats = l2_normalize(feats)  # re-norm apres PCA

    # 4. Choix de K
    k = select_best_k(feats, seed=args.seed) if args.auto_k else args.k

    # 5. KMeans final
    print(f"\nKMeans final K={k}...")
    kmeans = KMeans(n_clusters=k, random_state=args.seed, n_init=20)
    labels = kmeans.fit_predict(feats)
    sil = silhouette_score(feats, labels, sample_size=min(1000, len(feats)))
    print(f"Silhouette finale: {sil:.4f}")

    # 6. (Re)creer la structure et copier les images
    _rm_robust(images_dst)
    for c in range(k):
        (images_dst / f"cluster_{c}").mkdir(parents=True, exist_ok=True)

    rows = []
    for p, c in zip(image_paths, labels):
        cls = f"cluster_{int(c)}"
        new_name = f"{cls}_{p.stem}{p.suffix.lower()}"
        target = images_dst / cls / new_name
        shutil.copy2(p, target)
        rows.append({
            "image_path": f"images/{cls}/{new_name}",
            "style": cls,
            "event_type": "general",
            "description": f"Visage groupe {cls}",
        })

    df = pd.DataFrame(rows)
    df.to_csv(dst / "metadata.csv", index=False)

    print("\nDistribution des clusters:")
    for c in range(k):
        n = int((labels == c).sum())
        bar = "#" * (n * 40 // max(1, len(labels)))
        print(f"  cluster_{c}: {n:4d}  {bar}")
    print(f"\nDataset pret dans: {dst.resolve()}")
    print(f"Total: {len(rows)} images / {k} classes")
    print("\nProchaine etape:")
    print("  python train.py --all --data-dir ./datasets_faces")


if __name__ == "__main__":
    main()
