"""
MakeAIPrep - Labellisation semantique des visages avec CLIP zero-shot
======================================================================

Au lieu de KMeans (clusters anonymes), on utilise CLIP pour assigner a
chaque visage un label SEMANTIQUE reel (naturel, professionnel, glamour...).

Les labels sont alignes avec les vrais besoins du projet :
- Chaque classe = un style de maquillage/relooking
- Chaque classe est mappee a des contextes d'evenements
- Les modeles ResNet/ViT/CLIP+MLP peuvent ensuite apprendre dessus

Usage:
    python preprocessing/label_with_clip.py
    python preprocessing/label_with_clip.py --src "datasets/images_nettoyées_manuellement"
"""

import argparse
import os
import shutil
import stat
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


# 7 styles de relooking, alignes avec les attendus du projet
# (`moderne` retire car CLIP ne le detectait que sur 2/1507 images)
STYLE_PROMPTS = {
    "naturel":        "a portrait of a person with a natural fresh no-makeup look",
    "professionnel":  "a portrait of a person with a professional polished business look",
    "elegant":        "a portrait of a person with an elegant sophisticated evening look",
    "glamour":        "a portrait of a person with a glamorous bold makeup look",
    "minimaliste":    "a portrait of a person with a minimalist clean simple look",
    "boheme":         "a portrait of a person with a bohemian artistic free-spirited look",
    "doux":           "a portrait of a person with a soft romantic gentle makeup look",
}

STYLE_NAMES = list(STYLE_PROMPTS.keys())


def _rm_robust(path: Path, retries: int = 5):
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


def label_images(image_paths, device):
    import open_clip
    print("Chargement de CLIP ViT-B-32...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()

    # Encoder les prompts une seule fois
    prompts = list(STYLE_PROMPTS.values())
    with torch.no_grad():
        tokens = tokenizer(prompts).to(device)
        text_emb = model.encode_text(tokens).float()
        text_emb = F.normalize(text_emb, dim=-1)  # (8, 512)

    labels = []
    confidences = []
    with torch.no_grad():
        for p in tqdm(image_paths, desc="Labellisation CLIP"):
            img = Image.open(p).convert("RGB")
            x = preprocess(img).unsqueeze(0).to(device)
            img_emb = model.encode_image(x).float()
            img_emb = F.normalize(img_emb, dim=-1)
            sim = (img_emb @ text_emb.T).squeeze()  # (8,)
            probs = F.softmax(sim * 100, dim=-1)    # temperature CLIP standard
            idx = int(probs.argmax().item())
            labels.append(idx)
            confidences.append(float(probs[idx].item()))
    return np.array(labels), np.array(confidences)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/images_nettoyées_manuellement")
    ap.add_argument("--dst", default="datasets_faces")
    ap.add_argument("--limit", type=int, default=0,
                    help="Limiter a N images (test rapide). 0 = tout traiter.")
    ap.add_argument("--dry-run", action="store_true",
                    help="N'ecrit rien sur disque, affiche juste les labels.")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    images_dst = dst / "images"
    if not src.exists():
        raise SystemExit(f"Dossier source introuvable: {src.resolve()}")

    all_paths = sorted([p for p in src.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    print(f"Images disponibles: {len(all_paths)}")
    if not all_paths:
        raise SystemExit("Aucune image trouvee.")

    if args.limit > 0:
        # Echantillonnage espace pour avoir de la variete
        step = max(1, len(all_paths) // args.limit)
        image_paths = all_paths[::step][:args.limit]
        print(f"Mode test: {len(image_paths)} images echantillonnees")
    else:
        image_paths = all_paths

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    labels, confs = label_images(image_paths, device)

    # Mode dry-run : afficher le detail image par image et quitter
    if args.dry_run or args.limit > 0:
        print("\nResultat detaille (image -> label predit) :")
        for p, lab, c in zip(image_paths, labels, confs):
            print(f"  {p.name:<20} -> {STYLE_NAMES[lab]:<14} ({c*100:.1f}%)")
        if args.dry_run:
            print("\n[dry-run] aucun fichier ecrit. Relance sans --dry-run pour ecrire.")
            return

    # Recreer la structure
    _rm_robust(images_dst)
    for s in STYLE_NAMES:
        (images_dst / s).mkdir(parents=True, exist_ok=True)

    rows = []
    for p, lab, c in zip(image_paths, labels, confs):
        cls = STYLE_NAMES[lab]
        new_name = f"{cls}_{p.stem}{p.suffix.lower()}"
        target = images_dst / cls / new_name
        shutil.copy2(p, target)
        rows.append({
            "image_path": f"images/{cls}/{new_name}",
            "style": cls,
            "confidence": round(float(c), 4),
            "description": STYLE_PROMPTS[cls],
        })

    pd.DataFrame(rows).to_csv(dst / "metadata.csv", index=False)

    print("\nDistribution des labels semantiques:")
    for i, name in enumerate(STYLE_NAMES):
        n = int((labels == i).sum())
        bar = "#" * (n * 40 // max(1, len(labels)))
        avg_c = float(confs[labels == i].mean()) if n > 0 else 0.0
        print(f"  {name:<14} {n:4d}  conf_moy={avg_c:.2f}  {bar}")

    print(f"\nDataset pret dans: {dst.resolve()}")
    print(f"Total: {len(rows)} images / {len(STYLE_NAMES)} classes semantiques")
    print("\nProchaine etape:")
    print("  python train.py --all --data-dir ./datasets_faces")
    print("  python train_clip_mlp_fast.py --data-dir ./datasets_faces")


if __name__ == "__main__":
    main()
