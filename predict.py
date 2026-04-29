"""
MakeAIPrep - Prediction & recommandation pour une vraie image
==============================================================

Charge un modele entraine et donne une recommandation de style + look
pour une photo de visage et un contexte d'evenement.

Usage:
    # Avec CLIP+MLP (recommande, supporte le contexte texte)
    python predict.py --image chemin/vers/photo.jpg --event "entretien d'embauche"

    # Avec ResNet50 (image seule)
    python predict.py --image chemin/vers/photo.jpg --model resnet

    # Top-K
    python predict.py --image photo.jpg --top-k 5
"""

import argparse
import os
import ssl
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Conseils relooking par style semantique (alignes avec label_with_clip.py)
RELOOKING_TIPS = {
    "naturel":       "Teint frais et lumineux, mascara leger, gloss transparent, sourcils brosses. "
                     "Coiffure naturelle sans exces de produit.",
    "professionnel": "Peau matifiee, eyeliner discret, levres nude mate, sourcils structures. "
                     "Coiffure tiree ou chignon bas, look soigne et credible.",
    "elegant":       "Smoky eye doux, teint sophistique, levres bordeaux ou rose profond. "
                     "Coiffure ondulee ou chignon haut, accessoires raffines.",
    "glamour":       "Eyeliner marque, faux cils, contouring, levres rouges ou nude glossy. "
                     "Wavy hair ou volume, look impactant pour evenements.",
    "minimaliste":   "Peau nue lumineuse, mascara seul, baume teinte. "
                     "Coiffure simple, zero artifice : less is more.",
    "boheme":        "Teint dore, blush peche, couronne ou tresses, accessoires naturels. "
                     "Maquillage chaud et terreux, esprit libre.",
    "doux":          "Blush rose, eyeshadow pastel, levres rose poudre, mascara brun. "
                     "Coiffure souple et romantique, look feminin et delicat.",
    "moderne":       "Eyeliner graphique ou color block, teint glowy, levres mates tendance. "
                     "Coupe structuree ou wet look, esthetique contemporaine.",
}

# Description courte du contexte evenementiel
EVENT_DESCRIPTIONS = {
    "entretien":     "Sobriete, credibilite, peu d'artifice. Le maquillage doit rassurer pas distraire.",
    "alternance":    "Sobriete et serieux. Look professionnel adapte au monde de l'entreprise.",
    "presentation":  "Look net qui inspire confiance. Eviter ce qui peut detourner l'attention.",
    "reunion":       "Professionnel et soigne, sans tomber dans le formel excessif.",
    "stage":         "Premier jour : naturel et soigne, ne pas en faire trop.",
    "networking":    "Elegant et memorable, un cran au-dessus du quotidien.",
    "gala":          "Glamour assume, on ose les couleurs profondes et les details.",
    "soiree":        "Plus libre, on peut accentuer yeux ou levres selon le code.",
    "campus":        "Naturel et pratique, juste une mise en valeur legere.",
}


def get_classes(data_dir: Path):
    images_dir = data_dir / "images"
    return sorted([d.name for d in images_dir.iterdir() if d.is_dir()])


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_resnet(num_classes, ckpt_path, device):
    from models.resnet_classifier import create_resnet_model
    model = create_resnet_model(num_classes=num_classes, pretrained=False, dropout=0.3)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def predict_resnet(model, image_path, device):
    from torchvision import transforms
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        return F.softmax(out, dim=-1).squeeze().cpu().numpy()


def load_clip_mlp(num_classes, ckpt_path, cfg, device):
    """Charge FusionMLP + CLIP fige (necessite CLIP en runtime pour encoder)."""
    from models.clip_mlp_recommender import FusionMLP
    import open_clip
    mlp = FusionMLP(
        input_dim=1024,
        hidden_dims=cfg["models"]["clip_mlp"]["mlp_hidden_dims"],
        num_classes=num_classes,
        dropout=cfg["models"]["clip_mlp"]["dropout"],
    )
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state.get("model_state_dict", state) if isinstance(state, dict) else state
    # Cas 1: checkpoint Trainer complet (CLIPMLPRecommender) -> extraire fusion_mlp.*
    fusion_keys = {k.replace("fusion_mlp.", ""): v
                   for k, v in sd.items() if k.startswith("fusion_mlp.")}
    if fusion_keys:
        mlp.load_state_dict(fusion_keys)
    else:
        # Cas 2: checkpoint train_clip_mlp_fast.py (MLP standalone)
        mlp.load_state_dict(sd)
    mlp = mlp.to(device).eval()

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    clip_model = clip_model.to(device).eval()
    return mlp, clip_model, preprocess, tokenizer


def predict_clip_mlp(mlp, clip_model, preprocess, tokenizer,
                     image_path, event_text, classes, device):
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_emb = F.normalize(clip_model.encode_image(x).float(), dim=-1)
        # Texte: le contexte de l'evenement
        tokens = tokenizer([event_text]).to(device)
        txt_emb = F.normalize(clip_model.encode_text(tokens).float(), dim=-1)
        feat = torch.cat([img_emb, txt_emb], dim=-1)
        logits, _ = mlp(feat)
        return F.softmax(logits, dim=-1).squeeze().cpu().numpy()


def find_checkpoint(model_choice, save_dir: Path):
    """Trouve le .pth correspondant au modele demande."""
    if model_choice == "clip_mlp":
        for name in ["CLIP+MLP_best.pth", "CLIP+MLP_fast.pth"]:
            p = save_dir / name
            if p.exists():
                return p
    elif model_choice == "resnet":
        p = save_dir / "ResNet50_best.pth"
        if p.exists():
            return p
    elif model_choice == "vit":
        p = save_dir / "ViT-B16_best.pth"
        if p.exists():
            return p
    return None


def event_advice(event_text):
    et = event_text.lower()
    for k, v in EVENT_DESCRIPTIONS.items():
        if k in et:
            return v
    return "Adapter le look au contexte de l'evenement."


def expected_styles(event_text, cfg):
    """Retourne la liste des styles attendus pour cet evenement (via config.yaml)."""
    et = event_text.lower().replace(" ", "_").replace("'", "")
    mapping = cfg.get("event_to_style", {})
    # Match exact d'abord
    for key, styles in mapping.items():
        if key in et or et in key:
            return styles
    # Match par mot-cle
    for key, styles in mapping.items():
        for word in key.split("_"):
            if word in et and len(word) > 3:
                return styles
    return []


def build_recommendation(probs, classes, expected, top_k=3):
    """Croise la prediction du modele avec les styles attendus pour l'evenement."""
    order = probs.argsort()[::-1]
    top = [(classes[i], float(probs[i])) for i in order[:top_k]]
    best_pred = classes[order[0]]

    if not expected:
        verdict = "Pas de mapping evenement -> on suit la prediction du modele."
        reco = best_pred
    elif best_pred in expected:
        verdict = f"OK : le style predit ({best_pred}) correspond aux attendus de l'evenement."
        reco = best_pred
    else:
        # Le modele predit un style qui ne colle pas a l'evenement.
        # On cherche le style attendu avec la plus forte proba.
        exp_probs = [(s, float(probs[classes.index(s)]))
                     for s in expected if s in classes]
        exp_probs.sort(key=lambda x: -x[1])
        if exp_probs:
            reco = exp_probs[0][0]
            verdict = (f"ATTENTION : style detecte = {best_pred}, mais l'evenement "
                       f"attend plutot {expected[0]}. Reco ajustee -> {reco}.")
        else:
            reco = best_pred
            verdict = "Reco basee sur la prediction brute."
    return top, reco, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, nargs="+",
                    help="Une ou plusieurs images (ex: --image a.jpg b.jpg c.jpg)")
    ap.add_argument("--event", default="entretien d'embauche",
                    help="Contexte: ex 'entretien', 'soiree gala', 'journee campus'")
    ap.add_argument("--model", choices=["clip_mlp", "resnet", "vit", "clip_zeroshot", "all"],
                    default="clip_mlp")
    ap.add_argument("--data-dir", default="./datasets_faces")
    ap.add_argument("--save-dir", default="./results")
    ap.add_argument("--config", default="./models/config.yaml")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    img_paths = [Path(p) for p in args.image]
    for p in img_paths:
        if not p.exists():
            raise SystemExit(f"Image introuvable: {p}")

    data_dir = Path(args.data_dir)
    classes = get_classes(data_dir)
    cfg = yaml.safe_load(open(args.config))
    device = get_device()

    def build_predictor(model_name):
        ckpt = find_checkpoint(model_name, Path(args.save_dir))
        if ckpt is None and model_name != "clip_zeroshot":
            return None, None
        if model_name == "clip_mlp":
            mlp, clip_m, prep, tok = load_clip_mlp(len(classes), ckpt, cfg, device)
            return ckpt.name, lambda p: predict_clip_mlp(
                mlp, clip_m, prep, tok, p, args.event, classes, device)
        if model_name == "resnet":
            m = load_resnet(len(classes), ckpt, device)
            return ckpt.name, lambda p: predict_resnet(m, p, device)
        if model_name == "vit":
            from models.vit_classifier import create_vit_model
            m = create_vit_model(num_classes=len(classes),
                                  model_name=cfg["models"]["vit"]["model_name"],
                                  pretrained=False, dropout=0.1)
            ck = torch.load(ckpt, map_location=device, weights_only=False)
            m.load_state_dict(ck["model_state_dict"])
            m = m.to(device).eval()
            return ckpt.name, lambda p: predict_resnet(m, p, device)
        if model_name == "clip_zeroshot":
            from models.clip_zeroshot import create_clip_zeroshot
            m = create_clip_zeroshot(
                model_name=cfg["models"]["clip_zeroshot"]["model_name"],
                pretrained_dataset=cfg["models"]["clip_zeroshot"]["pretrained_dataset"],
                style_categories=classes,
            ).to(device).eval()
            from torchvision import transforms
            tfm = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
            def _pred(p):
                img = Image.open(p).convert("RGB")
                x = tfm(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    sim = m(x)
                    return F.softmax(sim, dim=-1).squeeze().cpu().numpy()
            return "(zero-shot, pas de checkpoint)", _pred
        return None, None

    if args.model == "all":
        model_list = ["resnet", "vit", "clip_zeroshot", "clip_mlp"]
    else:
        model_list = [args.model]

    print(f"Event : {args.event}")
    print(f"Images: {len(img_paths)}")
    print(f"Modeles testes: {', '.join(model_list)}\n")

    predictors = {}
    for mn in model_list:
        ckname, fn = build_predictor(mn)
        if fn is None:
            print(f"[skip] {mn}: pas de checkpoint trouve")
        else:
            predictors[mn] = (ckname, fn)
            print(f"[ok]   {mn}: {ckname}")
    print()

    expected = expected_styles(args.event, cfg)

    for i, img_path in enumerate(img_paths, 1):
        print("=" * 60)
        print(f"  IMAGE {i}/{len(img_paths)} - {img_path.name}")
        print("=" * 60)
        print(f"Contexte    : {args.event}")
        print(f"Conseil     : {event_advice(args.event)}")
        if expected:
            print(f"Styles ideaux: {', '.join(expected)}")
        print()

        for mn, (ckname, fn) in predictors.items():
            probs = fn(img_path)
            top, reco, verdict = build_recommendation(probs, classes, expected, args.top_k)
            print(f"--- {mn.upper()} ---")
            for rank, (cls, p) in enumerate(top, 1):
                bar = "#" * int(p * 100 / 3)
                print(f"  #{rank}  {cls:<14} {p*100:5.1f}%  {bar}")
            print(f"  Verdict : {verdict}")
            print(f"  STYLE RECOMMANDE : {reco.upper()}")
            print(f"  Conseil relooking : {RELOOKING_TIPS.get(reco, '')}\n")


if __name__ == "__main__":
    main()
