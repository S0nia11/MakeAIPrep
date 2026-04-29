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

# Conseils maquillage par style semantique (alignes avec label_with_clip.py)
MAKEUP_TIPS = {
    "naturel":       "Teint frais et lumineux, mascara leger, gloss transparent, sourcils brosses.",
    "professionnel": "Peau matifiee, eyeliner discret, levres nude mate, sourcils structures.",
    "elegant":       "Smoky eye doux, teint sophistique, levres bordeaux ou rose profond.",
    "glamour":       "Eyeliner marque, faux cils, contouring, levres rouges ou nude glossy.",
    "minimaliste":   "Peau nue lumineuse, mascara seul, baume teinte. Less is more.",
    "doux":          "Blush rose, eyeshadow pastel, levres rose poudre, mascara brun.",
    "moderne":       "Eyeliner graphique ou color block, teint glowy, levres mates tendance.",
    "gothique":      "Teint pale, eyeliner noir epais, smoky eye charbon, levres bordeaux ou prune fonce, "
                     "sourcils marques.",
    "vintage":       "Eyeliner pin-up en aile, levres rouge mat profond, peau matifiee, "
                     "eyeshadow nude rose, look annees 50-60.",
    "streetwear":    "Teint glowy naturel, mascara noir, baume teinte, sourcils naturels brosses, "
                     "look frais et casual urbain.",
    "romantique":    "Blush pommettes rose poudre, eyeshadow rose ou peche, levres rose framboise glossy, "
                     "mascara brun, look feminin et tendre.",
    "sophistique":   "Teint travaille au pinceau, contouring leger, eyeshadow taupe ou nude profond, "
                     "eyeliner tres fin, levres rose nude mat, sourcils impeccables.",
    "chic":          "Peau lumineuse, mascara seul ou eyeliner discret, levres rose vieux ou nude rose, "
                     "blush peche, look raffine et intemporel.",
    "festif":        "Paillettes ou shimmer sur paupiere mobile, eyeliner accentue, levres rouge ou prune, "
                     "highlighter intense, look party impactant.",
    "casual":        "Baume teinte, mascara leger, blush a peine visible, sourcils brosses, "
                     "look minimal et naturel pour tous les jours.",
    "sportif":       "Peau saine et nette (BB cream legere), mascara waterproof, baume hydratant, "
                     "sourcils brosses, zero complexite, look athletique.",
    "rock":          "Smoky eye charbon ou noir, eyeliner appuye, levres bordeaux ou nude mat, "
                     "sourcils marques, look intense et affirme.",
    "cocktail":      "Teint glowy, eyeliner discret, smoky eye doux, levres rouge profond ou nude glossy, "
                     "blush rose, look feminin soigne pour soiree.",
    "hippie":        "Teint dore au bronzer, eyeshadow terreux, mascara naturel, levres nude ou peche, "
                     "blush peche, look libre et solaire.",
    "bourgeois":     "Peau parfaite et hydratee, eyeshadow nude beige, mascara discret, "
                     "levres rose nude mat, sourcils naturellement structures, luxe discret.",
}

# Conseils coiffure par style semantique
HAIR_TIPS = {
    "naturel":       "Coiffure naturelle sans exces de produit, mouvement libre, "
                     "ondulations douces ou queue-de-cheval lache.",
    "professionnel": "Coiffure tiree, chignon bas ou queue-de-cheval nette, "
                     "raie sur le cote, look soigne et credible.",
    "elegant":       "Coiffure ondulee ou chignon haut sophistique, "
                     "accessoires raffines (barrette doree, peigne discret).",
    "glamour":       "Wavy hair travaille, volume aux racines, brushing impactant, "
                     "ondulations larges type tapis rouge.",
    "minimaliste":   "Coiffure simple : raie au milieu, cheveux lisses ou queue-de-cheval haute, "
                     "zero artifice, lignes nettes.",
    "doux":          "Coiffure souple et romantique, ondulations legeres, "
                     "demi-attache avec ruban ou epingle delicate.",
    "moderne":       "Coupe structuree (carre net, frange micro), wet look, "
                     "ou coiffure sleek avec gel pour effet contemporain.",
    "gothique":      "Cheveux noirs lisses ou crepus volumineux, raie au milieu, "
                     "ou coiffure crantee avec accessoires metalliques sombres.",
    "vintage":       "Victory rolls, ondulations Hollywood des annees 40-50, "
                     "chignon banane, ou queue-de-cheval haute avec foulard.",
    "streetwear":    "Cheveux laches au naturel, queue-de-cheval haute decontractee, "
                     "ou casquette/bandana, look spontane et urbain.",
    "romantique":    "Boucles douces, demi-attache avec ruban, cheveux ondules feminins, "
                     "tresse couronne ou epingles fleurs delicates.",
    "sophistique":   "Brushing impeccable, chignon bas tres lisse, ou queue-de-cheval basse "
                     "tres polie avec raie sur le cote, finition glossy.",
    "chic":          "Coiffure lisse et nette : carre droit, queue-de-cheval mi-haute lisse, "
                     "ou ondulations tres legeres, raie sur le cote.",
    "festif":        "Volume travaille, ondulations larges glamour, ou chignon haut destructure "
                     "avec accessoires brillants (barrettes strass, peigne dore).",
    "casual":        "Cheveux laches naturels, queue-de-cheval lache, chignon decoiffe, "
                     "ou bun haut decontracte sans produits.",
    "sportif":       "Queue-de-cheval haute serree, tresse plaquee ou demi-chignon haut, "
                     "front degage, look fonctionnel et propre.",
    "rock":          "Coupe asymetrique, cheveux mi-longs effiles, frange droite epaisse, "
                     "shaggy cut, ou wet look gel-back affirme.",
    "cocktail":      "Brushing impeccable, ondulations souples, demi-attache elegante avec "
                     "epingle dorée, ou chignon bas raffine.",
    "hippie":        "Cheveux longs ondules au naturel, tresses laterales ou couronne, "
                     "headband fleur ou bandeau ethnique, aspect non force.",
    "bourgeois":     "Brushing soigne, raie sur le cote impeccable, chignon bas tres ordonne, "
                     "ou cheveux longs lisses parfaitement entretenus, look discret.",
}

# Conseils vetements par style semantique
CLOTHES_TIPS = {
    "naturel":       "Tenue simple et confortable. Couleurs neutres (beige, blanc casse, kaki). "
                     "Matieres naturelles (lin, coton). Coupe decontractee mais soignee, "
                     "baskets blanches ou mocassins.",
    "professionnel": "Blazer cintre, chemise sobre (blanche, bleu clair), pantalon de tailleur "
                     "ou jupe crayon. Couleurs neutres (noir, gris, marine, beige). "
                     "Chaussures fermees, eviter motifs voyants et logos.",
    "elegant":       "Coupe ajustee, matieres nobles (laine, soie, satin). Robe fluide ou "
                     "tailleur structure. Couleurs profondes (bordeaux, marine, emeraude). "
                     "Accessoires raffines : sac a main, bijoux discrets, escarpins.",
    "glamour":       "Robe moulante ou tailleur audacieux, paillettes, sequins ou velours. "
                     "Decollete assume, talons hauts. Accessoires forts : bijoux brillants, "
                     "pochette, rouge a levres assorti.",
    "minimaliste":   "Lignes epurees, pas de logos, matieres premium. Total look monochrome "
                     "(blanc, noir, gris, beige). Coupes nettes, accessoires quasi invisibles. "
                     "Less is more : 1 ou 2 pieces fortes max.",
    "doux":          "Couleurs pastel (rose poudre, bleu ciel, lavande), matieres douces "
                     "(coton, mohair, mousseline). Coupes feminines : jupes mi-longues, "
                     "blouses fluides, details volants ou dentelle.",
    "moderne":       "Coupes audacieuses, asymetrie, color blocking. Matieres techniques "
                     "(cuir, vinyle, mesh). Couleurs vives ou metalliques. "
                     "Sneakers tendance ou bottines, accessoires statement.",
    "gothique":      "Total black ou tons sombres (bordeaux, violet profond, gris anthracite). "
                     "Cuir, dentelle noire, velours, mesh. Bottines cloutees ou Doc Martens, "
                     "ceintures a chaines, bijoux argent (croix, pentacles, chokers).",
    "vintage":       "Coupes retro annees 50-60-70 : robe trapeze, jupe corolle, jean taille haute, "
                     "blouse a col Claudine. Imprimes pois, fleurs, motifs geometriques. "
                     "Mocassins, escarpins kitten heels, sac vintage en cuir patine.",
    "streetwear":    "Sweat oversize, hoodie, jogging premium, jean baggy, cropped top. "
                     "Couleurs neutres avec touches vives. Sneakers tendance (Nike, Adidas, New Balance), "
                     "casquette, sac banane ou tote bag, accessoires logo.",
    "romantique":    "Robe fluide a fleurs, blouse a manches bouffantes, jupe midi plissee. "
                     "Couleurs douces (rose poudre, blanc casse, lilas, bleu ciel). Dentelle, "
                     "broderie anglaise. Ballerines, sandales tressees, sac en paille ou pochette satin.",
    "sophistique":   "Tailleur structure, robe fourreau noire, blouse en soie, pantalon cigarette. "
                     "Couleurs neutres haut de gamme (noir, marine, taupe, ivoire). Matieres premium "
                     "(soie, cachemire, laine fine). Escarpins fins, sac en cuir lisse.",
    "chic":          "Pantalon droit, blazer cintre, chemise de qualite, robe portefeuille. "
                     "Palette neutre raffinee (beige, blanc, noir, marine). Accessoires intemporels : "
                     "ballerines plates, mocassins, sac structure, foulard en soie.",
    "festif":        "Robe paillettes, top sequins, jupe satin, combinaison eclat. "
                     "Couleurs metallisees (or, argent, cuivre) ou tons profonds (bordeaux, emeraude). "
                     "Talons hauts, pochette a paillettes, bijoux brillants statement.",
    "casual":        "Jean classique, T-shirt blanc, sweat basique, sneakers, baskets. "
                     "Couleurs neutres et basiques. Pieces simples Uniqlo, H&M, COS, Levi's. "
                     "Confortable et passe-partout.",
    "sportif":       "Legging technique, brassiere, sneakers de course, sweat zip, jogging premium. "
                     "Marques : Nike, Adidas, Lululemon, Under Armour, On Running. "
                     "Look athleisure pour ville ou sport, casquette ou bonnet en option.",
    "rock":          "Jean noir slim, t-shirt rock, perfecto en cuir, pantalon vinyle. "
                     "Tons noir / gris / rouge fonce. Marques : Saint Laurent, IRO, The Kooples, "
                     "Zadig & Voltaire. Bottines a clous ou Doc Martens, ceinture cloutee.",
    "cocktail":      "Robe cocktail mi-longue, tailleur jupe, top satin avec pantalon de tailleur. "
                     "Couleurs noir, rouge, marine, emeraude. Matieres satin, dentelle, crepe. "
                     "Escarpins, pochette glamour, bijoux raffines (collier en or, boucles d'oreilles).",
    "hippie":        "Robe longue fluide, jupe gypsy, blouse paysanne, jean pat'd'eph. "
                     "Tons chauds (terracotta, ocre, kaki, creme), motifs ethniques et fleurs. "
                     "Sandales tressees, sac frange, bijoux ethniques superposes.",
    "bourgeois":     "Pull en cachemire beige, chemise blanche, pantalon en lin, jupe plissee marine. "
                     "Mocassins, sac en cuir patine, foulard. Materials premium (cachemire, soie, lin, "
                     "laine) sans logos. Marques : Sezane, Maje, Sandro, Massimo Dutti.",
}

# Alias pour compatibilite ascendante
RELOOKING_TIPS = MAKEUP_TIPS

# Description courte du contexte evenementiel.
# Le matching se fait par mot-cle via "if key in event.lower()".
EVENT_DESCRIPTIONS = {
    "entretien":     "Sobriete, credibilite, peu d'artifice. Le maquillage doit rassurer pas distraire.",
    "premier jour":  "Premier jour de travail : look soigne, naturel, premiere impression sans en faire trop.",
    "presentation":  "Look net qui inspire confiance. Eviter ce qui peut detourner l'attention.",
    "reunion":       "Professionnel et soigne, sans tomber dans le formel excessif.",
    "after work":    "Look chic mais decontracte, transition bureau-soiree, un cran au-dessus du quotidien.",
    "gala":          "Glamour assume, on ose les couleurs profondes et les details.",
    "soiree":        "Plus libre, on peut accentuer yeux ou levres selon le code.",
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
            print(f"  Maquillage : {MAKEUP_TIPS.get(reco, '')}")
            print(f"  Coiffure   : {HAIR_TIPS.get(reco, '')}")
            print(f"  Vetements  : {CLOTHES_TIPS.get(reco, '')}\n")


if __name__ == "__main__":
    main()
