"""
MakeAIPrep - Interface web Streamlit
=====================================

Demo interactive : upload une photo (ou prends-la avec ta camera),
choisis l'evenement, recois ta recommandation de relooking.

Usage:
    streamlit run app.py
"""

import os
import ssl
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from predict import (
    MAKEUP_TIPS, HAIR_TIPS, CLOTHES_TIPS, EVENT_DESCRIPTIONS,
    load_resnet, predict_resnet,
    load_clip_mlp, predict_clip_mlp,
    expected_styles, build_recommendation, find_checkpoint,
)
from llm_advisor import generate_advice, get_api_key

# ----- Configuration page -----
st.set_page_config(
    page_title="MakeAIPrep - Relooking IA",
    layout="wide",
)

# ----- CSS custom -----
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #fdf2f8 0%, #fef3c7 100%);
    }
    h1 {
        color: #be185d;
        text-align: center;
        font-family: 'Georgia', serif;
    }
    .stButton>button {
        background: linear-gradient(90deg, #ec4899, #f59e0b);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.6rem 2rem;
        font-weight: bold;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }
    .reco-box {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }
    .style-badge {
        display: inline-block;
        background: linear-gradient(90deg, #ec4899, #f59e0b);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 15px;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ----- Cache du chargement modeles -----
@st.cache_resource
def load_config_and_classes():
    cfg = yaml.safe_load(open("./models/config.yaml"))
    data_dir = Path("./datasets_faces")
    classes = sorted([d.name for d in (data_dir / "images").iterdir() if d.is_dir()])
    return cfg, classes


@st.cache_resource
def load_model_resnet(num_classes):
    ckpt = find_checkpoint("resnet", Path("./results"))
    if ckpt is None:
        return None
    return load_resnet(num_classes, ckpt, torch.device("cpu"))


@st.cache_resource
def load_model_clip_mlp(num_classes, _cfg):
    ckpt = find_checkpoint("clip_mlp", Path("./results"))
    if ckpt is None:
        return None
    return load_clip_mlp(num_classes, ckpt, _cfg, torch.device("cpu"))


# ----- Header -----
st.markdown("# MakeAIPrep")
st.markdown(
    "<h3 style='text-align:center; font-style:italic; font-weight:normal;'>"
    "Votre coach relooking IA pour briller à chaque évènement"
    "</h3>",
    unsafe_allow_html=True,
)
st.markdown("---")

cfg, classes = load_config_and_classes()
device = torch.device("cpu")

# Styles affiches dans l'UI (peut differer des classes du modele ML).
# - "boheme" est exclu de l'affichage meme si le modele peut le predire (filtre plus bas).
# - Styles supplementaires (gothique, vintage, streetwear) dispos uniquement en "Style impose"
#   car le modele ML n'a pas ete entraine dessus.
STYLE_BLACKLIST = {"boheme"}
EXTRA_STYLES = [
    "bourgeois",
    "casual",
    "chic",
    "cocktail",
    "festif",
    "gothique",
    "hippie",
    "moderne",
    "rock",
    "romantique",
    "sophistique",
    "sportif",
    "streetwear",
    "vintage",
]
DISPLAY_STYLES = sorted(
    [c for c in classes if c not in STYLE_BLACKLIST] + EXTRA_STYLES
)

# ----- Sidebar : juste le modele ML et la liste des styles -----
with st.sidebar:
    st.markdown("## Modèle IA")
    model_choice = st.radio(
        "Quel modèle utiliser ?",
        ["CLIP+MLP (recommandé)", "ResNet50", "Les deux"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### Styles disponibles")
    for c in DISPLAY_STYLES:
        st.markdown(f"- **{c.title()}**")

# Mapping evenement -> code interne (utilise par le form principal)
EVENT_OPTIONS = {
    "Entretien d'embauche": "entretien d'embauche",
    "Premier jour de travail": "premier jour de travail",
    "Présentation projet": "presentation projet",
    "Réunion professionnelle": "reunion professionnelle",
    "After work": "after work",
    "Soirée de gala": "soiree gala",
}


# ----- Section 1 : photo + formulaire (pleine largeur) -----
st.markdown("## 1. Configuration")

tab_upload, tab_camera = st.tabs(["Upload", "Webcam"])

image_file = None
with tab_upload:
    image_file = st.file_uploader(
        "Choisissez une photo (visage de préférence)",
        type=["jpg", "jpeg", "png"],
    )
with tab_camera:
    cam_image = st.camera_input("Prenez votre photo")
    if cam_image is not None:
        image_file = cam_image

if image_file is not None:
    img = Image.open(image_file).convert("RGB")
    # Photo en taille raisonnable (centree via colonnes)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.image(img, caption="Votre photo", width="stretch")

# Form regroupant les preferences : pas de rerun tant qu'on ne clique pas submit
with st.form("relooking_form", clear_on_submit=False):
    col_evt, col_style = st.columns(2)
    with col_evt:
        st.markdown("### Évènement")
        event_label = st.selectbox(
            "Pour quel évènement ?",
            list(EVENT_OPTIONS.keys()),
            index=0,
        )
        event = EVENT_OPTIONS[event_label]
    with col_style:
        st.markdown("### Style à imposer")
        forced_style = st.selectbox(
            "Forcer un style ? (optionnel)",
            ["Peu importe"] + DISPLAY_STYLES,
            index=0,
            format_func=lambda x: x if x == "Peu importe" else x.title(),
            help="Par défaut, le modèle ML détermine le style. Choisis un style ici pour bypass.",
        )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Genre")
        gender_choice = st.radio(
            "La personne sur la photo est :",
            ["Femme", "Homme", "Non binaire"],
            index=0,
        )
    with col_b:
        st.markdown("### Type de relooking")
        st.caption("Conseils souhaités (cochez ce qui vous intéresse)")
        focus_makeup = st.checkbox("Maquillage / Soin", value=True)
        focus_hair = st.checkbox("Coiffure", value=True)
        focus_clothes = st.checkbox("Vêtements", value=True)

    st.markdown("### Précisions (optionnel)")
    event_name = st.text_input(
        "Nom précis de l'événement",
        placeholder="Ex: Entretien chez BNP Paribas, Soutenance Ydays, Réunion équipe marketing...",
        help="Le nom exact peut influencer le ton de la recommandation (secteur, formel, festif, etc.).",
    )
    user_description = st.text_area(
        "Vos préférences / contraintes",
        placeholder="Ex: J'aime les couleurs sobres, peau sensible, budget étudiant, pas de talons...",
        height=100,
        help="Toutes les précisions à prendre en compte par l'IA.",
    )

    submitted = st.form_submit_button(
        "Générer la recommandation",
        type="primary",
        use_container_width=True,
    )

st.markdown("---")

# ----- Section 2 : recommandation (pleine largeur, sous la section 1) -----
st.markdown("## 2. Votre recommandation")

if image_file is None:
    st.info("Uploadez une photo ou utilisez la webcam pour commencer.")
elif not submitted:
    st.info(
        "Configurez vos préférences (genre, type de relooking, précisions) "
        "puis cliquez sur **Générer la recommandation**."
    )
else:
    # Mapper UI -> codes API LLM
    gender_code = {
        "Femme": "femme",
        "Homme": "homme",
        "Non binaire": "non binaire",
    }[gender_choice]

    # Construire la liste des focus areas selon les checkboxes cochees
    focus_areas = []
    if focus_makeup:
        focus_areas.append("maquillage")
    if focus_hair:
        focus_areas.append("coiffure")
    if focus_clothes:
        focus_areas.append("vetements")
    if not focus_areas:
        st.warning("Aucun type de relooking coché — affichage de tout par défaut.")
        focus_areas = ["maquillage", "coiffure", "vetements"]
    api_key_present = get_api_key() is not None
    if not api_key_present:
        st.warning(
            "Pas de clé API Gemini configurée — conseils statiques uniquement. "
            "Ajoute `GOOGLE_API_KEY = \"...\"` dans `.streamlit/secrets.toml` pour des conseils personnalisés et variés."
        )

    # Contexte evenementiel
    ev_desc = next((v for k, v in EVENT_DESCRIPTIONS.items()
                    if k in event.lower()),
                   "Adapter le look au contexte.")
    st.markdown(f"**Contexte** : *{ev_desc}*")

    # Liste des styles a render : (label, style)
    styles_to_render = []

    is_forced = forced_style != "Peu importe"
    if is_forced:
        # Style impose -> bypass des modeles ML
        st.success(f"Style imposé : **{forced_style.title()}**")
        styles_to_render.append(("imposé", forced_style))
    else:
        # Prediction par les modeles ML
        tmp_path = Path("./_tmp_input.jpg")
        img.save(tmp_path)

        with st.spinner("Analyse ML en cours..."):
            expected = expected_styles(event, cfg)

            models_to_run = []
            if model_choice in ("CLIP+MLP (recommandé)", "Les deux"):
                models_to_run.append("clip_mlp")
            if model_choice in ("ResNet50", "Les deux"):
                models_to_run.append("resnet")

            results = {}
            for mn in models_to_run:
                try:
                    if mn == "clip_mlp":
                        loaded = load_model_clip_mlp(len(classes), cfg)
                        if loaded is None:
                            st.warning(f"{mn} : pas de checkpoint trouvé. "
                                       "Lance `python train_clip_mlp_fast.py --data-dir ./datasets_faces`")
                            continue
                        mlp, clip_m, prep, tok = loaded
                        probs = predict_clip_mlp(mlp, clip_m, prep, tok,
                                                 tmp_path, event, classes, device)
                    else:
                        m = load_model_resnet(len(classes))
                        if m is None:
                            st.warning(f"{mn} : pas de checkpoint trouvé.")
                            continue
                        probs = predict_resnet(m, tmp_path, device)
                    results[mn] = probs
                except Exception as e:
                    st.error(f"Erreur {mn} : {e}")

        try:
            tmp_path.unlink()
        except Exception:
            pass

        if not results:
            st.error("Aucun modèle disponible. Entraîne d'abord les modèles ou choisis un style à imposer dans la sidebar.")
        else:
            if expected:
                st.markdown(f"**Styles idéaux pour cet évènement** : "
                            + " · ".join([f"`{s.title()}`" for s in expected]))

            for mn, probs in results.items():
                # Ecarter les styles blacklistes (boheme) de la prediction
                probs = probs.copy()
                for blacklisted in STYLE_BLACKLIST:
                    if blacklisted in classes:
                        probs[classes.index(blacklisted)] = 0.0
                top, reco, verdict = build_recommendation(probs, classes, expected, top_k=3)

                with st.container():
                    st.markdown(f"### {mn.upper().replace('_', '+')}")
                    for rank, (cls, p) in enumerate(top, 1):
                        col_a, col_b = st.columns([1, 4])
                        with col_a:
                            st.markdown(f"**#{rank} {cls.title()}**")
                        with col_b:
                            st.progress(float(p), text=f"{p*100:.1f}%")
                    if "OK" in verdict:
                        st.success(verdict)
                    elif "ATTENTION" in verdict:
                        st.warning(verdict)
                    else:
                        st.info(verdict)

                styles_to_render.append((mn, reco))

    # Generation des conseils pour chaque style retenu
    for label, style_used in styles_to_render:
        st.markdown(f"### Conseil relooking — **{style_used.title()}**")

        advice_text = None
        if api_key_present:
            with st.spinner("Génération du conseil personnalisé..."):
                try:
                    advice_text = generate_advice(
                        img,
                        style=style_used,
                        event=event,
                        gender=gender_code,
                        focus_areas=focus_areas,
                        event_name=event_name,
                        user_description=user_description,
                    )
                except Exception as e:
                    st.warning(f"Erreur API Gemini : {e}\n\nConseil générique affiché à la place.")

        if advice_text:
            # Conseil personnalise du LLM
            html_text = advice_text.replace("\n", "<br>")
            st.markdown(
                f'<div class="reco-box">'
                f'<span class="style-badge">{style_used.title()}</span>'
                f'<div style="margin-top:1rem; color:#333; line-height:1.6;">{html_text}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            # Fallback : dicts statiques
            fallback_blocks = ""
            if "maquillage" in focus_areas:
                fallback_blocks += (
                    f'<p style="margin-top:1rem; color:#444;">'
                    f'<strong>Maquillage :</strong><br>'
                    f'{MAKEUP_TIPS.get(style_used, "")}'
                    f'</p>'
                )
            if "coiffure" in focus_areas:
                fallback_blocks += (
                    f'<p style="margin-top:1rem; color:#444;">'
                    f'<strong>Coiffure :</strong><br>'
                    f'{HAIR_TIPS.get(style_used, "")}'
                    f'</p>'
                )
            if "vetements" in focus_areas:
                fallback_blocks += (
                    f'<p style="margin-top:1rem; color:#444;">'
                    f'<strong>Tenue :</strong><br>'
                    f'{CLOTHES_TIPS.get(style_used, "")}'
                    f'</p>'
                )
            st.markdown(
                f'<div class="reco-box">'
                f'<span class="style-badge">{style_used.title()}</span>'
                f'{fallback_blocks}'
                f'</div>',
                unsafe_allow_html=True,
            )

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#999; font-size:0.85rem;'>"
    "MakeAIPrep · Projet Ydays · Powered by ResNet50 + CLIP"
    "</div>",
    unsafe_allow_html=True,
)
