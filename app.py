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
from image_search import search_products_batch

# ----- Configuration page -----
st.set_page_config(
    page_title="MakeAIPrep - Relooking IA",
    layout="wide",
)

# ----- CSS custom (charte vert/cyan inspiree des maquettes Visily) -----
st.markdown("""
<style>
    /* Palette couleurs */
    :root {
        --primary-cyan: #4FD1B5;
        --primary-cyan-dark: #3BC4A6;
        --cta-green: #D4F58F;
        --cta-green-dark: #C2EB78;
        --bg-cream: #FAFAF7;
        --text-dark: #1F1F1F;
        --text-muted: #666;
        --card-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }

    /* Fond general */
    .main, .stApp {
        background-color: var(--bg-cream);
    }

    /* Titres */
    h1 {
        color: var(--text-dark);
        font-weight: 700;
        text-align: center;
    }
    h2, h3, h4 {
        color: var(--text-dark);
        font-weight: 600;
    }

    /* Boutons primary (form_submit_button type primary) */
    .stButton>button[kind="primary"],
    button[data-testid="stFormSubmitButton"] {
        background: var(--cta-green) !important;
        color: var(--text-dark) !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.2s !important;
        box-shadow: 0 2px 6px rgba(212,245,143,0.5) !important;
    }
    .stButton>button[kind="primary"]:hover,
    button[data-testid="stFormSubmitButton"]:hover {
        background: var(--cta-green-dark) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(212,245,143,0.6) !important;
    }

    /* Card de conseil / recommandation */
    .reco-box {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: var(--card-shadow);
        margin: 1rem 0;
    }

    /* Badge du style recommande */
    .style-badge {
        display: inline-block;
        background: var(--primary-cyan);
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
    }

    /* Header MakeAIPrep */
    .makeaiprep-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #eee;
    }
    .makeaiprep-logo {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 700;
        font-size: 1.4rem;
        color: var(--text-dark);
    }
    .makeaiprep-logo svg {
        width: 28px;
        height: 28px;
    }
    .makeaiprep-nav {
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }
    .makeaiprep-nav a {
        color: var(--text-dark);
        text-decoration: none;
        font-weight: 500;
        font-size: 0.95rem;
    }
    .makeaiprep-nav a:hover {
        color: var(--primary-cyan-dark);
    }
    .makeaiprep-cta {
        background: var(--cta-green);
        color: var(--text-dark);
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 600;
        text-decoration: none;
        border: none;
    }

    /* Footer */
    .makeaiprep-footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        margin-top: 3rem;
        border-top: 1px solid #eee;
        color: var(--text-muted);
        font-size: 0.85rem;
    }
    .makeaiprep-footer-icons {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        margin-bottom: 1rem;
    }
    .makeaiprep-footer-icons a {
        color: var(--text-muted);
        text-decoration: none;
    }
    .makeaiprep-footer-icons a:hover {
        color: var(--primary-cyan-dark);
    }
    .makeaiprep-footer-icons svg {
        width: 22px;
        height: 22px;
    }

    /* Sidebar plus discrete */
    [data-testid="stSidebar"] {
        background-color: white;
        border-right: 1px solid #eee;
    }

    /* Inputs / form fields */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div {
        border-radius: 10px !important;
    }

    /* Image hover effect */
    [data-testid="stImage"] img {
        border-radius: 10px;
        transition: transform 0.2s;
    }
    [data-testid="stImage"] img:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)


# ----- Header MakeAIPrep -----
st.markdown("""
<div class="makeaiprep-header">
  <div class="makeaiprep-logo">
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 2L13.5 8.5L20 10L13.5 11.5L12 18L10.5 11.5L4 10L10.5 8.5L12 2Z"
            fill="#4FD1B5" stroke="#4FD1B5" stroke-width="0.5" stroke-linejoin="round"/>
      <circle cx="19" cy="5" r="1.5" fill="#D4F58F"/>
      <circle cx="5" cy="18" r="1" fill="#D4F58F"/>
    </svg>
    <span>Make<span style="color:#4FD1B5;">AI</span>Prep</span>
  </div>
  <div class="makeaiprep-nav">
    <a href="#">Accueil</a>
    <a href="#">Analyser mon look</a>
    <a href="#" class="makeaiprep-cta">Connexion</a>
  </div>
</div>
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


# ----- Sous-titre principal -----
st.markdown(
    "<h2 style='text-align:center; font-weight:600; margin-bottom:0.3rem;'>"
    "Téléchargez votre photo pour une analyse IA"
    "</h2>"
    "<p style='text-align:center; color:#666; margin-bottom:2rem;'>"
    "Uploadez une photo claire de votre visage pour que notre IA puisse "
    "vous proposer les meilleurs looks."
    "</p>",
    unsafe_allow_html=True,
)

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

# Modele ML toujours utilise (CLIP+MLP donne les meilleurs resultats,
# on cache le choix a l'utilisateur final).
model_choice = "CLIP+MLP (recommandé)"

# ----- Sidebar : juste la liste des styles disponibles -----
with st.sidebar:
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
st.markdown("### Votre photo")

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
        "Lancer l'analyse",
        type="primary",
        use_container_width=True,
    )

st.markdown("---")

# ----- Section 2 : recommandation (pleine largeur, sous la section 1) -----
st.markdown("### Vos Looks Recommandés par l'IA")

if image_file is None:
    st.info("Uploadez une photo ou utilisez la webcam pour commencer.")
elif not submitted:
    st.info(
        "Configurez vos préférences (genre, type de relooking, précisions) "
        "puis cliquez sur **Lancer l'analyse**."
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

        advice_data = None
        if api_key_present:
            with st.spinner("Génération du conseil personnalisé..."):
                try:
                    advice_data = generate_advice(
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

        # Badge du style en haut
        st.markdown(
            f'<div style="margin-bottom:1rem;">'
            f'<span class="style-badge">{style_used.title()}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if advice_data and "sections" in advice_data:
            # Affichage structure du LLM avec images des produits
            for section in advice_data["sections"]:
                section_title = section.get("title") or section.get("type", "Conseil")
                section_text = section.get("text", "")
                products = section.get("products", [])

                # Carte conseil
                st.markdown(f"#### {section_title}")
                st.markdown(
                    f'<div style="background:#fff; border-radius:12px; padding:1.2rem; '
                    f'box-shadow:0 2px 8px rgba(0,0,0,0.05); color:#333; line-height:1.6; '
                    f'margin-bottom:1rem;">{section_text}</div>',
                    unsafe_allow_html=True,
                )

                # Grille des produits avec images
                if products:
                    with st.spinner(f"Recherche des produits pour {section_title.lower()}..."):
                        products_with_images = search_products_batch(products[:6])

                    st.markdown("**Articles / produits suggérés :**")
                    # Affichage en lignes de 3 colonnes
                    for row_start in range(0, len(products_with_images), 3):
                        row = products_with_images[row_start:row_start + 3]
                        cols = st.columns(3)
                        for i, prod in enumerate(row):
                            with cols[i]:
                                img_url = prod.get("image_url")
                                name = prod.get("name", "")
                                brand = prod.get("brand", "")
                                if img_url:
                                    try:
                                        st.image(img_url, use_container_width=True)
                                    except Exception:
                                        st.markdown("*(image indisponible)*")
                                st.markdown(
                                    f"<div style='text-align:center; font-size:0.85rem; "
                                    f"line-height:1.3; margin-top:0.3rem;'>"
                                    f"<strong>{brand}</strong><br>{name}"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                    st.markdown("---")
        else:
            # Fallback : dicts statiques (pas d'images)
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
                f'<div class="reco-box">{fallback_blocks}</div>',
                unsafe_allow_html=True,
            )

# ----- Footer MakeAIPrep -----
st.markdown("""
<div class="makeaiprep-footer">
  <div class="makeaiprep-footer-icons">
    <a href="#" aria-label="Facebook">
      <svg viewBox="0 0 24 24" fill="currentColor"><path d="M22 12c0-5.52-4.48-10-10-10S2 6.48 2 12c0 4.84 3.44 8.87 8 9.8V15H8v-3h2V9.5C10 7.57 11.57 6 13.5 6H16v3h-2c-.55 0-1 .45-1 1v2h3v3h-3v6.95c5.05-.5 9-4.76 9-9.95z"/></svg>
    </a>
    <a href="#" aria-label="Instagram">
      <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2.16c3.2 0 3.58.01 4.85.07 1.17.05 1.8.25 2.23.41.56.22.96.48 1.38.9.42.42.68.82.9 1.38.16.42.36 1.06.41 2.23.06 1.27.07 1.65.07 4.85s-.01 3.58-.07 4.85c-.05 1.17-.25 1.8-.41 2.23-.22.56-.48.96-.9 1.38-.42.42-.82.68-1.38.9-.42.16-1.06.36-2.23.41-1.27.06-1.65.07-4.85.07s-3.58-.01-4.85-.07c-1.17-.05-1.8-.25-2.23-.41-.56-.22-.96-.48-1.38-.9-.42-.42-.68-.82-.9-1.38-.16-.42-.36-1.06-.41-2.23-.06-1.27-.07-1.65-.07-4.85s.01-3.58.07-4.85c.05-1.17.25-1.8.41-2.23.22-.56.48-.96.9-1.38.42-.42.82-.68 1.38-.9.42-.16 1.06-.36 2.23-.41 1.27-.06 1.65-.07 4.85-.07M12 0C8.74 0 8.33.01 7.05.07 5.78.13 4.9.33 4.14.63c-.79.3-1.46.72-2.13 1.39C1.34 2.7.92 3.36.62 4.16.32 4.92.13 5.79.07 7.06.01 8.33 0 8.74 0 12s.01 3.67.07 4.95c.06 1.27.26 2.14.56 2.91.3.8.72 1.46 1.39 2.13.67.67 1.34 1.09 2.13 1.39.77.3 1.65.5 2.92.56C8.33 23.99 8.74 24 12 24s3.67-.01 4.95-.07c1.27-.06 2.14-.26 2.91-.56.8-.3 1.46-.72 2.13-1.39.67-.67 1.09-1.34 1.39-2.13.3-.77.5-1.65.56-2.92.06-1.27.07-1.68.07-4.94s-.01-3.67-.07-4.95c-.06-1.27-.26-2.14-.56-2.91-.3-.8-.72-1.46-1.39-2.13C21.31 1.34 20.65.92 19.85.62c-.77-.3-1.65-.5-2.92-.56C15.67.01 15.26 0 12 0zm0 5.84c-3.4 0-6.16 2.76-6.16 6.16s2.76 6.16 6.16 6.16 6.16-2.76 6.16-6.16S15.4 5.84 12 5.84zM12 16c-2.21 0-4-1.79-4-4s1.79-4 4-4 4 1.79 4 4-1.79 4-4 4zm6.41-11.85c-.8 0-1.44.65-1.44 1.44s.65 1.44 1.44 1.44 1.44-.65 1.44-1.44-.65-1.44-1.44-1.44z"/></svg>
    </a>
    <a href="#" aria-label="Twitter">
      <svg viewBox="0 0 24 24" fill="currentColor"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117L17.083 19.77z"/></svg>
    </a>
  </div>
  <div>MakeAIPrep · Projet Ydays · CLIP+MLP & Gemini Vision</div>
</div>
""", unsafe_allow_html=True)
