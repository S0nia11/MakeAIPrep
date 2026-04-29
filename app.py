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
    RELOOKING_TIPS, EVENT_DESCRIPTIONS,
    load_resnet, predict_resnet,
    load_clip_mlp, predict_clip_mlp,
    expected_styles, build_recommendation, find_checkpoint,
)

# ----- Configuration page -----
st.set_page_config(
    page_title="MakeAIPrep - Relooking IA",
    page_icon="💄",
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
st.markdown("# 💄 MakeAIPrep")
st.markdown("### *Votre coach relooking IA pour briller à chaque évènement*")
st.markdown("---")

cfg, classes = load_config_and_classes()
device = torch.device("cpu")

# ----- Sidebar : choix de l'evenement et du modele -----
with st.sidebar:
    st.markdown("## 🎯 Contexte")

    event_options = {
        "🎓 Entretien d'embauche": "entretien d'embauche",
        "💼 Entretien d'alternance": "entretien d'alternance",
        "📊 Présentation projet": "presentation projet",
        "🏢 Réunion professionnelle": "reunion professionnelle",
        "🌟 Premier jour de stage": "stage premier jour",
        "🍸 Soirée networking": "soiree networking",
        "✨ Soirée de gala": "soiree gala",
        "📚 Journée campus": "journee campus",
    }
    event_label = st.selectbox(
        "Pour quel évènement ?",
        list(event_options.keys()),
        index=0,
    )
    event = event_options[event_label]

    st.markdown("## 🤖 Modèle IA")
    model_choice = st.radio(
        "Quel modèle utiliser ?",
        ["CLIP+MLP (recommandé)", "ResNet50", "Les deux"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 📊 Styles disponibles")
    for c in classes:
        st.markdown(f"- **{c}**")


# ----- Zone principale : input image -----
col_input, col_result = st.columns([1, 1.3])

with col_input:
    st.markdown("## 📸 Votre photo")
    tab_upload, tab_camera = st.tabs(["📁 Upload", "📷 Webcam"])

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
        st.image(img, caption="Votre photo", width="stretch")

with col_result:
    st.markdown("## ✨ Votre recommandation")

    if image_file is None:
        st.info("👈 Uploadez une photo ou utilisez la webcam pour commencer")
    else:
        # Sauvegarder temporairement pour la prediction
        tmp_path = Path("./_tmp_input.jpg")
        img.save(tmp_path)

        with st.spinner("✨ Analyse en cours..."):
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
                            st.warning(f"⚠️ {mn} : pas de checkpoint trouvé. "
                                       "Lance `python train_clip_mlp_fast.py --data-dir ./datasets_faces`")
                            continue
                        mlp, clip_m, prep, tok = loaded
                        probs = predict_clip_mlp(mlp, clip_m, prep, tok,
                                                 tmp_path, event, classes, device)
                    else:
                        m = load_model_resnet(len(classes))
                        if m is None:
                            st.warning(f"⚠️ {mn} : pas de checkpoint trouvé.")
                            continue
                        probs = predict_resnet(m, tmp_path, device)
                    results[mn] = probs
                except Exception as e:
                    st.error(f"Erreur {mn} : {e}")

        if not results:
            st.error("Aucun modèle disponible. Entraîne d'abord les modèles.")
        else:
            # Contexte de l'evenement
            ev_desc = next((v for k, v in EVENT_DESCRIPTIONS.items()
                            if k in event.lower()),
                           "Adapter le look au contexte.")
            st.markdown(f"**Contexte** : *{ev_desc}*")
            if expected:
                st.markdown(f"**Styles idéaux pour cet évènement** : "
                            + " · ".join([f"`{s}`" for s in expected]))

            for mn, probs in results.items():
                top, reco, verdict = build_recommendation(probs, classes, expected, top_k=3)

                with st.container():
                    st.markdown(f"### 🤖 {mn.upper().replace('_', '+')}")

                    # Top-3 avec barres
                    for rank, (cls, p) in enumerate(top, 1):
                        col_a, col_b = st.columns([1, 4])
                        with col_a:
                            st.markdown(f"**#{rank} {cls}**")
                        with col_b:
                            st.progress(float(p), text=f"{p*100:.1f}%")

                    # Verdict
                    if "OK" in verdict:
                        st.success(f"✅ {verdict}")
                    elif "ATTENTION" in verdict:
                        st.warning(f"⚠️ {verdict}")
                    else:
                        st.info(f"ℹ️ {verdict}")

                    # Reco finale
                    st.markdown(
                        f'<div class="reco-box">'
                        f'<p style="margin:0; color:#666;">Style recommandé :</p>'
                        f'<span class="style-badge">{reco.upper()}</span>'
                        f'<p style="margin-top:1rem; color:#444;">'
                        f'<strong>💡 Conseil relooking :</strong><br>'
                        f'{RELOOKING_TIPS.get(reco, "")}'
                        f'</p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # Cleanup
        try:
            tmp_path.unlink()
        except Exception:
            pass

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#999; font-size:0.85rem;'>"
    "MakeAIPrep · Projet Ydays · Powered by ResNet50 + CLIP + ❤️"
    "</div>",
    unsafe_allow_html=True,
)
