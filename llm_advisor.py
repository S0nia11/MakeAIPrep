"""
MakeAIPrep - Conseil relooking personnalise via Gemini Vision
==============================================================

Analyse une photo + contexte (evenement, sexe, style) et genere
un conseil relooking adapte (maquillage / coiffure / vetements).

Usage:
    from llm_advisor import generate_advice
    text = generate_advice(pil_image, style="professionnel",
                           event="entretien d'embauche",
                           gender="homme", focus="les_deux")
"""

import os
import time
from io import BytesIO
from typing import Optional

from PIL import Image


# Modeles Gemini par ordre de preference (premier = meilleur, dernier = fallback)
MODEL_FALLBACK_CHAIN = [
    "gemini-2.5-flash",  # Plus recent, meilleure qualite
    "gemini-2.0-flash",  # Fallback robuste, large dispo
    "gemini-flash-latest",  # Alias toujours pointant sur le dernier flash dispo
]

# Codes HTTP a retry (transitoires)
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


SYSTEM_PROMPT = """Tu es un coach relooking expert, specialise dans l'adaptation du look au contexte professionnel et evenementiel.

Regles strictes :
- Reponds en francais, sans emoji, sans preambule, sans formule de politesse.
- Tiens compte des elements visibles sur la photo : couleur de peau, forme du visage, couleur et coupe des cheveux, morphologie si visible, accessoires deja portes (lunettes, bijoux).
- Adapte tes conseils au sexe/genre indique :
  * Femme/autre : maquillage detaille, coiffure feminine, tenue feminine
  * Homme : pas de maquillage, mais soin de peau / barbe / sourcils, coupe masculine, tenue masculine
- Sois concret, personnalise et actionnable.
- Cite TOUJOURS des marques concretes :
  * Maquillage : 1-2 references abordables (Maybelline, L'Oreal, NYX, Sephora Collection, Bourjois) + 1 reference premium si pertinent (MAC, Charlotte Tilbury, Nars, Dior, Chanel)
  * Coiffure / produits coiffants : Schwarzkopf, L'Oreal Elnett, GHD, Babyliss, Kerastase, Olaplex
  * Soins homme : Nivea Men, L'Oreal Men Expert, Bulldog, Horace, Le Baigneur
  * Vetements : 1-2 enseignes accessibles (Zara, Mango, H&M, Uniqlo, Asos, COS) + 1 marque plus haut de gamme si justifie (Sandro, Maje, Sezane, The Kooples, Hugo Boss)
- Adapte le niveau de gamme au contexte : entretien d'embauche etudiant = accessible, gala = peut monter en gamme.
- 3 a 4 phrases par section, denses et utiles.
- Format strict avec sections en majuscules suivies de deux-points et d'un retour a la ligne.

Sections possibles selon ce qui est demande :

MAQUILLAGE :
[3-4 phrases : teint, yeux, levres, sourcils + couleurs et marques adaptees au teint/yeux visibles]

SOIN ET BARBE :
(uniquement si personne = homme et maquillage demande)
[3-4 phrases : routine soin peau + barbe/rasage + sourcils + produits cites par marque]

COIFFURE :
[3-4 phrases : coiffage adapte a la longueur/texture/forme du visage + produits cites par marque]

TENUE :
[3-4 phrases : pieces precises (chemise, blazer, jean, robe, chaussures) + couleurs + matieres + enseignes citees + accessoires]

Varie tes formulations a chaque appel - n'utilise pas de phrases generiques recyclees."""


def build_focus_instruction(focus_areas, gender: str) -> str:
    """Construit l'instruction de focus pour le LLM selon les sections demandees."""
    sections = []
    if "maquillage" in focus_areas:
        sections.append("SOIN ET BARBE" if gender == "homme" else "MAQUILLAGE")
    if "coiffure" in focus_areas:
        sections.append("COIFFURE")
    if "vetements" in focus_areas:
        sections.append("TENUE")
    if not sections:
        sections = ["MAQUILLAGE", "COIFFURE", "TENUE"]
    if len(sections) == 1:
        return f"Donne UNIQUEMENT la section {sections[0]}. Pas de section supplementaire."
    return f"Donne UNIQUEMENT ces sections, dans cet ordre : {', '.join(sections)}. Aucune autre section."


def _encode_image(img: Image.Image) -> bytes:
    """Convertit une PIL Image en JPEG bytes (compresse)."""
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def get_api_key() -> Optional[str]:
    """Recupere la cle API Gemini depuis Streamlit secrets ou env var."""
    try:
        import streamlit as st
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


def _is_retryable_error(exc: Exception) -> bool:
    """Determine si l'exception merite un retry (5xx, 429, ou erreurs reseau)."""
    msg = str(exc).upper()
    if "UNAVAILABLE" in msg or "OVERLOADED" in msg or "RESOURCE_EXHAUSTED" in msg:
        return True
    for code in RETRYABLE_STATUSES:
        if str(code) in msg:
            return True
    return False


def generate_advice(
    image: Image.Image,
    style: str,
    event: str,
    gender: str,
    focus_areas,
    event_name: Optional[str] = None,
    user_description: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_retries: int = 2,
) -> str:
    """
    Genere un conseil relooking personnalise via Gemini Vision.

    Strategie de robustesse :
    1. Essaie le modele demande (defaut: gemini-2.5-flash)
    2. En cas de 503/429/etc, retry avec backoff
    3. Si toujours echec, fallback sur gemini-2.0-flash puis gemini-flash-latest

    Args:
        image: PIL Image du visage de la personne.
        style: Style cible (ex: 'professionnel', 'elegant').
        event: Evenement (ex: "entretien d'embauche").
        gender: 'homme' | 'femme' | 'autre'.
        focus: 'maquillage' | 'vetements' | 'les_deux'.
        api_key: Cle API Gemini (optionnel, sinon lue depuis env/secrets).
        model: Nom du modele Gemini (defaut: chaine de fallback complete).
        max_retries: Nombre de retry par modele en cas d'erreur transitoire.

    Returns:
        Le texte du conseil genere.

    Raises:
        ValueError: si pas de cle API.
        Exception: si tous les modeles de la chaine echouent.
    """
    from google import genai
    from google.genai import types

    key = api_key or get_api_key()
    if not key:
        raise ValueError(
            "Cle API Gemini manquante. Mets-la dans .streamlit/secrets.toml "
            "(GOOGLE_API_KEY = '...') ou dans la variable d'env GOOGLE_API_KEY."
        )

    client = genai.Client(api_key=key)
    img_bytes = _encode_image(image)

    # Construit le prompt utilisateur en incluant les infos optionnelles si presentes
    prompt_parts = [
        f"Personne : {gender}",
        f"Type d'evenement : {event}",
    ]
    if event_name and event_name.strip():
        prompt_parts.append(f"Nom precis de l'evenement : {event_name.strip()}")
    prompt_parts.append(f"Style cible : {style}")
    if user_description and user_description.strip():
        prompt_parts.append(
            f"Precisions / preferences de la personne : {user_description.strip()}\n"
            "Tiens compte de ces preferences dans tes recommandations."
        )
    prompt_parts.append(build_focus_instruction(focus_areas, gender))
    user_prompt = "\n".join(prompt_parts)

    # Desactiver le mode "thinking" de Gemini 2.5 Flash (consomme des tokens
    # invisibles qui rognent sur la reponse visible) et donner un budget large.
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=2000,
        temperature=0.8,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    contents = [
        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
        user_prompt,
    ]

    # Determine la chaine de modeles a essayer
    models_to_try = [model] if model else MODEL_FALLBACK_CHAIN

    last_exc: Optional[Exception] = None
    for current_model in models_to_try:
        for attempt in range(max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=current_model,
                    contents=contents,
                    config=config,
                )
                text = (response.text or "").strip()
                if text:
                    return text
                # Reponse vide -> on traite comme erreur retryable
                last_exc = RuntimeError(f"Reponse vide de {current_model}")
            except Exception as e:
                last_exc = e
                if not _is_retryable_error(e):
                    # Erreur non-transitoire (auth, quota epuise, etc) : on tente le modele suivant
                    break
                if attempt < max_retries:
                    # Backoff exponentiel : 1s, 2s
                    time.sleep(2 ** attempt)

    raise last_exc if last_exc else RuntimeError("Tous les modeles Gemini ont echoue")
