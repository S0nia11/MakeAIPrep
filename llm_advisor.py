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
- Reponds en francais, sans emoji.
- Tiens compte des elements visibles sur la photo : couleur de peau, forme du visage, couleur et coupe des cheveux, morphologie si visible, accessoires deja portes (lunettes, bijoux).
- Adapte tes conseils au sexe/genre indique :
  * Femme/autre : maquillage, coiffure feminine, tenue feminine
  * Homme : pas de maquillage classique, mais soin de peau / barbe / sourcils, coupe masculine, tenue masculine
- Sois concret, personnalise et actionnable.
- Cite des marques concretes :
  * Maquillage : Maybelline, L'Oreal, NYX, Sephora Collection, Bourjois, MAC, Charlotte Tilbury, Nars, Dior, Chanel
  * Coiffure : Schwarzkopf, L'Oreal Elnett, GHD, Babyliss, Kerastase, Olaplex
  * Soins homme : Nivea Men, L'Oreal Men Expert, Bulldog, Horace, Le Baigneur
  * Vetements : Zara, Mango, H&M, Uniqlo, Asos, COS, Sandro, Maje, Sezane, The Kooples, Hugo Boss, Levi's
- Adapte le niveau de gamme au contexte : etudiant = accessible, gala = peut monter en gamme.

REPONDS UNIQUEMENT EN JSON, suivant ce schema :
{
  "sections": [
    {
      "type": "MAQUILLAGE" | "SOIN_BARBE" | "COIFFURE" | "TENUE",
      "title": "Titre lisible affiche a l'utilisateur (ex: Maquillage, Coiffure, Tenue, Soin et Barbe)",
      "text": "3-4 phrases denses, personnalisees, citant les marques. C'est le conseil principal.",
      "products": [
        {
          "name": "Nom du produit (ex: Rouge a levres Ruby Woo, Jean 501, Blazer cintre noir)",
          "brand": "Marque (ex: MAC, Levi's, Zara)",
          "query": "Texte de recherche image optimise (ex: 'MAC Ruby Woo rouge a levres', 'Levi 501 jean homme noir', 'Zara blazer noir cintre femme'). Doit donner un seul produit precis dans une recherche Google Images."
        }
      ]
    }
  ]
}

Pour chaque section :
- 3 a 5 produits max dans products
- Le champ "query" est crucial : il sert a chercher une image du produit. Utilise marque + nom + type d'objet + couleur + genre si pertinent. Eviter les ambiguites.
- N'incluez que les sections demandees.

Varie tes formulations a chaque appel."""


def build_focus_instruction(focus_areas, gender: str) -> str:
    """Construit l'instruction de focus pour le LLM selon les sections demandees."""
    sections = []
    if "maquillage" in focus_areas:
        sections.append("SOIN_BARBE" if gender == "homme" else "MAQUILLAGE")
    if "coiffure" in focus_areas:
        sections.append("COIFFURE")
    if "vetements" in focus_areas:
        sections.append("TENUE")
    if not sections:
        sections = ["MAQUILLAGE", "COIFFURE", "TENUE"]
    return (
        f"Inclus UNIQUEMENT ces types de sections dans le JSON: {', '.join(sections)}. "
        f"Aucune autre section."
    )


def _response_schema():
    """Schema JSON force pour la reponse du LLM."""
    from google.genai import types
    return {
        "type": "OBJECT",
        "properties": {
            "sections": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "type": {"type": "STRING"},
                        "title": {"type": "STRING"},
                        "text": {"type": "STRING"},
                        "products": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "name": {"type": "STRING"},
                                    "brand": {"type": "STRING"},
                                    "query": {"type": "STRING"},
                                },
                                "required": ["name", "brand", "query"],
                            },
                        },
                    },
                    "required": ["type", "title", "text", "products"],
                },
            }
        },
        "required": ["sections"],
    }


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
) -> dict:
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
    # response_mime_type=application/json + response_schema force un output JSON valide.
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=4000,
        temperature=0.8,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json",
        response_schema=_response_schema(),
    )

    contents = [
        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
        user_prompt,
    ]

    # Determine la chaine de modeles a essayer
    models_to_try = [model] if model else MODEL_FALLBACK_CHAIN

    import json
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
                    try:
                        data = json.loads(text)
                        if isinstance(data, dict) and "sections" in data:
                            return data
                    except json.JSONDecodeError:
                        pass
                # Reponse vide ou non-JSON -> on traite comme erreur retryable
                last_exc = RuntimeError(f"Reponse invalide de {current_model}")
            except Exception as e:
                last_exc = e
                if not _is_retryable_error(e):
                    # Erreur non-transitoire (auth, quota epuise, etc) : on tente le modele suivant
                    break
                if attempt < max_retries:
                    # Backoff exponentiel : 1s, 2s
                    time.sleep(2 ** attempt)

    raise last_exc if last_exc else RuntimeError("Tous les modeles Gemini ont echoue")
