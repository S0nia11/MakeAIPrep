# MakeAIPrep

> Projet Ydays — Coach relooking IA pour adapter son look à chaque évènement.

Application web qui analyse une photo de visage, détecte le style, et génère un conseil relooking personnalisé (maquillage, coiffure, tenue) avec images des produits réels.

## Stack technique

- **CLIP+MLP** (PyTorch · open-clip) — classification de style à partir d'une photo + contexte texte
- **Gemini 2.5 Flash Vision** (Google) — génération de conseils personnalisés en JSON structuré
- **DuckDuckGo Image Search** — recherche d'images réelles des produits cités
- **Streamlit** — interface web

## Documentation

- 📖 **[SETUP.md](SETUP.md)** — Guide d'installation et de lancement step-by-step
- 📝 **[RAPPORT.md](RAPPORT.md)** — Documentation complète : architecture, choix techniques, benchmarks, pitch jury

## Quick start

```bash
git clone https://github.com/S0nia11/MakeAIPrep.git
cd MakeAIPrep
git checkout data
pip install -r requirements.txt
# Configurer la clé Gemini (voir SETUP.md étape 3)
python train_clip_mlp_fast.py --data-dir ./datasets/final  # entraîner le MLP
python -m streamlit run app.py
```

## Résultats

| Métrique | Valeur |
|---|---|
| Test accuracy | **100%** |
| F1 macro | **1.00** |
| Inférence ML | 256 ms |
| Train time | ~10 sec sur CPU |

Voir **[RAPPORT.md](RAPPORT.md)** pour le benchmark complet (CLIP+MLP vs ResNet50 vs ViT-B16 vs CLIP-ZeroShot).

## Auteurs

Projet étudiant Ydays.
