# MakeAIPrep — Guide d'installation et lancement

> Step-by-step pour faire tourner le projet à partir d'un clone Git frais.

---

## 0. Prérequis

| Outil | Version | Note |
|---|---|---|
| **Python** | 3.11, 3.12 ou 3.14 | 3.10 ou inférieur peut poser souci sur PyTorch récent |
| **Git** | Récent | Pour cloner le repo |
| **Espace disque** | ~3 GB | PyTorch + CLIP pèsent lourd |
| **Connexion internet** | Requise | Pour télécharger CLIP au 1er run + appels API Gemini + recherche images DDG |
| **Compte Google** | Requis | Pour la clé API Gemini gratuite |

⚠️ **Pas de GPU nécessaire** — tout tourne sur CPU.

---

## 1. Cloner le repo

```powershell
git clone https://github.com/S0nia11/MakeAIPrep.git
cd MakeAIPrep
git checkout data
```

> 💡 La branche principale du projet est **`data`**, pas `main`.

---

## 2. Installer les dépendances Python

```powershell
pip install -r requirements.txt
```

Cela installe :
- `torch`, `torchvision`, `open-clip-torch` → modèle CLIP
- `streamlit` → UI web
- `google-genai` → API Gemini
- `ddgs` → recherche d'images DuckDuckGo
- + autres dépendances (numpy, pandas, scikit-learn, etc.)

⏱️ **Durée** : 3 à 10 min selon connexion (PyTorch fait ~110 MB).

> 💡 Si tu vois des warnings genre `script tqdm.exe is installed in 'C:\...\Scripts' which is not on PATH` → ignore. On lance l'app via `python -m streamlit` qui contourne le PATH.

---

## 3. Configurer la clé API Gemini

L'app utilise **Google Gemini Vision** pour générer les conseils relooking personnalisés. C'est gratuit avec un compte Google (free tier généreux : 15 req/min, 1500/jour).

### a) Obtenir une clé API

1. Va sur **https://aistudio.google.com/app/apikey**
2. Connecte-toi avec un compte Google
3. Clique **"Créer une clé API"** (en haut à droite)
4. Choisis **"Créer dans un nouveau projet"**
5. Copie la clé qui commence par `AIzaSy...`

> ⚠️ Si tu vois "permission denied", utilise un compte perso (pas un compte Google Workspace d'entreprise — l'admin IT bloque souvent l'accès).

### b) Placer la clé dans le projet

Copier le template :

```powershell
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
```

Éditer le fichier `secrets.toml` avec ta clé :

```toml
GOOGLE_API_KEY = "AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
```

> 🛡️ `secrets.toml` est listé dans `.gitignore` → la clé ne sera **jamais** push sur GitHub.

---

## 4. Récupérer ou entraîner le modèle ML

Le checkpoint `.pth` du modèle CLIP+MLP est **exclu du Git** (trop gros, ignoré dans `.gitignore`). Tu as 2 options :

### Option A — Ré-entraîner localement (~5-10 min sur CPU)

```powershell
python train_clip_mlp_fast.py --data-dir ./datasets/final
```

Ce script :
1. Télécharge CLIP ViT-B-32 depuis HuggingFace (~600 MB, fait UNE seule fois)
2. Calcule les embeddings CLIP de toutes les photos (~2 min)
3. Cache ces embeddings dans `datasets/final/_clip_cache.pt`
4. Entraîne le MLP par-dessus (~10 sec)
5. Sauvegarde le checkpoint final dans `results/CLIP+MLP_fast.pth`

### Option B — Récupérer un checkpoint existant

Si un coéquipier t'a partagé un `.pth` (Drive, WeTransfer, clé USB), place-le ici :

```
results/CLIP+MLP_fast.pth
```

---

## 5. Lancer l'app

```powershell
python -m streamlit run app.py
```

L'app s'ouvre automatiquement sur **http://localhost:8501**.

> 💡 On utilise `python -m streamlit` (et pas juste `streamlit`) pour éviter les soucis de PATH Windows.

### Premier lancement

Streamlit te demandera ton email pour des newsletters → **laisse vide et appuie Entrée**.

---

## 6. Test fonctionnel

Une fois l'app ouverte, tu dois voir :

- ✅ Header avec logo **MakeAIPrep** (étincelle cyan/turquoise) et navigation
- ✅ Sous-titre **"Téléchargez votre photo pour une analyse IA"**
- ✅ Sidebar à gauche avec **Styles disponibles** (20 styles)
- ✅ Zone d'upload photo + form (Genre, Évènement, Style à imposer, Type de relooking, Précisions)
- ✅ Bouton **"Lancer l'analyse"** en vert chartreuse pâle
- ✅ Footer avec icônes Facebook / Instagram / Twitter

### Workflow de test

1. Upload une photo de visage (JPG / PNG, < 200 MB)
2. Choisis : Évènement = `Entretien d'embauche`, Genre = `Femme`, Type de relooking = tout coché
3. Clique **"Lancer l'analyse"**
4. Tu vois 3 spinners successifs :
   - "Analyse ML en cours..." (~1 sec)
   - "Génération du conseil personnalisé..." (~2-3 sec)
   - "Recherche des produits pour maquillage..." (~5-15 sec)
5. Résultat affiché : badge cyan du style + sections Maquillage / Coiffure / Tenue avec texte + grille d'images des produits cités

---

## 🐛 Troubleshooting

### `streamlit: command not found`

```powershell
pip install streamlit
# Puis lance avec :
python -m streamlit run app.py
```

### `clip_mlp : pas de checkpoint trouvé`

→ Étape 4 manquée. Lance `python train_clip_mlp_fast.py --data-dir ./datasets/final`.

### `Pas de clé API Gemini configurée`

→ Le fichier `.streamlit/secrets.toml` n'existe pas ou ne contient pas `GOOGLE_API_KEY`. Refais l'étape 3.

> ⚠️ Streamlit ne recharge **PAS** automatiquement `secrets.toml`. Après modif → **Ctrl+C puis relancer** `python -m streamlit run app.py`.

### `Erreur API Gemini : 503 UNAVAILABLE`

→ Gemini est temporairement surchargé. Le code fait automatiquement :
1. 3 retries avec backoff (1s, 2s)
2. Fallback sur `gemini-2.0-flash`
3. Fallback sur `gemini-flash-latest`

Si toujours rouge → attendre 5-10 min ou réessayer plus tard. Le fallback statique (conseils texte sans images) sera affiché en attendant.

### Le bouton "Lancer l'analyse" ne change rien

→ Vérifier qu'une photo est bien uploadée (preview affichée). Sans photo, le form ne génère rien.

### Les images des produits ne s'affichent pas

→ DuckDuckGo Image Search peut être rate-limité. Réessayer dans 1-2 min. Le texte du conseil s'affiche quand même.

### Photo upload ne marche pas

→ Vérifier format (JPG / PNG / JPEG) et taille (< 200 MB).

### Erreur Python 3.14 incompatible

→ Si pip refuse certaines dépendances, créer un venv Python 3.12 :

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 📁 Structure du projet

```
MakeAIPrep/
├── app.py                     # UI Streamlit (point d'entrée)
├── predict.py                 # Inférence ML (CLIP+MLP)
├── static_tips.py             # Dictionnaires de conseils statiques (fallback)
├── llm_advisor.py             # Appel Gemini API + structured JSON output
├── image_search.py            # Recherche d'images via DuckDuckGo
├── train_clip_mlp_fast.py     # Script d'entraînement rapide du MLP
├── train.py                   # Script d'entraînement complet (ResNet, ViT, etc.)
│
├── requirements.txt           # Dépendances pip
├── RAPPORT.md                 # Documentation complète du projet
├── SETUP.md                   # Ce fichier
├── README.md                  # Description courte du projet
├── .gitignore                 # Exclusions git
│
├── .streamlit/
│   ├── config.toml            # Thème de l'app (couleurs)
│   ├── secrets.toml.example   # Template pour la clé Gemini
│   └── secrets.toml           # Clé API (NON versionnée, à créer)
│
├── models/
│   ├── config.yaml            # Config ML (classes, hyperparamètres)
│   ├── clip_mlp_recommender.py  # Architecture du MLP
│   ├── clip_zeroshot.py       # CLIP zero-shot (baseline)
│   ├── resnet_classifier.py   # ResNet50 (alternative testée)
│   ├── vit_classifier.py      # ViT-B16 (alternative testée)
│   ├── trainer.py             # Trainer générique
│   ├── dataset.py             # Dataset PyTorch
│   ├── metrics.py             # Métriques d'éval
│   └── visualization.py       # Plots
│
├── datasets/                  # Pipeline data complet
│   ├── README.md              # Explique raw -> cleaned -> final
│   ├── download_images.py     # Script de téléchargement
│   ├── raw/                   # 1. Images brutes (par prompt)
│   ├── cleaned/               # 2. Images nettoyées manuellement
│   └── final/                 # 3. Dataset final pour le ML
│       ├── images/            # Photos par classe (boheme, doux, elegant...)
│       ├── metadata.csv       # Labels
│       └── _clip_cache.pt     # Cache embeddings CLIP (généré, NON versionné)
│
├── results/
│   ├── CLIP+MLP_fast.pth      # Modèle entraîné (NON versionné)
│   ├── *.json                 # Métriques par modèle
│   ├── *.png                  # Graphiques benchmark
│   └── benchmark_summary.txt  # Résumé comparatif
│
├── preprocessing/             # Scripts de préparation des données
│   ├── label_with_clip.py     # Labellisation auto via CLIP zero-shot
│   ├── prepare_data.py        # Split train/val/test
│   └── cluster_faces.py       # Clustering des visages
│
├── notebooks/                 # Notebooks d'analyse exploratoire
└── docs/                      # Slides présentation (PDF, PPTX)
```

---

## 🔗 Documents associés

- **`RAPPORT.md`** : documentation complète du projet (archi, choix techniques, benchmarks, pitch jury)
- **`README.md`** : présentation rapide du projet

---

## ⚙️ Commandes utiles

```powershell
# Lancer l'app
python -m streamlit run app.py

# Ré-entraîner le modèle
python train_clip_mlp_fast.py --data-dir ./datasets/final

# Test ML uniquement (sans UI)
python predict.py --image chemin/vers/photo.jpg --event "entretien d'embauche"

# Vider le cache Streamlit (si bug d'affichage)
# Dans l'app : menu burger en haut à droite > "Clear cache"

# Vérifier que les dépendances sont OK
python -c "import torch, streamlit, google.genai, open_clip, ddgs; print('OK')"
```

---

Bon courage pour faire tourner le projet ! 🚀
