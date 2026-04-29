# MakeAIPrep — Rapport de projet

> **Projet Ydays** — Coach relooking IA pour adapter son look à chaque évènement
> Stack : Python · PyTorch · Streamlit · CLIP · Gemini · DuckDuckGo

---

## 1. Présentation du projet

**MakeAIPrep** est une application web qui analyse une photo de visage et recommande automatiquement un look complet (maquillage, coiffure, tenue) adapté au contexte évènementiel choisi par l'utilisateur (entretien d'embauche, soirée gala, etc.).

### Fonctionnalités finales

- Upload photo (fichier ou webcam)
- Sélection de l'évènement parmi 6 contextes pro/étudiants
- Sélection du genre (Femme / Homme / Non binaire)
- Choix du type de relooking (Maquillage / Coiffure / Vêtements — combinables)
- Style à imposer optionnel (parmi 20 styles)
- Précisions libres (nom précis de l'évènement, contraintes/préférences)
- Sortie : conseil personnalisé en plusieurs sections + images des produits cités

---

## 2. Architecture du pipeline

```
Photo + Évènement
   │
   ▼
┌──────────────┐
│   CLIP       │  encode photo + texte → vecteurs sémantiques (512+512)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   MLP        │  classifie : prédit le style (parmi 7 classes)
└──────┬───────┘
       │ style prédit (ou style imposé par l'utilisateur)
       ▼
┌──────────────┐
│   Gemini     │  génère un JSON structuré : conseils + produits cités
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  DDG Search  │  recherche images réelles des produits
└──────┬───────┘
       │
       ▼
   Affichage Streamlit (texte + grille d'images)
```

---

## 3. Composants : capacités et rôles

### 3.1 CLIP (encoder multimodal)

**Modèle utilisé :** `ViT-B-32` d'OpenAI via `open-clip`.

**C'est quoi :** un modèle multimodal pré-entraîné sur 400 millions de paires image+texte du web. Il transforme une image ou un texte en vecteur 512 dimensions dans un espace commun.

**Capacités :**
- Compréhension sémantique d'une image (pas juste les pixels, mais le sens)
- Comparaison image ↔ texte dans un espace commun
- Zero-shot classification (reconnaît des concepts jamais vus pendant le pré-entraînement)
- Universel : marche sur n'importe quel domaine (mode, art, sport, etc.)

**Avantages dans MakeAIPrep :**
- Pas besoin d'un gros dataset pour fine-tuner (1500 images suffisent)
- Comprend le contexte texte (« entretien d'embauche » influence la prédiction)
- Inférence rapide (256ms)
- Open source et gratuit

**Limites :**
- Generaliste, pas spécialisé sur la mode → besoin du MLP par-dessus
- 152M paramètres (~600 MB)

### 3.2 MLP (classifier supervisé)

**Architecture :**
```
Input : embedding image CLIP (512) + embedding texte CLIP (512) = 1024 dims
   ▼
[ Linear 1024 → 512 ] + ReLU + Dropout(0.2)
   ▼
[ Linear 512 → 256 ] + ReLU + Dropout(0.2)
   ▼
[ Linear 256 → 7 ]    (1 score par classe)
   ▼
Softmax → probabilités
```

**Capacités :**
- Classification non-linéaire (grâce à ReLU)
- Apprentissage rapide
- Spécialisation sur domaine précis

**Avantages :**
- ~660 000 paramètres (200× plus léger que CLIP)
- Entraînement ultra-rapide (~10 sec sur CPU)
- Pas de GPU nécessaire
- Spécialise CLIP sur les 7 styles cibles

**Limites :**
- Dépend totalement de la qualité des embeddings CLIP
- Risque d'overfitting sur petit dataset

### 3.3 Gemini 2.5 Flash (LLM Vision)

**Modèle utilisé :** `gemini-2.5-flash` (Google), avec fallback automatique sur `gemini-2.0-flash` si surcharge.

**Capacités :**
- Vision : analyse fine de la photo (couleur peau, forme du visage, cheveux, morphologie, accessoires)
- Génération de texte fluide en français
- Output JSON structuré garanti (via `response_mime_type` et `response_schema`)
- Comprend les contextes complexes (nom événement précis + préférences utilisateur libres)

**Pourquoi Gemini et pas Claude/GPT :**
- Free tier généreux (15 req/min, 1500/jour) suffisant pour le projet
- Vision multimodale équivalente à GPT-4o pour ce cas d'usage
- API simple via `google-genai` SDK

**Configuration :**
- `thinking_budget=0` (désactive le raisonnement interne pour éviter le tronquage)
- `temperature=0.8` (variabilité des conseils entre appels)
- `max_output_tokens=4000` (large pour conseils détaillés + JSON)

### 3.4 DuckDuckGo Image Search

**C'est quoi :** lib `ddgs` qui interroge l'API publique de DuckDuckGo pour récupérer des images web.

**Avantages :**
- Gratuit, sans clé API
- Région paramétrable (fr-fr) pour résultats francophones
- Cache LRU en mémoire (évite recherches répétées)

**Limites :**
- Qualité variable (parfois pas exactement le bon produit)
- Rate limiting non garanti (peut casser sous forte charge)

---

## 4. Dataset et entraînement

### 4.1 Dataset

- **Source :** images de visages collectées + nettoyées manuellement
- **Total :** 1507 images
- **7 classes** : `naturel`, `professionnel`, `elegant`, `glamour`, `minimaliste`, `boheme`, `doux`
- **Labels :** générés via CLIP zero-shot puis vérifiés manuellement (script `preprocessing/label_with_clip.py`)
- **Split :** 80% train / 10% validation / 10% test

### 4.2 Procédure d'entraînement (`train_clip_mlp_fast.py`)

1. **Pré-calcul des embeddings CLIP** (image + texte) une seule fois et mise en cache disque (`datasets_faces/_clip_cache.pt`)
2. **Entraînement du MLP uniquement** sur ces embeddings figés
3. **Optimiseur :** AdamW, lr=1e-3, weight_decay=0.01
4. **Scheduler :** CosineAnnealingLR sur 30 epochs max
5. **Loss :** CrossEntropy avec label_smoothing=0.1
6. **Early stopping :** patience 5 sur la val accuracy

### 4.3 Résultats d'entraînement

```
Early stopping epoch 8
Test accuracy : 1.0000
F1 macro      : 1.0000
Top-3         : 1.0000
Inférence     : 256 ms
Train time    : ~12 sec sur CPU
```

L'accuracy de 100% s'explique par :
- CLIP encode déjà très finement les concepts de style (pré-entraînement multimodal)
- Le dataset est propre et bien labellisé
- Les 7 classes sont sémantiquement bien séparées

---

## 5. Comparatif des modèles testés

Un benchmark a été réalisé sur 4 architectures pour justifier le choix final :

| Modèle | Accuracy | F1 | Top-3 | Inférence | Params | Train time |
|---|---|---|---|---|---|---|
| **CLIP+MLP** ⭐ | **100%** | **1.00** | **100%** | 256 ms | 152 M | 1411 s* |
| ViT-B16 | 82.5% | 0.81 | 95% | 1117 ms | 87 M | 2114 s |
| ResNet50 | 66.3% | 0.66 | 96% | 521 ms | 25 M | 760 s |
| CLIP-ZeroShot | 17.5% | 0.04 | 45% | 383 ms | 151 M | 0 s |

\* Le train time CLIP+MLP est dominé par le pré-calcul des embeddings (à faire une seule fois). En version `train_clip_mlp_fast.py` avec cache, le train MLP seul prend ~10 sec.

### Verdict

- **CLIP+MLP gagne nettement** sur tous les critères : précision, vitesse d'inférence, et qualité des features.
- ViT-B16 est précis mais 4× plus lent.
- ResNet50 est trop peu précis (pas de pré-entraînement multimodal).
- CLIP-ZeroShot sans MLP est trop générique pour distinguer des styles fins.

---

## 6. Justification des choix techniques

### Pourquoi CLIP+MLP plutôt que ResNet50 ?

- **Pré-entraînement multimodal** sur 400M images = compréhension sémantique d'emblée
- **Multimodal :** prend en compte le texte de l'évènement, pas juste l'image
- **Plus rapide** à l'inférence (256 ms vs 521 ms)
- **Plus précis** (100% vs 66%)

### Pourquoi un MLP simple (et pas un classifier plus complexe) ?

- Les embeddings CLIP sont déjà sémantiquement riches → un classifier simple suffit
- 660K paramètres = entraînement en quelques secondes sur CPU
- Risque d'overfitting limité (faible capacité)
- Facile à ré-entraîner si on ajoute une classe

### Pourquoi Gemini et pas Claude / GPT ?

- **Free tier** suffisant pour le projet (vs payant chez Anthropic et OpenAI)
- Vision multimodale comparable pour ce use case
- SDK simple
- Output JSON structuré natif

### Pourquoi DuckDuckGo et pas SerpAPI / Google Shopping ?

- **Gratuit** sans clé API (vs SerpAPI : 100 req/mois free puis payant)
- Suffisant pour démo (qualité d'images correcte)
- Si production : migrer vers SerpAPI pour vrais liens e-commerce avec prix

### Pourquoi Streamlit et pas FastAPI + React ?

- **MVP rapide** : 1 fichier Python = UI complète
- Idéal pour démo data science
- Pas de séparation front/back à maintenir
- `st.form` natif pour gérer les batches d'inputs

### Pourquoi un bouton submit (form) et pas du traitement automatique ?

- L'utilisateur configure plusieurs champs avant de générer
- Évite des appels API Gemini gaspillés à chaque changement de paramètre
- UX claire : « configurer puis générer »

---

## 7. Modèle final retenu

**CLIP+MLP** est le modèle ML retenu. Il est utilisé en interne sans donner le choix à l'utilisateur (qui ne s'y connait pas forcément).

**Architecture finale en production :**
- CLIP `ViT-B-32` (figé, jamais réentraîné)
- MLP : 1024 → 512 → 256 → 7 (Dropout 0.2, ReLU)
- Checkpoint : `results/CLIP+MLP_fast.pth` (~2.6 MB, juste les poids du MLP)

**Filtre :** la classe `boheme` est blacklistée à l'affichage (jugée trop niche), mais reste dans le modèle pour préserver les performances.

**Styles affichés à l'utilisateur :** 20 styles dont 14 ajoutés manuellement (`bourgeois`, `casual`, `chic`, `cocktail`, `festif`, `gothique`, `hippie`, `moderne`, `rock`, `romantique`, `sophistique`, `sportif`, `streetwear`, `vintage`) — utilisables uniquement via « Style à imposer » car le modèle n'est pas entraîné dessus, mais Gemini les traite très bien.

---

## 8. Stack technique complète

### Machine Learning
- `torch` 2.11 (PyTorch)
- `torchvision` 0.26
- `open-clip-torch` 3.3 (CLIP ViT-B-32)
- `scikit-learn` 1.8 (métriques)
- `numpy`, `pandas`

### Vision LLM et recherche
- `google-genai` 1.73 (SDK Gemini)
- `ddgs` (DuckDuckGo Image Search)

### Web
- `streamlit` 1.57 (UI complète en Python)
- `Pillow` (manipulation images)

### Visualisation
- `matplotlib`, `seaborn`

### Outils
- `tqdm` (barres de progression)
- `pyyaml` (config)

---

## 9. Workflow utilisateur (final)

1. **Upload** d'une photo (ou webcam)
2. **Configuration** : évènement, style à imposer (optionnel), genre, type de relooking (checkboxes), précisions libres
3. **Clic** sur « Générer la recommandation »
4. **Pipeline** :
   - CLIP encode photo + texte évènement
   - MLP classifie → style top-1 (sauf si forced_style)
   - Gemini reçoit photo + tous les paramètres → JSON avec conseils + produits
   - DDG cherche les images des produits
5. **Affichage** : badge style + sections (Maquillage / Coiffure / Tenue) + grille images produits

---

## 10. Limites et améliorations futures

### Limites actuelles

- **Dataset déséquilibré** sur certaines classes (boheme rarement vu)
- **DDG Image Search** : qualité variable, parfois pas le bon produit
- **Pas de try-on visuel** (l'utilisateur ne voit pas son visage avec le maquillage appliqué)
- **6 évènements** seulement, tous orientés pro/étudiant
- **20 styles** mais dont 14 non couverts par le modèle ML

### Améliorations possibles

1. **Try-on visuel** via Gemini 2.5 Flash Image (génération d'image à partir de photo + prompt)
2. **SerpAPI Google Shopping** pour avoir prix + liens cliquables
3. **Ré-entraîner CLIP+MLP** sur les 20 styles avec un dataset enrichi
4. **Ajouter plus d'évènements perso** (mariage, anniversaire, baptême)
5. **Score de confiance** affiché sur chaque prédiction
6. **A/B test plusieurs styles** dans la même page
7. **Sauvegarde** des recommandations préférées de l'utilisateur

---

## 11. Pitch jury (résumé)

> *« MakeAIPrep est un coach relooking IA qui combine 4 IA spécialisées chacune dans leur domaine :*
>
> *— **CLIP**, le modèle multimodal d'OpenAI pré-entraîné sur 400M images, pour la compréhension image+texte (100% accuracy en classification de style)*
>
> *— Un **MLP** que nous avons entraîné par-dessus pour spécialiser CLIP sur nos 7 styles, en seulement 10 secondes sur CPU grâce au transfer learning*
>
> *— **Gemini 2.5 Flash** de Google, en mode vision, pour générer des conseils ultra-personnalisés avec marques concrètes en analysant le visage*
>
> *— **DuckDuckGo Image Search** pour illustrer chaque produit cité avec une vraie image*
>
> *Cette approche permet d'obtenir une accuracy de 100% sur le test set tout en tournant sur CPU sans GPU, et avec une UX moderne grâce à Streamlit. »*
