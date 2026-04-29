# Datasets — Pipeline data MakeAIPrep

Ce dossier contient toutes les données du projet, organisées selon le pipeline de préparation **raw → cleaned → final**.

```
Téléchargement web ──▶ raw/ ──▶ cleaned/ ──▶ final/ ──▶ Modèle ML
                                  (humain)    (script labelling)
```

---

## Structure

```
datasets/
├── README.md                # Ce fichier
├── download_images.py       # Script qui scrape des images du web
│
├── raw/                     # Étape 1 : images brutes telles que téléchargées
│   ├── business_casual_outfit_man_full_body/
│   ├── corporate_attire_woman.../
│   ├── elegant_professional_outfit.../
│   ├── ...                  # Un sous-dossier par prompt de recherche
│   └── metadata.csv         # URLs et prompts d'origine
│
├── cleaned/                 # Étape 2 : images nettoyées manuellement
│   ├── 000006.jpg           # Filtrage humain : photos floues / non pertinentes retirées
│   ├── 000007.jpg
│   └── ...                  # Numérotation séquentielle, prêt pour labelling
│
└── final/                   # Étape 3 : dataset final utilisé par le modèle ML
    ├── images/              # Organisé par classe sémantique
    │   ├── boheme/
    │   ├── doux/
    │   ├── elegant/
    │   ├── glamour/
    │   ├── minimaliste/
    │   ├── naturel/
    │   └── professionnel/
    ├── metadata.csv         # Labels (image_path, classe)
    └── _clip_cache.pt       # Cache embeddings CLIP (généré, NON versionné)
```

---

## Pipeline détaillé

### 1. Raw (`raw/`)

Images téléchargées depuis le web par `download_images.py`. Une recherche par prompt produit un sous-dossier (ex: `business_casual_outfit_man_full_body/`).

**Re-générer** :
```bash
python datasets/download_images.py
```

⚠️ Les images sont brutes : qualité variable, pas filtrées, parfois non pertinentes (logos, bannières publicitaires, photos floues).

### 2. Cleaned (`cleaned/`)

Filtrage manuel des images brutes : on garde uniquement celles qui montrent clairement un visage / une tenue exploitable. Les images sont renumérotées séquentiellement (`000006.jpg`, `000007.jpg`, etc.).

⚠️ Étape humaine, pas automatisée. Les images retirées : floues, watermarks, illustrations dessinées, plusieurs personnes, etc.

### 3. Final (`final/`)

Dataset prêt pour l'entraînement ML. Les images de `cleaned/` ont été labellisées en 7 classes sémantiques via le script `preprocessing/label_with_clip.py` (CLIP zero-shot) puis vérifiées manuellement.

**Structure attendue par le modèle** :
```
final/
└── images/
    ├── <classe_1>/
    │   ├── img001.jpg
    │   └── ...
    ├── <classe_2>/
    └── ...
```

C'est ce dossier qu'utilisent :
- `train_clip_mlp_fast.py --data-dir ./datasets/final`
- `train.py --data-dir ./datasets/final`
- `app.py` (chargement runtime)

Le fichier `_clip_cache.pt` est créé au 1er entraînement pour éviter de recalculer les embeddings CLIP à chaque epoch (gain : ~50-100×).

---

## Statistiques

| Niveau | Contenu | Images | Versionné Git ? |
|---|---|---|---|
| `raw/` | Images brutes par prompt | quelques milliers | ⚠️ Selon taille |
| `cleaned/` | Images filtrées manuellement | ~1500 | ✅ Oui |
| `final/images/` | 1500 images réparties en 7 classes | 1507 | ✅ Oui |
| `final/_clip_cache.pt` | Cache embeddings CLIP | 1 fichier | ❌ NON (gitignore) |

---

## Modifier le dataset

### Ajouter une nouvelle classe

1. Créer un dossier `final/images/<nouvelle_classe>/` et y copier les images
2. Mettre à jour `models/config.yaml` → `style_categories` et `event_to_style`
3. Re-lancer l'entraînement : `python train_clip_mlp_fast.py --data-dir ./datasets/final --no-cache`

### Ajouter des images à une classe existante

1. Ajouter les images dans `final/images/<classe>/`
2. Re-lancer avec `--no-cache` pour invalider le cache embeddings

### Re-labelliser depuis cleaned/

```bash
python preprocessing/label_with_clip.py --src datasets/cleaned --dst datasets/final
```
