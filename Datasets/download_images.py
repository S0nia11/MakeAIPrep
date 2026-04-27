import os
from icrawler.builtin import BingImageCrawler

# =========================
# 1. REQUÊTES OPTIMISÉES
# =========================
queries = [
    # 👔 OUTFITS HOMMES
    "man business outfit full body office",
    "man formal suit full body professional",
    "man business casual outfit full body",
    "man corporate attire full body",

    # 👗 OUTFITS FEMMES
    "woman business outfit full body office",
    "woman formal outfit heels blazer full body",
    "woman business casual outfit full body",
    "woman corporate attire full body",

    # 💇 COIFFURES HOMMES
    "man professional haircut portrait",
    "man business hairstyle clean haircut",
    "short haircut man professional look",

    # 💇 COIFFURES FEMMES
    "woman professional hairstyle office",
    "woman business haircut clean look",
    "elegant hairstyle woman office"
    
    # 💄 MAKEUP FEMMES
    "woman professional makeup natural look",
    "woman office makeup subtle look",
    "business woman makeup elegant natural",
    "corporate makeup woman clean look",
    "soft glam makeup professional woman",
    "minimal makeup office woman",
    "natural beauty makeup woman portrait",
    "light makeup professional headshot woman"
]

# =========================
# 2. DOSSIER DE SORTIE
# =========================
BASE_DIR = "Datasets/images_downold"
os.makedirs(BASE_DIR, exist_ok=True)

# =========================
# 3. FILTRES ANTI-BRUIT
# =========================
EXCLUDE_TERMS = "-color -palette -background -vector -logo -drawing -sketch -cartoon"

# =========================
# 4. TÉLÉCHARGEMENT
# =========================
for q in queries:
    print(f"\n🔍 Téléchargement pour : {q}")

    folder_name = q.replace(" ", "_")
    save_path = os.path.join(BASE_DIR, folder_name)
    os.makedirs(save_path, exist_ok=True)

    crawler = BingImageCrawler(
        storage={'root_dir': save_path},
        downloader_threads=8
    )

    crawler.crawl(
        keyword=f"{q} {EXCLUDE_TERMS}",
        max_num=1000,          # 🔥 dataset LARGE
        min_size=(600, 800),   # meilleure qualité
        file_idx_offset=0
    )

print("\n✅ Téléchargement terminé dans images_downold !")