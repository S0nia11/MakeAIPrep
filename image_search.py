"""
Recherche d'images de produits via DuckDuckGo (gratuit, sans cle API).

Utilise pour illustrer les conseils de relooking : pour chaque produit cite
par le LLM (rouge a levres, jean, blazer...), on recupere une image du web.
"""

import functools
from typing import List, Optional, Dict


# Domaines a eviter (resultats non pertinents : math, code, encyclopedie...)
BLACKLIST_DOMAINS = (
    "wikipedia.org", "wiktionary.org", "mathworld",
    "stackexchange", "stackoverflow", "github.io",
    "pinterest.com",  # souvent moodboards, pas le vrai produit
)

# Domaines e-commerce / beauty preferes (URL avec ces domaines = boost prioritaire)
PREFERRED_DOMAINS = (
    "sephora", "marionnaud", "nocibe", "fnac", "amazon",
    "zalando", "asos", "zara", "hm.com", "mango", "uniqlo",
    "cos.com", "sandro-paris", "maje", "sezane", "kooples",
    "lookfantastic", "feelunique", "monoprix", "naturalia",
    "fenty", "loreal", "maybelline", "nyx", "mac-cosmetics",
    "douglas", "yves-rocher", "kiko", "bourjois", "dior",
    "chanel", "lvmh", "ghd-hair", "babyliss",
    "ztat.net",  # CDN Zalando
    "scene7.com",  # CDN multi-retailers
    "cdn.shopify",  # boutiques Shopify
)


def _is_valid_image_url(url: str, timeout: float = 2.5) -> bool:
    """Verifie rapidement qu'une URL pointe vers une image accessible
    avec dimensions raisonnables (pas une banniere extra-large).
    """
    import requests
    try:
        r = requests.head(
            url,
            timeout=timeout,
            allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if r.status_code != 200:
            return False
        ct = r.headers.get("Content-Type", "").lower()
        return ct.startswith("image/")
    except Exception:
        return False


def _has_weird_ratio(width: int, height: int) -> bool:
    """Detecte les ratios anormaux : bannieres, illustrations panoramiques,
    icones minuscules. Une vraie photo produit est ~carre (0.6 a 1.7).
    """
    if not width or not height:
        return False
    if width < 200 or height < 200:
        return True  # trop petit, probablement icone ou thumbnail
    ratio = width / height
    if ratio > 2.0 or ratio < 0.4:
        return True  # banniere ou format extrement vertical
    return False


def _is_preferred_url(url: str) -> bool:
    """True si l'URL contient un domaine e-commerce / beauty connu."""
    url_lower = url.lower()
    return any(d in url_lower for d in PREFERRED_DOMAINS)


@functools.lru_cache(maxsize=500)
def _search_single_query(query: str, region: str = "fr-fr",
                          strict_ecommerce: bool = True) -> Optional[str]:
    """Cherche une image pour UNE query. Retourne URL valide ou None.

    Si strict_ecommerce=True, ne retourne QUE les URLs sur domaines preferes.
    Sinon, accepte tout domaine non-blackliste.
    """
    if not query or not query.strip():
        return None
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.images(
                query.strip(),
                region=region,
                safesearch="moderate",
                size="Medium",
                type_image="photo",
                max_results=15,
            ))

        # Filtrer : retirer blacklist + ratios anormaux
        candidates = []
        for r in results:
            url = r.get("image") or r.get("thumbnail")
            if not url:
                continue
            if any(bad in url.lower() for bad in BLACKLIST_DOMAINS):
                continue
            try:
                w = int(r.get("width") or 0)
                h = int(r.get("height") or 0)
            except (TypeError, ValueError):
                w, h = 0, 0
            if _has_weird_ratio(w, h):
                continue
            candidates.append(url)

        # En mode strict : ne garder que les URLs e-commerce
        if strict_ecommerce:
            candidates = [u for u in candidates if _is_preferred_url(u)]
        else:
            # Sinon : trier en privilegiant les e-commerce
            candidates.sort(key=lambda u: 0 if _is_preferred_url(u) else 1)

        # Validation HEAD : retourner la 1ere URL qui repond OK
        for url in candidates:
            if _is_valid_image_url(url):
                return url
    except Exception as e:
        print(f"[image_search] Erreur DDG pour '{query}': {e}")
    return None


def search_product_image(queries, region: str = "fr-fr") -> Optional[str]:
    """
    Cherche une image en testant plusieurs queries successivement.

    Strategie :
    1. Essaie chaque query en mode STRICT e-commerce uniquement
    2. Si aucune query ne donne de resultat strict, retourne None
       (preferable a une image non pertinente)

    Args:
        queries: liste de strings (ou un seul string pour compat ascendante)
        region: Region DDG (fr-fr).

    Returns:
        URL d'une image produit sur un site e-commerce, ou None.
    """
    # Compat ascendante : si on passe un string, le convertir en liste
    if isinstance(queries, str):
        queries = [queries]
    if not queries:
        return None

    # Pass 1 : strict e-commerce uniquement
    for q in queries:
        url = _search_single_query(q, region=region, strict_ecommerce=True)
        if url:
            return url

    # Pass 2 (fallback) : on assouplit, accepte tout domaine non-blackliste
    # Commente par defaut car cause souvent les incoherences signalees.
    # Decommenter si trop de produits restent sans image.
    # for q in queries:
    #     url = _search_single_query(q, region=region, strict_ecommerce=False)
    #     if url:
    #         return url

    return None


def search_products_batch(products: List[Dict]) -> List[Dict]:
    """
    Cherche les images pour une liste de produits.

    Args:
        products: liste de dicts. Supporte 2 formats :
            - {"name": ..., "brand": ..., "queries": [q1, q2, q3]}  (preferable)
            - {"name": ..., "brand": ..., "query": ...}              (legacy)

    Returns:
        Meme liste enrichie avec un champ "image_url" (peut etre None).
    """
    enriched = []
    for p in products:
        queries = p.get("queries") or []
        if not queries:
            # Fallback sur ancien format
            q = p.get("query")
            if q:
                queries = [q]
            else:
                # Construit une query de base depuis name + brand
                base = f"{p.get('brand', '')} {p.get('name', '')}".strip()
                queries = [base] if base else []
        url = search_product_image(queries) if queries else None
        enriched.append({**p, "image_url": url})
    return enriched
