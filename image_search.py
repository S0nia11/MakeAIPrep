"""
Recherche d'images de produits via DuckDuckGo (gratuit, sans cle API).

Utilise pour illustrer les conseils de relooking : pour chaque produit cite
par le LLM (rouge a levres, jean, blazer...), on recupere une image du web.
"""

import functools
from typing import List, Optional, Dict


@functools.lru_cache(maxsize=300)
def search_product_image(query: str, region: str = "fr-fr") -> Optional[str]:
    """
    Cherche une image pour le produit donne et retourne l'URL de la 1ere.

    Cache LRU pour eviter les recherches repetees pendant la session.

    Args:
        query: Texte de recherche (ex: "MAC Ruby Woo lipstick").
        region: Region DDG (fr-fr pour resultats francais).

    Returns:
        URL de la 1ere image trouvee, ou None si echec.
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
                max_results=3,
            ))
            for r in results:
                url = r.get("image") or r.get("thumbnail")
                if url:
                    return url
    except Exception as e:
        print(f"[image_search] Erreur DDG pour '{query}': {e}")
    return None


def search_products_batch(products: List[Dict[str, str]]) -> List[Dict]:
    """
    Cherche les images pour une liste de produits.

    Args:
        products: liste de dicts {"name": ..., "brand": ..., "query": ...}

    Returns:
        Meme liste enrichie avec un champ "image_url" (peut etre None).
    """
    enriched = []
    for p in products:
        query = p.get("query") or f"{p.get('brand', '')} {p.get('name', '')}".strip()
        url = search_product_image(query) if query else None
        enriched.append({**p, "image_url": url})
    return enriched
