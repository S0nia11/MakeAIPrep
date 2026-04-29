"""
MakeAIPrep - Conseils statiques par style et descriptions d'evenements
=======================================================================

Dictionnaires de conseils utilises comme fallback quand l'API Gemini n'est pas
disponible (pas de cle API, erreur reseau, etc.). Importes par predict.py et
app.py.

Pour chaque style semantique :
- MAKEUP_TIPS : conseils maquillage (3-4 phrases)
- HAIR_TIPS   : conseils coiffure (3-4 phrases)
- CLOTHES_TIPS: conseils tenue vestimentaire (3-4 phrases)

EVENT_DESCRIPTIONS : description courte du contexte evenementiel, matchee par
mot-cle via "if key in event.lower()" pour adapter le ton de la reco.
"""


# Conseils maquillage par style semantique (alignes avec label_with_clip.py)
MAKEUP_TIPS = {
    "naturel":       "Teint frais et lumineux, mascara leger, gloss transparent, sourcils brosses.",
    "professionnel": "Peau matifiee, eyeliner discret, levres nude mate, sourcils structures.",
    "elegant":       "Smoky eye doux, teint sophistique, levres bordeaux ou rose profond.",
    "glamour":       "Eyeliner marque, faux cils, contouring, levres rouges ou nude glossy.",
    "minimaliste":   "Peau nue lumineuse, mascara seul, baume teinte. Less is more.",
    "doux":          "Blush rose, eyeshadow pastel, levres rose poudre, mascara brun.",
    "moderne":       "Eyeliner graphique ou color block, teint glowy, levres mates tendance.",
    "gothique":      "Teint pale, eyeliner noir epais, smoky eye charbon, levres bordeaux ou prune fonce, "
                     "sourcils marques.",
    "vintage":       "Eyeliner pin-up en aile, levres rouge mat profond, peau matifiee, "
                     "eyeshadow nude rose, look annees 50-60.",
    "streetwear":    "Teint glowy naturel, mascara noir, baume teinte, sourcils naturels brosses, "
                     "look frais et casual urbain.",
    "romantique":    "Blush pommettes rose poudre, eyeshadow rose ou peche, levres rose framboise glossy, "
                     "mascara brun, look feminin et tendre.",
    "sophistique":   "Teint travaille au pinceau, contouring leger, eyeshadow taupe ou nude profond, "
                     "eyeliner tres fin, levres rose nude mat, sourcils impeccables.",
    "chic":          "Peau lumineuse, mascara seul ou eyeliner discret, levres rose vieux ou nude rose, "
                     "blush peche, look raffine et intemporel.",
    "festif":        "Paillettes ou shimmer sur paupiere mobile, eyeliner accentue, levres rouge ou prune, "
                     "highlighter intense, look party impactant.",
    "casual":        "Baume teinte, mascara leger, blush a peine visible, sourcils brosses, "
                     "look minimal et naturel pour tous les jours.",
    "sportif":       "Peau saine et nette (BB cream legere), mascara waterproof, baume hydratant, "
                     "sourcils brosses, zero complexite, look athletique.",
    "rock":          "Smoky eye charbon ou noir, eyeliner appuye, levres bordeaux ou nude mat, "
                     "sourcils marques, look intense et affirme.",
    "cocktail":      "Teint glowy, eyeliner discret, smoky eye doux, levres rouge profond ou nude glossy, "
                     "blush rose, look feminin soigne pour soiree.",
    "hippie":        "Teint dore au bronzer, eyeshadow terreux, mascara naturel, levres nude ou peche, "
                     "blush peche, look libre et solaire.",
    "bourgeois":     "Peau parfaite et hydratee, eyeshadow nude beige, mascara discret, "
                     "levres rose nude mat, sourcils naturellement structures, luxe discret.",
}

# Conseils coiffure par style semantique
HAIR_TIPS = {
    "naturel":       "Coiffure naturelle sans exces de produit, mouvement libre, "
                     "ondulations douces ou queue-de-cheval lache.",
    "professionnel": "Coiffure tiree, chignon bas ou queue-de-cheval nette, "
                     "raie sur le cote, look soigne et credible.",
    "elegant":       "Coiffure ondulee ou chignon haut sophistique, "
                     "accessoires raffines (barrette doree, peigne discret).",
    "glamour":       "Wavy hair travaille, volume aux racines, brushing impactant, "
                     "ondulations larges type tapis rouge.",
    "minimaliste":   "Coiffure simple : raie au milieu, cheveux lisses ou queue-de-cheval haute, "
                     "zero artifice, lignes nettes.",
    "doux":          "Coiffure souple et romantique, ondulations legeres, "
                     "demi-attache avec ruban ou epingle delicate.",
    "moderne":       "Coupe structuree (carre net, frange micro), wet look, "
                     "ou coiffure sleek avec gel pour effet contemporain.",
    "gothique":      "Cheveux noirs lisses ou crepus volumineux, raie au milieu, "
                     "ou coiffure crantee avec accessoires metalliques sombres.",
    "vintage":       "Victory rolls, ondulations Hollywood des annees 40-50, "
                     "chignon banane, ou queue-de-cheval haute avec foulard.",
    "streetwear":    "Cheveux laches au naturel, queue-de-cheval haute decontractee, "
                     "ou casquette/bandana, look spontane et urbain.",
    "romantique":    "Boucles douces, demi-attache avec ruban, cheveux ondules feminins, "
                     "tresse couronne ou epingles fleurs delicates.",
    "sophistique":   "Brushing impeccable, chignon bas tres lisse, ou queue-de-cheval basse "
                     "tres polie avec raie sur le cote, finition glossy.",
    "chic":          "Coiffure lisse et nette : carre droit, queue-de-cheval mi-haute lisse, "
                     "ou ondulations tres legeres, raie sur le cote.",
    "festif":        "Volume travaille, ondulations larges glamour, ou chignon haut destructure "
                     "avec accessoires brillants (barrettes strass, peigne dore).",
    "casual":        "Cheveux laches naturels, queue-de-cheval lache, chignon decoiffe, "
                     "ou bun haut decontracte sans produits.",
    "sportif":       "Queue-de-cheval haute serree, tresse plaquee ou demi-chignon haut, "
                     "front degage, look fonctionnel et propre.",
    "rock":          "Coupe asymetrique, cheveux mi-longs effiles, frange droite epaisse, "
                     "shaggy cut, ou wet look gel-back affirme.",
    "cocktail":      "Brushing impeccable, ondulations souples, demi-attache elegante avec "
                     "epingle dorée, ou chignon bas raffine.",
    "hippie":        "Cheveux longs ondules au naturel, tresses laterales ou couronne, "
                     "headband fleur ou bandeau ethnique, aspect non force.",
    "bourgeois":     "Brushing soigne, raie sur le cote impeccable, chignon bas tres ordonne, "
                     "ou cheveux longs lisses parfaitement entretenus, look discret.",
}

# Conseils vetements par style semantique
CLOTHES_TIPS = {
    "naturel":       "Tenue simple et confortable. Couleurs neutres (beige, blanc casse, kaki). "
                     "Matieres naturelles (lin, coton). Coupe decontractee mais soignee, "
                     "baskets blanches ou mocassins.",
    "professionnel": "Blazer cintre, chemise sobre (blanche, bleu clair), pantalon de tailleur "
                     "ou jupe crayon. Couleurs neutres (noir, gris, marine, beige). "
                     "Chaussures fermees, eviter motifs voyants et logos.",
    "elegant":       "Coupe ajustee, matieres nobles (laine, soie, satin). Robe fluide ou "
                     "tailleur structure. Couleurs profondes (bordeaux, marine, emeraude). "
                     "Accessoires raffines : sac a main, bijoux discrets, escarpins.",
    "glamour":       "Robe moulante ou tailleur audacieux, paillettes, sequins ou velours. "
                     "Decollete assume, talons hauts. Accessoires forts : bijoux brillants, "
                     "pochette, rouge a levres assorti.",
    "minimaliste":   "Lignes epurees, pas de logos, matieres premium. Total look monochrome "
                     "(blanc, noir, gris, beige). Coupes nettes, accessoires quasi invisibles. "
                     "Less is more : 1 ou 2 pieces fortes max.",
    "doux":          "Couleurs pastel (rose poudre, bleu ciel, lavande), matieres douces "
                     "(coton, mohair, mousseline). Coupes feminines : jupes mi-longues, "
                     "blouses fluides, details volants ou dentelle.",
    "moderne":       "Coupes audacieuses, asymetrie, color blocking. Matieres techniques "
                     "(cuir, vinyle, mesh). Couleurs vives ou metalliques. "
                     "Sneakers tendance ou bottines, accessoires statement.",
    "gothique":      "Total black ou tons sombres (bordeaux, violet profond, gris anthracite). "
                     "Cuir, dentelle noire, velours, mesh. Bottines cloutees ou Doc Martens, "
                     "ceintures a chaines, bijoux argent (croix, pentacles, chokers).",
    "vintage":       "Coupes retro annees 50-60-70 : robe trapeze, jupe corolle, jean taille haute, "
                     "blouse a col Claudine. Imprimes pois, fleurs, motifs geometriques. "
                     "Mocassins, escarpins kitten heels, sac vintage en cuir patine.",
    "streetwear":    "Sweat oversize, hoodie, jogging premium, jean baggy, cropped top. "
                     "Couleurs neutres avec touches vives. Sneakers tendance (Nike, Adidas, New Balance), "
                     "casquette, sac banane ou tote bag, accessoires logo.",
    "romantique":    "Robe fluide a fleurs, blouse a manches bouffantes, jupe midi plissee. "
                     "Couleurs douces (rose poudre, blanc casse, lilas, bleu ciel). Dentelle, "
                     "broderie anglaise. Ballerines, sandales tressees, sac en paille ou pochette satin.",
    "sophistique":   "Tailleur structure, robe fourreau noire, blouse en soie, pantalon cigarette. "
                     "Couleurs neutres haut de gamme (noir, marine, taupe, ivoire). Matieres premium "
                     "(soie, cachemire, laine fine). Escarpins fins, sac en cuir lisse.",
    "chic":          "Pantalon droit, blazer cintre, chemise de qualite, robe portefeuille. "
                     "Palette neutre raffinee (beige, blanc, noir, marine). Accessoires intemporels : "
                     "ballerines plates, mocassins, sac structure, foulard en soie.",
    "festif":        "Robe paillettes, top sequins, jupe satin, combinaison eclat. "
                     "Couleurs metallisees (or, argent, cuivre) ou tons profonds (bordeaux, emeraude). "
                     "Talons hauts, pochette a paillettes, bijoux brillants statement.",
    "casual":        "Jean classique, T-shirt blanc, sweat basique, sneakers, baskets. "
                     "Couleurs neutres et basiques. Pieces simples Uniqlo, H&M, COS, Levi's. "
                     "Confortable et passe-partout.",
    "sportif":       "Legging technique, brassiere, sneakers de course, sweat zip, jogging premium. "
                     "Marques : Nike, Adidas, Lululemon, Under Armour, On Running. "
                     "Look athleisure pour ville ou sport, casquette ou bonnet en option.",
    "rock":          "Jean noir slim, t-shirt rock, perfecto en cuir, pantalon vinyle. "
                     "Tons noir / gris / rouge fonce. Marques : Saint Laurent, IRO, The Kooples, "
                     "Zadig & Voltaire. Bottines a clous ou Doc Martens, ceinture cloutee.",
    "cocktail":      "Robe cocktail mi-longue, tailleur jupe, top satin avec pantalon de tailleur. "
                     "Couleurs noir, rouge, marine, emeraude. Matieres satin, dentelle, crepe. "
                     "Escarpins, pochette glamour, bijoux raffines (collier en or, boucles d'oreilles).",
    "hippie":        "Robe longue fluide, jupe gypsy, blouse paysanne, jean pat'd'eph. "
                     "Tons chauds (terracotta, ocre, kaki, creme), motifs ethniques et fleurs. "
                     "Sandales tressees, sac frange, bijoux ethniques superposes.",
    "bourgeois":     "Pull en cachemire beige, chemise blanche, pantalon en lin, jupe plissee marine. "
                     "Mocassins, sac en cuir patine, foulard. Materials premium (cachemire, soie, lin, "
                     "laine) sans logos. Marques : Sezane, Maje, Sandro, Massimo Dutti.",
}

# Alias pour compatibilite ascendante
RELOOKING_TIPS = MAKEUP_TIPS


# Description courte du contexte evenementiel.
# Le matching se fait par mot-cle via "if key in event.lower()".
EVENT_DESCRIPTIONS = {
    "entretien":     "Sobriete, credibilite, peu d'artifice. Le maquillage doit rassurer pas distraire.",
    "premier jour":  "Premier jour de travail : look soigne, naturel, premiere impression sans en faire trop.",
    "presentation":  "Look net qui inspire confiance. Eviter ce qui peut detourner l'attention.",
    "reunion":       "Professionnel et soigne, sans tomber dans le formel excessif.",
    "after work":    "Look chic mais decontracte, transition bureau-soiree, un cran au-dessus du quotidien.",
    "gala":          "Glamour assume, on ose les couleurs profondes et les details.",
    "soiree":        "Plus libre, on peut accentuer yeux ou levres selon le code.",
}
