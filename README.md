# 🏠 Airbnb Paris — Prédiction du prix d'une nuitée

> **Projet final M102 · NEEKOCODE · IPSSI · Mai 2026**

Modèle de **régression** qui prédit le prix d'une nuitée Airbnb à Paris à partir des caractéristiques d'un logement (quartier, type, capacité, équipements, hôte, reviews).

---

## 🎯 Problème métier

Une plateforme de location courte durée veut **aider ses hôtes parisiens à fixer le bon prix** :
- **Trop bas** → manque à gagner
- **Trop haut** → l'annonce ne se loue pas

**Mission** : à partir des caractéristiques d'un logement (quartier, type, capacité, équipements, scores reviews, profil hôte), prédire un prix conseillé par nuit.

**Données** : [Inside Airbnb — Paris](https://insideairbnb.com/get-the-data/), dump `listings.csv` (~84 000 lignes × 79 colonnes), prix en `$X.YZ` (parsé en `float`).

---

## 🧰 Stack technique

| Outil | Version | Rôle |
|---|---|---|
| Python | 3.13 | runtime |
| pandas | 3.0 | data wrangling |
| scikit-learn | 1.8 | Pipeline, modèles, CV, GridSearch |
| matplotlib / seaborn | — | visualisations |
| FastAPI / uvicorn | — | API `/predict` (Bonus A) |
| joblib | — | sérialisation modèle |
| uv | — | gestion env & deps |

---

## ⚙️ Installation

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## ▶️ Exécution

### 1. Notebook complet (EDA → modèle → exports)

```bash
jupyter notebook Airbnb_Paris_Price.ipynb
# ou exécution headless qui régénère model.joblib + metrics.json :
jupyter nbconvert --to notebook --execute Airbnb_Paris_Price.ipynb \
    --output Airbnb_Paris_Price.ipynb --ExecutePreprocessor.timeout=1200
```

### 2. Démo : prédire le prix d'un cas concret

Une cellule du notebook prédit le prix pour : *Buttes-Montmartre, 2 chambres, capacité 4, balcon, superhost, instant book, hôte avec 2 listings, 5 ans d'ancienneté*.
Résultat actuel : **≈ 309 €/nuit** (fourchette ±MAE : ≈ 251 – 368 €).

### 3. API FastAPI (Bonus A)

```bash
uvicorn app:app --reload
# Swagger interactif :  http://localhost:8000/docs
# Smoke test :
curl -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{
        "accommodates": 4, "bedrooms": 2, "beds": 2, "bathrooms": 1,
        "minimum_nights": 2, "availability_365": 180, "number_of_reviews": 30,
        "review_scores_rating": 4.7, "review_scores_location": 4.8, "review_scores_cleanliness": 4.6,
        "host_response_rate": 95, "host_acceptance_rate": 90, "reviews_per_month": 2.0,
        "latitude": 48.886, "longitude": 2.343,
        "host_since_days": 1825, "host_total_listings_count": 2, "bedrooms_x_accommodates": 8,
        "neighbourhood_cleansed": "Buttes-Montmartre",
        "room_type": "Entire home/apt", "property_type": "Entire rental unit",
        "host_is_superhost": 1, "instant_bookable": 1,
        "has_wifi": 1, "has_kitchen": 1, "has_washer": 1, "has_tv": 1,
        "has_air_conditioning": 0, "has_elevator": 1, "has_balcony": 1, "has_free_parking": 0
      }'
# -> {"predicted_price_eur": 199.5, "model": "GradientBoostingRegressor", ...}
# (Le notebook utilise df['latitude'].median()/df['longitude'].median() (médian Paris global)
#  et renvoie ~309 €. Le curl ci-dessus utilise les coords précises de Buttes-Montmartre
#  (18ᵉ arr.) — plus excentré → prédiction plus basse. Les deux sont des sorties valides
#  du même modèle, montrant la sensibilité à la géolocalisation exacte.)
```

---

## 🔬 Méthodologie

### Données et cleaning

- **84 055 lignes** Inside Airbnb Paris → après cleaning : **52 883 logements** (drop des prix manquants + filtre **Q98** ≤ ~700 €/nuit — Q98 plutôt que Q99 pour exclure les hôtels de luxe / annonces frauduleuses qui plombent la MAE absolue)
- `price` parsé depuis `"$135.00"` → `float`
- `amenities` parsé depuis une string JSON-like → **8 features booléennes** (`has_wifi`, `has_kitchen`, `has_washer`, `has_tv`, `has_air_conditioning`, `has_elevator`, `has_balcony`, `has_free_parking`)
- `bathrooms_text` parsé en float (gère `"1 private bath"`, `"half-bath"`, …)
- `property_type` rabotté aux 15 plus fréquents (les autres → `"Other"`) pour éviter une explosion OHE

### Distribution du prix — log-transform sur la cible

Le prix est **fortement skewed à droite** → on utilise `TransformedTargetRegressor(func=log1p, inverse_func=expm1)` : le modèle apprend sur `log(1+price)`, mais les métriques (MAE, R²) sont calculées en € après inverse-transform. Cf. décision **D2** dans `decisions.md`.

### Pipeline anti-fuite

```
TransformedTargetRegressor(log1p / expm1)
  └─ Pipeline
       ├─ ColumnTransformer
       │     ├─ num : SimpleImputer(median) + StandardScaler   (15 features)
       │     ├─ cat : SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown='ignore')   (3 features)
       │     └─ bin : SimpleImputer(0)                          (10 features amenities + flags)
       └─ Modèle (LinearReg / Ridge / RandomForest / GBM)
```

**Anti-fuite garantie** : tout fit (impute, scale, OHE) se fait **uniquement** sur le train fold via `cross_validate` / `GridSearchCV`. Le test set (20 %, `random_state=42`) n'est touché qu'à la toute fin pour l'évaluation finale.

---

## 📊 Résultats — Comparaison 4 modèles (CV 5-fold, MAE en €)

| Modèle | R² (mean ± std) | MAE € (mean ± std) | Verdict |
|---|---|---|---|
| LinearRegression  | −456 846 ± 913 686 | 680.6 ± 1 204.1 | ❌ instable (cf. ci-dessous) |
| Ridge (RidgeCV α∈{0.1…1000}) | −431 759 ± 863 512 | 663.8 ± 1 170.6 | ❌ instable malgré CV interne |
| RandomForest (n=200, depth=20) | 0.632 ± 0.005 | 61.4 ± 0.8 | ✅ compétitif |
| **GradientBoosting** (n=400, depth=5, lr=0.1 — tuné) | **0.636 ± 0.010** | **61.0 ± 0.6** | ✅ **retenu** |

### ⚠️ Pourquoi les modèles linéaires échouent (même avec RidgeCV)

L'OHE de `neighbourhood_cleansed` (102 quartiers, dont beaucoup à <20 listings) + `property_type` (15) + `room_type` (4) génère ~120 colonnes catégorielles sparse. Avec une régression linéaire wrappée par `TransformedTargetRegressor(log1p, expm1)` :

1. Sur certains splits, le solveur ajuste des coefficients très grands sur des modalités rares.
2. `expm1` **amplifie exponentiellement** une erreur en `log(1+€)` → quelques prédictions à 10⁶ € qui font exploser MAE et tirent R² très négatif.

**On a essayé deux mitigations**, sans succès :
- `OneHotEncoder(min_frequency=20)` pour regrouper les modalités rares
- `RidgeCV(alphas=[0.1, 1, 10, 100, 1000])` pour auto-tuner α via CV interne

→ **Pourquoi ça ne marche pas** : RidgeCV optimise R² en **log-space** (sa cible interne), mais on évalue en **€** après `expm1`. La régularisation log-optimale n'est pas la même que la régularisation €-optimale.

→ **RandomForest et GBM sont structurellement robustes** : ils sont bornés à l'enveloppe convexe des prix d'entraînement → aucune prédiction aberrante possible.

C'est un excellent **point pédagogique de soutenance** : le choix de la métrique évaluée (€) et la métrique optimisée (log) doivent être cohérents.

### Métriques finales du modèle retenu (GradientBoosting tuné via GridSearchCV)

Hyperparamètres retenus par `GridSearchCV` : `n_estimators=400, max_depth=5, learning_rate=0.1`.

| Métrique | Valeur | Interprétation |
|---|---|---|
| **MAE test set** | **58.34 €** | « Le modèle se trompe en moyenne de ~58 €/nuit » |
| **R² test set** | **0.663** | ~66 % de la variance du prix expliquée |
| CV 5-fold MAE | 60.99 ± 0.64 € | stable sur les 5 folds |
| CV 5-fold R² | 0.636 ± 0.010 | stable sur les 5 folds |
| Train / Test | 42 306 / 10 577 | split 80/20 `random_state=42` |
| # features | **31** | 18 num + 3 cat (OHE+min_freq=20) + 10 bin |

**Évolution v1 → v2** : MAE 71 € → 58 € (−18 %), R² 0.59 → 0.66 (+0.07). Améliorations cumulées : filtre Q99→Q98 (les ~600 annonces à 700–2000 € qui plombaient la MAE absolue), 3 features additionnelles (`host_since_days`, `host_total_listings_count`, `bedrooms_x_accommodates`), `OneHotEncoder(min_frequency=20)`, `RidgeCV` (pour les linéaires — sans succès, cf. note ci-dessus).

📁 Toutes les métriques sont sauvegardées dans **`metrics.json`** au format imposé par l'énoncé.

### Où le modèle se trompe le plus ?

Analyse des résidus dans le notebook (section 7.1). MAE par tranche de prix réel sur le test set :

| Prix réel (€/nuit) | n logements | MAE | Erreur relative |
|---|---|---|---|
| 0–50 | 258 | 18 € | ~50 % |
| 50–100 | 2 274 | 23 € | ~30 % |
| 100–150 | 2 462 | 29 € | ~25 % |
| 150–250 | 2 722 | 44 € | ~22 % |
| 250–500 | 2 066 | 80 € | ~22 % |
| **500 +** | 795 | 257 € | ~45 % |

Le modèle est **homogène à ~22-30 % d'erreur relative sur 50–500 €** (le cœur du marché parisien), ce qui est honnête pour un conseil de prix. Il dérape sur les < 50 € (peu de signal, listings particuliers) et les > 500 € (luxe, peu d'exemples). C'est cohérent avec la nature `log` du problème : 22 % d'erreur sur 100 € fait 22 €, sur 500 € fait 110 €.

---

## 🎁 Bonus

### Bonus A — API FastAPI `/predict` ✅
Fichier `app.py`. Voir section *Exécution → 3.* ci-dessus. Endpoints :
- `POST /predict` — JSON avec les 28 features → `{"predicted_price_eur": float}`
- `GET /health` — ping + nom du modèle chargé
- `GET /docs` — Swagger UI interactif

### Bonus B — Augment LLM documenté ✅

L'assistant LLM (Claude Opus 4.7) a été mobilisé à **2 étapes** précises du projet :

#### Étape 1 — Choix des features amenities discriminantes

**Prompt utilisé** *(intégral)* :
> *« Tu as un dataset Airbnb Paris où chaque listing a une colonne `amenities` qui est une liste de ~30-50 strings (`["Wifi", "Hair dryer", "Free washer – In unit", "Kitchen", "Elevator", ...]`). Je veux dériver au moins 5 features booléennes qui ont un signal métier crédible pour le prix d'une nuitée. Donne-moi une liste de 8 amenities à extraire et explique en une phrase pourquoi chacune influence le prix. Robustesse : il faut matcher les variantes ("Wifi" vs "Free wifi" vs "Wifi at the property") via un keyword contains, pas un equality. »*

**Output reçu** *(résumé)* :
LLM propose : `has_wifi`, `has_kitchen` (autonomie longs séjours), `has_washer` (long séjour familles), `has_tv`, `has_air_conditioning` (été Paris), `has_elevator` (haussmannien sans ascenseur = -€), `has_balcony/terrace` (premium parisien), `has_free_parking` (rareté Paris = signal fort). Pour le matching, lowercase + `any(keyword in joined for keyword in [...])`.

**Décision prise** : **retenu intégralement**. Les 8 features sont implémentées dans le notebook (cellule §3.3) avec le dictionnaire `AMENITY_KEYWORDS`. Le matching robuste m'a permis de capturer ~80 % des variantes Inside Airbnb (« Free wifi », « Pocket wifi », « Wifi – 50 Mbps », …).

#### Étape 2 — Vulgarisation du MAE et choix de transformation cible

**Prompt utilisé** *(intégral)* :
> *« Mon MAE test est de 71 € sur des prix Airbnb Paris qui vont de 30 à 600 €/nuit avec une moyenne ~150 €. La distribution du prix est très skewed à droite. (1) Comment vulgariser ce chiffre pour un hôte non technique ? (2) Pourquoi `TransformedTargetRegressor(log1p, expm1)` est-il une meilleure idée ici qu'un RobustScaler ou une suppression d'outliers brutale ? Réponds en 4-5 lignes. »*

**Output reçu** *(résumé)* :
(1) « Le modèle se trompe en moyenne de 71 €/nuit en plus ou en moins » + souligner que l'erreur **n'est pas absolue** : sur les 80 €/nuit elle est plus petite, sur les 500 € elle est plus grande (homoscédasticité log). (2) `log1p` stabilise la variance (le bruit est multiplicatif sur les prix, pas additif), garde la skewness sans détruire les outliers légitimes (luxe), et `expm1` ramène l'inférence en €. Versus `RobustScaler` qui ne change pas la distribution de la cible (juste les features).

**Décision prise** : **retenu et adapté**. La vulgarisation MAE est intégrée mot pour mot dans le notebook section 7 et dans les slides. L'argumentaire `log1p` est repris dans `decisions.md` (D2) et dans la section *« Pourquoi les modèles linéaires échouent »* du README pour expliquer la fragilité du `expm1` sur les régresseurs non bornés.

> ⚠️ **L'auteur du projet est l'étudiant, pas le LLM.** Toute ligne de code générée avec assistance LLM a été reproduite à la main et est compréhensible pour l'auteur. La cellule §3.3 (parsing amenities) et la section §6 (CV scoring dict + nommage R²/MAE) sont les seules zones où l'assistance LLM a été déterminante ; le reste du notebook est code maison.

---

## 🚧 Limites assumées

1. **Biais de sélection** : on n'a que les annonces actives publiées sur Airbnb fin 2024 — pas les annonces refusées par le marché ni les short-let off-platform.
2. **Inside Airbnb retire `price` sur les dumps récents Paris** — il a fallu trouver un dump qui conservait encore la colonne. Reproductibilité limitée si Inside Airbnb supprime tout.
3. **OneHotEncoder `handle_unknown='ignore'`** : un quartier inconnu sera prédit avec un vecteur OHE nul → le modèle se rabat sur les autres features. Pas idéal pour de la nouvelle géographie.
4. **Pas de features texte** : `name`, `description`, `host_about` non utilisés → on rate sans doute du signal sur le positionnement (« cosy », « luxury », « charmant »).
5. **Pas de saisonnalité** : `price` est un snapshot, pas une moyenne calendaire — on ne capture pas les variations Olympiques / Fashion Week / Noël.
6. **MAE 58 € ≈ 35 % du prix moyen (165 €)** : honnête pour 31 features simples mais loin du niveau prod (cibler MAE < 30 € demanderait du feature engineering géographique et texte).

---

## 🔮 Perspectives

- **Features texte** : embeddings sur `name` + `description` (TF-IDF puis Truncated SVD, ou small model HuggingFace).
- **Géo-clustering** : k-means sur (lat, lon) pour créer des micro-zones (ex. 100 zones) plutôt que 102 quartiers administratifs souvent trop larges.
- **Features hôte** : ancienneté (`host_since` → days), volume (`host_total_listings_count`), historique des reviews.
- **Saisonnalité** : merger `calendar.csv.gz` pour avoir le prix moyen par mois, training un modèle prix journalier au lieu d'un snapshot.
- **CatBoost / LightGBM** : gèrent les catégorielles haute-cardinalité nativement, sans OHE → souvent +2 à +5 points de R² gratuits sur ce type de dataset.

---

## 📁 Structure du dépôt

```
airbnbAI/
├── Airbnb_Paris_Price.ipynb   # notebook principal (EDA + Pipeline + comparaison + démo)
├── app.py                     # API FastAPI /predict (Bonus A)
├── model.joblib               # modèle entraîné rechargeable
├── metrics.json               # métriques au format imposé
├── README.md                  # ce fichier
├── slides.md                  # outline soutenance (5-7 slides)
├── decisions.md               # journal des choix techniques
├── requirements.txt           # deps Python
├── listings.csv               # dataset Inside Airbnb Paris (non versionné)
└── Projet_Final_M102_Enonce.md
```

---

## 🎤 3 phrases pour défendre le projet en soutenance

1. **« On a construit un modèle qui prédit le prix d'une nuitée Airbnb à Paris à partir de 31 features (quartier, capacité, équipements, hôte, reviews, interactions), avec un MAE de 58 € sur 10 577 logements de test (R² 0.66). »**
2. **« Le pipeline complet — imputation, scaling, OHE, log-transform de la cible — est strictement anti-fuite : tout est fit uniquement sur le train fold, validé par CV 5-fold (R² 0.636 ± 0.010). »**
3. **« GradientBoosting bat les modèles linéaires non pas parce qu'il est meilleur dans l'absolu, mais parce qu'il est borné aux prix d'entraînement — même un RidgeCV qui tune α échoue, parce que la régularisation log-optimale ≠ la régularisation €-optimale après `expm1`. »**

---

*Made with ❤️ — IPSSI M1 · projet M102.*
