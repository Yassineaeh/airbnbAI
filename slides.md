# 🏠 Airbnb Paris — Prix d'une nuitée
*Outline soutenance · M102 · IPSSI · 6 slides*

---

## Slide 1 — Le problème métier (1 phrase)

> **Aider les hôtes Airbnb parisiens à fixer le bon prix de leur annonce — ni trop bas (manque à gagner), ni trop haut (annonce qui ne se loue pas).**

→ Régression : `price (€/nuit)` à partir de 28 features (quartier, capacité, équipements, hôte, reviews).

---

## Slide 2 — Données & EDA

- **Source** : Inside Airbnb · `listings.csv` Paris (snapshot 2024-2025)
- **Volume** : 84 055 listings × 79 colonnes → **52 883 après cleaning (filtre Q98)**
- **Cible** : `price` parsée depuis `"$135.00"` → `float`, distribution **skewed à droite** → `log1p` pour stabiliser
- **3 pièges techniques rencontrés** :
  1. `price` en string `"$X.YZ"` → parser
  2. `amenities` string JSON-like → 8 features booléennes (`has_wifi`, `has_kitchen`, `has_balcony`, …)
  3. Distribution skewed → `TransformedTargetRegressor(log1p, expm1)`
- **Visus du notebook à afficher** : histogramme brut vs log + boxplot quartier + heatmap corrélations

---

## Slide 3 — Architecture du Pipeline

```
TransformedTargetRegressor(log1p, expm1)
└── Pipeline
    ├── ColumnTransformer
    │   ├── num (15)  : SimpleImputer(median) + StandardScaler
    │   ├── cat (3)   : SimpleImputer(most_frequent) + OneHotEncoder
    │   └── bin (10)  : SimpleImputer(0)
    └── RandomForestRegressor / GBM / Ridge / LinearReg
```

**Anti-fuite garantie** : tout fit (impute/scale/OHE) se fait *uniquement sur train fold* via `cross_validate` + `GridSearchCV`. Test set figé `random_state=42`.

---

## Slide 4 — Résultats — Comparaison 4 modèles (CV 5-fold)

| Modèle | R² (mean ± std) | MAE € (mean ± std) |
|---|---|---|
| LinearRegression  | −456 846 ± 913 686 | 680.6 ± 1 204 |
| Ridge (RidgeCV α∈{0.1…1000}) | −431 759 ± 863 512 | 663.8 ± 1 170 |
| RandomForest  | 0.632 ± 0.005 | 61.4 ± 0.8 |
| **GradientBoosting** (n=400, depth=5, lr=0.1) | **0.636 ± 0.010** | **61.0 ± 0.6** |

→ **Test set final : MAE = 58.34 €  ·  R² = 0.663** (n=10 577, 31 features)

🔎 **Pourquoi Linear/Ridge explosent même avec RidgeCV** : RidgeCV optimise α en **log-space**, mais on évalue en **€** après `expm1`. La régularisation log-optimale ≠ €-optimale. Modèles tree-based sont **bornés** à l'enveloppe convexe des prix d'entraînement → robustes par construction.

---

## Slide 5 — Démo — Un cas concret 🎯

> *« Un hôte propose un logement à Buttes-Montmartre, 2 chambres, capacité 4, avec balcon, superhost, 5 ans d'ancienneté. Quel prix conseiller ? »*

```python
demo = {
  "neighbourhood_cleansed": "Buttes-Montmartre",
  "room_type": "Entire home/apt", "property_type": "Entire rental unit",
  "accommodates": 4, "bedrooms": 2, "bathrooms": 1,
  "has_balcony": 1, "has_wifi": 1, "has_kitchen": 1, "has_elevator": 1,
  "host_is_superhost": 1, "host_since_days": 1825, "host_total_listings_count": 2,
  "bedrooms_x_accommodates": 8, "review_scores_rating": 4.7, ...
}
model.predict(demo)  # -> 309 €
```

**Vulgarisation** : *« Le modèle conseille ~309 €/nuit. Compte tenu de la précision du modèle (±58 €), une fourchette raisonnable est 251–368 €. »*

→ Démo live via **API FastAPI** (`POST /predict` · Swagger UI) — bonus A.

---

## Slide 6 — Limites & perspectives

**Limites assumées** :
- MAE 58 € ≈ 35 % du prix moyen — honnête mais loin du niveau prod
- Pas de features texte (`name`, `description`) → on rate « cosy », « luxury », « charmant »
- Pas de saisonnalité (Olympiques, Fashion Week, Noël)
- Biais de sélection (annonces actives uniquement)
- OneHotEncoder ne sait pas généraliser à un nouveau quartier

**Perspectives concrètes** :
- Embeddings TF-IDF / SVD sur `description`
- Géo-clustering k-means sur (lat, lon) au lieu des 102 quartiers admin
- Merge `calendar.csv.gz` → features prix journalier saisonnier
- LightGBM / CatBoost : gèrent catégorielles haute-cardinalité nativement → +2-5 pts R² typiques

---

## Slide 7 (bonus) — Bonus A + B

- **Bonus A — FastAPI** : `app.py` · endpoint `POST /predict` · Swagger UI · démo curl OK
- **Bonus B — LLM documenté** : Claude Opus 4.7 utilisé à 2 étapes (choix amenities + vulgarisation MAE), prompts intégraux + outputs + décisions dans le README. Tout code généré est compris et reproductible à la main par l'auteur.

---

> Conversion PDF : `pandoc slides.md -o slides.pdf` (ou copier dans Google Slides / Canva).
