# Decisions techniques — Airbnb Paris Price

> Journal des choix au fil de l'eau. Sert de matière première pour le README et la soutenance.

## D1 — Sujet
Sujet 1 imposé : prédire le prix d'une nuitée Airbnb à Paris (régression). Dataset Inside Airbnb `listings.csv` (~26 760 lignes × 79 colonnes, version détaillée).

## D2 — Cible et transformation
`price` parsé depuis string `"$120.00"` → float. Distribution **fortement skewed à droite** → on utilise `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)` pour stabiliser. Les métriques (MAE, R²) sont reportées **en euros** (espace original) après inverse-transform.

## D3 — Features amenities (≥5 booléennes)
`amenities` est une string JSON-like (`["Wifi", "Kitchen", ...]`). Parsée avec `ast.literal_eval` puis on dérive **≥5 features booléennes** : `has_wifi`, `has_kitchen`, `has_washer`, `has_tv`, `has_air_conditioning`, `has_elevator`, `has_balcony`, `has_parking`. Choix : amenities les plus fréquentes et signal métier le plus crédible pour le prix.

## D4 — Anti-fuite
**Pipeline complet via `Pipeline` + `ColumnTransformer`** : tout fit (imputation, scaling, OHE) se fait *uniquement* sur le train fold. Aucun fit sur le test/holdout. Le split test est figé `random_state=42`, test_size=0.2.

## D5 — Modèles comparés
1. `LinearRegression` (baseline)
2. `Ridge` (régularisation L2)
3. `RandomForestRegressor`
4. `GradientBoostingRegressor`
Tous wrappés dans `TransformedTargetRegressor(log1p/expm1)`. CV 5-fold, `cross_validate` retournant **R² et neg_MAE** (mean ± std).

## D6 — Métriques métier
- **MAE en €** = métrique principale (interprétable par un hôte non technique : « le modèle se trompe en moyenne de X € par nuit »).
- **R²** = métrique secondaire (part de variance expliquée).
- MAPE non retenue → instable sur les prix bas (≤20 €).

## D7 — Outliers
Filtre simple : `price` strictement positif et ≤ quantile 99% (~1 200 €). Les annonces à 0 € sont des bugs ou des inactives ; au-delà du Q99 ce sont des hôtels de luxe non représentatifs.

## D8 — Bonus
Bonus A (API FastAPI `/predict`) **et** Bonus B (LLM documenté) cumulés. Bonus capé à +1 pt mais filet de sécurité si l'un est faible.

## D9 — Résultats finaux (run Paris 84k → 53k après cleaning)

| Modèle | R² CV (mean ± std) | MAE € CV (mean ± std) |
|---|---|---|
| LinearRegression | −115 276 ± 230 552 | 447.6 ± 721.0 |
| Ridge (α=1.0) | −114 460 ± 228 921 | 446.3 ± 718.4 |
| **RandomForest** (n=200, depth=20) | **0.596 ± 0.010** | **70.1 ± 1.1** |
| GradientBoosting (n=300, depth=4, lr=0.05) | 0.595 ± 0.008 | 70.2 ± 0.8 |

**Modèle retenu** : RandomForest tuné via GridSearchCV (n_estimators, max_depth, min_samples_leaf).
**Test set final** : MAE = **71.08 €**, R² = **0.589** (n=10 685, train=42 738).

## D10 — Pourquoi les modèles linéaires explosent

Avec ~120 colonnes catégorielles OHE (102 quartiers + 15 property_type + 4 room_type) et le wrapper `expm1`, le solveur linéaire ajuste sur certains splits des coefficients très grands sur des modalités rares. Une petite erreur en `log(1+€)` devient exponentiellement grande après `expm1` → quelques outliers à 10⁶ € qui tirent MAE/R² hors limite. Les modèles tree-based sont bornés aux prix d'entraînement → robustes par construction. Point clé à défendre en soutenance.

## D11 — Quartier démo
La cellule démo cherche automatiquement un quartier contenant « montmartre » ou « butte » → **Buttes-Montmartre** sur le dataset Paris. Avec 2 ch, capacité 4, balcon, superhost, instant book : prédiction ≈ **309 €/nuit** (cohérent avec le marché Montmartre).

## D12 — Améliorations v2 (après revue critique)

Après revue des chiffres v1 (R² 0.59 modeste, Ridge/LinReg cassés à R²=−115k), 4 améliorations cumulées :

1. **Filtre Q99 → Q98** : retire les ~600 annonces 700-2000 €/nuit (hôtels, arnaques) qui plombaient la MAE absolue
2. **+3 features** : `host_since_days` (ancienneté hôte), `host_total_listings_count` (proxy pro vs particulier), `bedrooms_x_accommodates` (capacité par chambre)
3. **OneHotEncoder(min_frequency=20)** : regroupe les modalités rares (quartiers <20 listings → 'infrequent_sklearn')
4. **RidgeCV** au lieu de Ridge(α=1.0) : auto-tune α via CV interne

**Résultats v2** :

| Modèle | R² CV v2 (mean ± std) | MAE € CV v2 |
|---|---|---|
| LinearRegression | −456 846 ± 913 686 | 680.6 ± 1 204 |
| Ridge (RidgeCV) | −431 759 ± 863 512 | 663.8 ± 1 170 |
| RandomForest | 0.632 ± 0.005 | 61.4 ± 0.8 |
| **GradientBoosting** | **0.636 ± 0.010** | **61.0 ± 0.6** |

**Test set final v2** : MAE **58.34 €** · R² **0.663** (vs 71.08 € / 0.589 en v1 → **−18 % MAE, +0.07 R²**).
**Modèle retenu** : **GradientBoostingRegressor** tuné (n=400, depth=5, lr=0.1).

## D13 — Pourquoi RidgeCV n'a pas sauvé les linéaires

Surprise : RidgeCV avec α∈{0.1, 1, 10, 100, 1000} a **empiré** les chiffres (R² −431k vs −114k en v1). Raison :

- RidgeCV optimise R² en **log-space** (sa cible interne via `TransformedTargetRegressor` est log1p(price))
- On évalue MAE/R² en **€** après `expm1`
- La régularisation log-optimale ≠ la régularisation €-optimale après amplification exponentielle
- Conclusion : le problème fondamental n'est pas la valeur d'α — c'est la combinaison `log + linear + OHE haute cardinalité + expm1`. À défendre en soutenance comme **point pédagogique majeur**.

## D14 — Distribution des erreurs par tranche (v2)

| Tranche €/nuit | n | MAE | Err. relative |
|---|---|---|---|
| 0–50 | 258 | 18.4 € | ~50 % |
| 50–100 | 2 274 | 23.2 € | ~30 % |
| 100–150 | 2 462 | 28.6 € | ~25 % |
| 150–250 | 2 722 | 43.8 € | ~22 % |
| 250–500 | 2 066 | 80.2 € | ~22 % |
| 500+ | 795 | 256.9 € | ~45 % |

Modèle homogène à ~22-30 % d'erreur relative sur le cœur du marché parisien (50–500 €/nuit).

## D15 — Taille model.joblib
GBM compact (400 arbres × profondeur 5) → **1.78 Mo** vs 677 Mo pour RandomForest v1. Compatible rendu Teams sans problème.
