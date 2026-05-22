# Pourquoi les modèles linéaires échouent sur ce dataset

**Projet** : Airbnb Paris — prédiction de prix
**Question prof** : pourquoi `LinearRegression` donne des R² catastrophiques, et est-ce que `HuberRegressor` corrige le problème ?

---

## 1. Les résultats observés (CV 5-fold, train set)

| Modèle             | R² moyen          | MAE moyen (€) |
|--------------------|-------------------|---------------|
| LinearRegression   | **−456 846**      | 680           |
| HuberRegressor     | **−36 248 208**   | 5 470         |
| Ridge (RidgeCV)    | **−431 758**      | 663           |
| RandomForest       | **+0.632**        | 61            |
| GradientBoosting   | **+0.636**        | 60            |

Les modèles linéaires explosent. Les modèles d'ensemble (arbres) donnent de bons résultats.

---

## 2. Pourquoi `LinearRegression` et `Ridge` explosent

### Le pipeline contient deux choix qui interagissent mal avec un modèle linéaire :

**(a) `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)`**
On entraîne sur `log(1 + price)` (cible quasi-gaussienne) et on inverse avec `expm1` pour restituer en euros. C'est la bonne pratique pour une cible skewed, et c'est très bénéfique pour les arbres.

**(b) `OneHotEncoder` sur `neighbourhood_cleansed`, `room_type`, `property_type`**
Cela crée ~120 colonnes binaires, dont beaucoup correspondent à des combinaisons rares (quartier peu fréquent + property_type peu fréquent).

### L'enchaînement qui casse :

1. Sur un fold de validation, le modèle linéaire rencontre une combinaison de OHE qu'il a très peu vue en entraînement.
2. Les coefficients linéaires pour ces colonnes rares sont mal estimés et peuvent prendre des valeurs extrêmes.
3. Le modèle prédit en espace log : par exemple `log_pred = 15`.
4. L'inverse `expm1(15) = 3 268 014 €` au lieu de ~200 €.
5. **Un seul outlier de prédiction de cette ampleur suffit à détruire la MAE et le R² du fold entier.**

C'est mathématique : `R² = 1 − SS_res / SS_tot`. Si un résidu vaut 3 000 000 €, son carré pèse `9 × 10¹²`, ce qui écrase totalement le dénominateur (variance totale ~10⁵).

---

## 3. Pourquoi on a tenté `HuberRegressor`

`HuberRegressor` est une **régression linéaire robuste aux outliers** :
- Loss quadratique pour les petits résidus (`|r| < epsilon`)
- Loss linéaire pour les grands résidus (`|r| > epsilon`)

L'idée : si le problème vient d'outliers dans `y_train`, Huber les ignorerait et donnerait un modèle plus stable. C'est ce que la littérature recommande quand `LinearRegression` est dominé par quelques valeurs extrêmes.

---

## 4. Pourquoi Huber est encore *pire* (−36M de R²)

Diagnostic après expérimentation : **le problème n'est pas dans `y_train`, il est dans les prédictions.**

- `y_train` est déjà nettoyé : on a cappé à Q98 (1201 €) et appliqué `log1p`, donc la cible d'entraînement est quasi-gaussienne et sans outliers. Huber n'a rien à "ignorer" côté entrée.
- L'explosion vient des **prédictions** : le modèle linéaire surajuste les colonnes OHE rares, prédit `log_pred` extrême, et `expm1` amplifie exponentiellement.
- Huber, en pénalisant **moins** fortement les gros résidus pendant le fit (loss linéaire au lieu de quadratique), **accepte des coefficients encore plus extrêmes** sur les colonnes rares. Résultat : prédictions encore plus instables après `expm1`.

**Conclusion** : Huber est conçu pour des outliers dans la *cible*, pas pour un problème d'**extrapolation exponentielle en sortie**. Ici, le remède aggrave le mal.

---

## 5. Pourquoi RandomForest et GradientBoosting fonctionnent

Les arbres de décision ne sont **pas sensibles à ce problème** :

- Ils ne font pas de combinaison linéaire des features → pas d'amplification de coefficients mal estimés.
- Leurs prédictions sont **bornées par les valeurs vues en entraînement** (moyennes de feuilles) → `expm1` ne peut jamais déborder dans l'absurde.
- Ils gèrent naturellement les interactions entre quartier × room_type × accommodates sans avoir besoin d'OHE.

C'est pour ça qu'on obtient **MAE ≈ 60 €** avec ces modèles, ce qui est très acceptable pour un prix moyen de ~160 €.

---

## 6. Ce qu'on retient (à dire au prof)

1. **L'échec des modèles linéaires n'est pas un bug** : c'est une conséquence prévisible du combo `OHE haute dimension + log1p/expm1`. C'est documenté dans la littérature (problème classique d'instabilité numérique des régressions linéaires sur cibles transformées).

2. **Huber ne corrige rien ici** parce que le problème n'est pas dans `y_train` mais dans la phase d'inférence. C'est un bon test négatif : ça prouve que la nature du problème est différente de ce qu'on aurait pu croire.

3. **Le bon choix de modèle est `GradientBoosting`** :
   - MAE test = **58.34 €**
   - R² test = **0.663**
   - C'est cohérent avec la CV (R² = 0.636, MAE = 60.99 €) → pas d'overfit.

4. **Pédagogiquement**, garder les 5 modèles dans la comparaison est utile : ça montre qu'on a testé plusieurs familles (linéaires simples, linéaires robustes, ensemble de bagging, ensemble de boosting) et qu'on a justifié notre choix final par des résultats chiffrés.

---

## 7. Pistes pour "sauver" la régression linéaire (si la question revient)

Si le prof demande "comment on pourrait faire marcher la régression linéaire", voici les options :

- **Retirer le `TransformedTargetRegressor`** (entraîner directement sur `price`) → on perd la propriété "cible gaussienne" mais on évite l'`expm1` explosif.
- **Cliper les prédictions** dans `expm1` à un plafond raisonnable (par ex. Q99 du train).
- **Régulariser plus fort** : `Ridge(alpha=10000)` ou `Lasso` qui mettrait les coefs des colonnes rares à zéro.
- **Réduire la dimension OHE** : `min_frequency=100` au lieu de 20 pour fusionner encore plus de modalités rares (déjà fait à 20 dans le notebook actuel).

Ces pistes ne sont **pas nécessaires** pour le projet puisque GradientBoosting donne déjà un bon résultat, mais elles montrent qu'on a compris le mécanisme du bug.
