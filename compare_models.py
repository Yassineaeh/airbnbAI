"""Compare models in CV 5-fold including HuberRegressor.

Réplique le pipeline du notebook Airbnb_Paris_Price.ipynb pour comparer rapidement
LinearRegression, HuberRegressor, Ridge, RandomForest, GradientBoosting.
"""
import ast
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LinearRegression, RidgeCV
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')
RNG = 42
DATA_PATH = Path('listings.csv')


def parse_price(s):
    if pd.isna(s):
        return np.nan
    return float(str(s).replace('$', '').replace(',', '').strip())


def parse_bathrooms(s):
    if pd.isna(s):
        return np.nan
    s = str(s).lower().strip()
    if 'half' in s:
        return 0.5
    for tok in s.split():
        try:
            return float(tok)
        except ValueError:
            continue
    return np.nan


def parse_amenities(s):
    if pd.isna(s):
        return []
    try:
        return [a.lower() for a in ast.literal_eval(s)]
    except Exception:
        return []


AMENITY_KEYWORDS = {
    'has_wifi':              ['wifi'],
    'has_kitchen':           ['kitchen'],
    'has_washer':            ['washer'],
    'has_tv':                [' tv', 'tv ', 'hdtv', 'television'],
    'has_air_conditioning':  ['air conditioning', 'ac unit'],
    'has_elevator':          ['elevator'],
    'has_balcony':           ['balcony', 'patio', 'terrace'],
    'has_free_parking':      ['free parking', 'free street parking', 'free residential garage'],
}


def has_any(lst, keywords):
    joined = ' | '.join(lst)
    return any(k in joined for k in keywords)


def load_and_clean():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df['price'] = df['price'].apply(parse_price)
    df = df[df['price'].notna() & (df['price'] > 0)].copy()
    q98 = df['price'].quantile(0.98)
    df = df[df['price'] <= q98].copy()

    if df['bathrooms'].isna().mean() > 0.5:
        df['bathrooms'] = df['bathrooms_text'].apply(parse_bathrooms)

    df['amenities_list'] = df['amenities'].apply(parse_amenities)
    for col, kws in AMENITY_KEYWORDS.items():
        df[col] = df['amenities_list'].apply(lambda lst, kws=kws: int(has_any(lst, kws)))

    df['host_is_superhost'] = (df['host_is_superhost'] == 't').astype(int)
    df['instant_bookable']  = (df['instant_bookable']  == 't').astype(int)
    for col in ['host_response_rate', 'host_acceptance_rate']:
        df[col] = df[col].astype(str).str.rstrip('%').replace('nan', np.nan).astype(float)

    top_pt = df['property_type'].value_counts().head(15).index
    df['property_type'] = df['property_type'].where(df['property_type'].isin(top_pt), 'Other')

    df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
    ref_date = pd.Timestamp('2025-01-01')
    df['host_since_days'] = (ref_date - df['host_since']).dt.days
    df['host_total_listings_count'] = pd.to_numeric(df['host_total_listings_count'], errors='coerce')
    df['bedrooms_x_accommodates'] = df['bedrooms'].fillna(0) * df['accommodates'].fillna(0)

    print(f'Données nettoyées : {len(df)} lignes (Q98 prix = {q98:.0f} €)')
    return df


def build_preprocess():
    NUM_FEATURES = [
        'accommodates', 'bedrooms', 'beds', 'bathrooms',
        'minimum_nights', 'availability_365', 'number_of_reviews',
        'review_scores_rating', 'review_scores_location', 'review_scores_cleanliness',
        'host_response_rate', 'host_acceptance_rate', 'reviews_per_month',
        'latitude', 'longitude',
        'host_since_days', 'host_total_listings_count', 'bedrooms_x_accommodates',
    ]
    CAT_FEATURES = ['neighbourhood_cleansed', 'room_type', 'property_type']
    BIN_FEATURES = ['host_is_superhost', 'instant_bookable'] + list(AMENITY_KEYWORDS)

    num_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False, min_frequency=20)),
    ])
    bin_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value=0)),
    ])
    preprocess = ColumnTransformer([
        ('num', num_pipe, NUM_FEATURES),
        ('cat', cat_pipe, CAT_FEATURES),
        ('bin', bin_pipe, BIN_FEATURES),
    ])
    return preprocess, NUM_FEATURES + CAT_FEATURES + BIN_FEATURES


def make_pipeline(estimator, preprocess):
    base = Pipeline([('prep', preprocess), ('model', estimator)])
    return TransformedTargetRegressor(regressor=base, func=np.log1p, inverse_func=np.expm1)


def main():
    df = load_and_clean()
    preprocess, FEATURES = build_preprocess()

    X = df[FEATURES]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RNG)
    print(f'train: {X_train.shape} | test: {X_test.shape}\n')

    models = {
        'LinearRegression':    LinearRegression(),
        'HuberRegressor':      HuberRegressor(epsilon=1.35, max_iter=200),
        'Ridge':               RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0]),
        'RandomForest':        RandomForestRegressor(n_estimators=200, max_depth=20,
                                                     n_jobs=1, random_state=RNG),
        'GradientBoosting':    GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                                         learning_rate=0.05, random_state=RNG),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=RNG)
    results = []
    for name, est in models.items():
        pipe = make_pipeline(est, preprocess)
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv, n_jobs=1,
            scoring={'r2': 'r2', 'mae': 'neg_mean_absolute_error'},
            return_train_score=False,
        )
        results.append({
            'model': name,
            'R2_mean': scores['test_r2'].mean(),
            'R2_std':  scores['test_r2'].std(),
            'MAE_mean': -scores['test_mae'].mean(),
            'MAE_std':   scores['test_mae'].std(),
        })
        print(f"{name:<20} R² = {scores['test_r2'].mean():>10.3f} ± {scores['test_r2'].std():.3f} | "
              f"MAE = {-scores['test_mae'].mean():.2f} ± {scores['test_mae'].std():.2f} €")

    print()
    cv_df = pd.DataFrame(results).sort_values('MAE_mean')
    print(cv_df.to_string(index=False))


if __name__ == '__main__':
    main()
