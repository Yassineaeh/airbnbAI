"""FastAPI minimal — Bonus A
Sert le modèle Airbnb Paris entraîné dans le notebook.

Lancer : uvicorn app:app --reload
Doc Swagger : http://localhost:8000/docs
"""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model.joblib"

app = FastAPI(
    title="Airbnb Paris Price API",
    description="Prédit le prix d'une nuitée à Paris à partir des caractéristiques d'un logement.",
    version="1.0.0",
)

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(500, f"model.joblib introuvable ({MODEL_PATH}). Exécute le notebook d'abord.")
        _model = joblib.load(MODEL_PATH)
    return _model


class ListingFeatures(BaseModel):
    accommodates: int = Field(2, ge=1, le=20)
    bedrooms: float = Field(1, ge=0, le=15)
    beds: float = Field(1, ge=0, le=20)
    bathrooms: float = Field(1.0, ge=0, le=10)
    minimum_nights: int = Field(1, ge=1)
    availability_365: int = Field(180, ge=0, le=365)
    number_of_reviews: int = Field(10, ge=0)
    review_scores_rating: float = Field(4.5, ge=0, le=5)
    review_scores_location: float = Field(4.7, ge=0, le=5)
    review_scores_cleanliness: float = Field(4.5, ge=0, le=5)
    host_response_rate: float = Field(90.0, ge=0, le=100, description="% (0-100)")
    host_acceptance_rate: float = Field(85.0, ge=0, le=100, description="% (0-100)")
    reviews_per_month: float = Field(1.5, ge=0)
    latitude: float = Field(48.8566)
    longitude: float = Field(2.3522)
    host_since_days: float = Field(1825, ge=0, description="Ancienneté de l'hôte en jours (réf. 2025-01-01)")
    host_total_listings_count: float = Field(2, ge=0, description="Nb total de listings de l'hôte")
    bedrooms_x_accommodates: float = Field(8, ge=0, description="Interaction bedrooms × accommodates")
    neighbourhood_cleansed: str = Field("Buttes-Montmartre", description="Nom exact d'un quartier Paris (cleansed)")
    room_type: str = Field("Entire home/apt")
    property_type: str = Field("Entire rental unit")
    host_is_superhost: int = Field(0, ge=0, le=1)
    instant_bookable: int = Field(0, ge=0, le=1)
    has_wifi: int = Field(1, ge=0, le=1)
    has_kitchen: int = Field(1, ge=0, le=1)
    has_washer: int = Field(1, ge=0, le=1)
    has_tv: int = Field(1, ge=0, le=1)
    has_air_conditioning: int = Field(0, ge=0, le=1)
    has_elevator: int = Field(0, ge=0, le=1)
    has_balcony: int = Field(0, ge=0, le=1)
    has_free_parking: int = Field(0, ge=0, le=1)


class PredictionResponse(BaseModel):
    predicted_price_eur: float
    model: str
    inputs_received: dict


@app.get("/health")
def health():
    model = get_model()
    inner = model.regressor_.named_steps["model"]
    return {
        "status": "ok",
        "model": type(inner).__name__,
        "model_file": str(MODEL_PATH.name),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: ListingFeatures):
    model = get_model()
    df = pd.DataFrame([features.model_dump()])
    try:
        price = float(model.predict(df)[0])
    except Exception as e:
        raise HTTPException(400, f"Échec prédiction : {e}")
    inner = model.regressor_.named_steps["model"]
    return PredictionResponse(
        predicted_price_eur=round(price, 2),
        model=type(inner).__name__,
        inputs_received=features.model_dump(),
    )


@app.get("/")
def root():
    return {
        "message": "Airbnb Paris Price API",
        "endpoints": {"POST /predict": "prédit le prix", "GET /health": "ping", "GET /docs": "Swagger UI"},
    }
