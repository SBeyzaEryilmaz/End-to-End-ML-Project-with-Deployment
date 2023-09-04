import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from ml_model_monitoring.dataset_ops import DataProcessing

app = FastAPI()
preprocessor = DataProcessing()

model = joblib.load("models/model.joblib")


class FeatureModel(BaseModel):
    ID: int
    AGE: str
    GENDER: str
    RACE: str
    DRIVING_EXPERIENCE: str
    EDUCATION: str
    INCOME: str
    CREDIT_SCORE: float
    VEHICLE_OWNERSHIP: int
    VEHICLE_YEAR: str
    MARRIED: int
    CHILDREN: int
    POSTAL_CODE: int
    ANNUAL_MILEAGE: float
    VEHICLE_TYPE: str
    SPEEDING_VIOLATIONS: int
    DUIS: int
    PAST_ACCIDENTS: int


@app.get("/")
def main_page() -> dict:
    return {"message": "Welcome"}


@app.post("/predict/")
def predict(features: FeatureModel) -> dict:
    features_dict = features.dict()
    features_df = pd.DataFrame.from_dict(features_dict, orient="index").transpose()

    preprocessed_features = preprocessor.preprocess_data(features_df)
    prediction = model.predict(preprocessed_features)

    return {"prediction": int(prediction[0])}
