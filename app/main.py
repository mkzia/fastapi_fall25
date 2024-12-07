from typing import Union

from fastapi import FastAPI
import dill
from pydantic import BaseModel
import pandas as pd
import numpy as np

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
import joblib
app = FastAPI()

import dill
with open('./app/rfr_v1.pkl', 'rb') as f:
    reloaded_model = dill.load(f)


class Payload(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str

app = FastAPI()


@app.get("/")
def read_root():
    return {
        "Name": "John Doe",
        "Project": "blah",
        "Model": "blah"
    }


@app.post("/predict")
def predict(payload: Payload):
    df = pd.DataFrame([payload.model_dump().values()], columns=payload.model_dump().keys())
    y_hat = reloaded_model.predict(df)
    return {"prediction": y_hat[0]}