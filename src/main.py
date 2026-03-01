from fastapi import FastAPI, HTTPException
# https://fastapi.tiangolo.com/tutorial/cors/
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
from contextlib import asynccontextmanager
import pickle
import pandas as pd
from functools import partial

from src.models import ChurnData, ChurnPrediction, ChurnBatchPrediction
from config.logging import setup_logger


log = setup_logger()
context = {}


def predict(df: pd.DataFrame, preprocessing: dict, model):
    """Preprocess the received data."""
    # Encode categorical variables
    for col in preprocessing["categorical_cols"]:
        le = preprocessing["label_encoders"][col]
        df[col] = le.transform(df[col])

    # Scale numerical variables
    num_cols = preprocessing["numerical_cols"]
    df[num_cols] = preprocessing["scaler"].transform(df[num_cols])

    # Predict
    df = df[preprocessing["feature_columns"]]
    prediction = model.predict(df)
    probability = model.predict_proba(df)

    return {
        "prediction": prediction.tolist(),
        "probability": probability.tolist()
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Define pre-startup and post-shutdown logic. Based on
    https://fastapi.tiangolo.com/advanced/events/"""
    # Pre-startup logic.
    model_path = None
    try:
        model_path = Path(os.getenv("MODEL_PATH", None))
    except TypeError:
        log.error("Environment missing the MODEL_PATH variable.")

    try:
        with open(model_path / "churn_model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        with open(model_path / "preprocessing.pkl", "rb") as prep_file:
            preprocessing = pickle.load(prep_file)
        with open(model_path / "metrics.pkl", "rb") as metrics_file:
            context["metrics"] = pickle.load(metrics_file)
        log.info(f"Model loaded from {model_path}.")
        context["model"] = model
        context["preprocessing"] = preprocessing
        # Since the model and preprocessing dict won't change during runtime
        # a partial can be used for the prediction function.
        context["predict"] = partial(predict, preprocessing=preprocessing, model=model)
    except (TypeError, FileNotFoundError, pickle.UnpicklingError) as e:
        log.error(f"Could not load model from {model_path}")
    yield
    # Post-shutdown logic


app = FastAPI(
    title="churn-fastapi",
    lifespan=lifespan
)

origins = os.getenv("CORS_ORIGINS", "http://localhost,http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=[],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def checkhealth():
    """Health probe endpoint."""
    return {"version": os.getenv("VERSION", "unknown")}


@app.get("/readiness")
async def readiness():
    """Readiness probe endpoint."""
    # Check if everything necessary for the API was unpickled
    if not all(map(
        lambda k: k in context,
        ["model", "preprocessing", "metrics"]
    )):
        # https://fastapi.tiangolo.com/tutorial/handling-errors
        raise HTTPException(
            status_code=503,
            detail="Service cannot accept requests at the moment."
        )
    # Return metrics here
    return {"info": "Loaded model metrics.", **context["metrics"]}


@app.post("/predict")
async def predict_single(data: ChurnData) -> ChurnPrediction:
    try:
        df = pd.DataFrame([data.model_dump()])
        log.debug(f"Received data: {df.iloc[0]}.")
        result = context["predict"](df)
        return {
            "prediction": result["prediction"],
            "probability": result["probability"][0]
        }
    except Exception as e:
        log.error(f"Prediction error: {e}")
        log.debug(e.with_traceback())
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error"
        )


@app.post("/batch-predict")
async def predict_batch(data: list[ChurnData]) -> ChurnBatchPrediction:
    try:
        df = pd.DataFrame([d.model_dump() for d in data])
        log.debug(f"Received data: {df}.")
        return context["predict"](df)
    except Exception as e:
        log.error(f"Prediction error: {e}")
        log.debug(e.with_traceback())
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error"
        )
