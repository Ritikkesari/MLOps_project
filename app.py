# app.py

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import mlflow
import pandas as pd
import uvicorn
import logging

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define your MLflow model name and version
model_name = "House-pricing-model"
model_version = "4"

# SQLite URI for MLflow tracking server
mlflow_uri = "sqlite:///02-experiment_tracking/mlflow.db"  # Replace with the actual path to your mlflow.db file

try:
    # Initialize MLflow client
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient()

    # Get registered model details
    model_details = client.get_registered_model(model_name)

    # Load the model using its latest version
    model_version_details = client.get_model_version(model_details.name, model_version)
    model_uri = model_version_details.source

    # Alternatively, you can load the model using specific version URI
    # model_uri = f"{mlflow_uri}#models:/{model_name}/{model_version}"

    # Load the model
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    logger.info(
        f"Successfully loaded MLflow model '{model_name}' version '{model_version}' from '{model_uri}'"
    )
except Exception as e:
    logger.error(f"Failed to load MLflow model: {str(e)}")
    raise e


# Define input schema using Pydantic BaseModel
class HousePriceInput(BaseModel):
    area: float = Field(..., example=1500)
    bedrooms: int = Field(..., example=3)
    bathrooms: int = Field(..., example=2)
    stories: int = Field(..., example=2)
    parking: int = Field(..., example=1)


# Define output schema using Pydantic BaseModel
class HousePriceOutput(BaseModel):
    predicted_price: float


# Endpoint to predict house price
@app.post("/predict", response_model=HousePriceOutput)
async def predict_house_price(input_data: HousePriceInput):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data.dict()])

    try:
        # Perform prediction using the loaded model
        prediction = loaded_model.predict(input_df)
        return {"predicted_price": round(prediction[0])}  # Assuming a single prediction output
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to make predictions")


# Middleware to add process time header
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    import time

    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# If running this script directly, start the FastAPI application using uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
