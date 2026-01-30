# main_new.py
# This file will contain the new implementation of the FastAPI application,
# aligned with the wide-format database schema and multi-model strategy.

# We will build this file phase by phase.

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load environment variables
load_dotenv()

# --- Database Configuration ---
DB_HOST = os.getenv("DB_HOST", "postgres")
DB_NAME = os.getenv("POSTGRES_DB_CORE")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection error.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Energy AI Platform - Core API v2",
    version="2.0.0",
    description="API for training and serving energy consumption models with a multi-model strategy."
)

# --- Health Check Endpoints ---
@app.get("/", tags=["Health"])
def read_root():
    """Root endpoint providing a welcome message."""
    return {"message": "Welcome to the Energy AI Platform Core API v2!"}

@app.get("/health", tags=["Health"])
def health():
    """Health check endpoint to verify service status."""
    return {"status": "ok", "message": "API is healthy"}

# --- Global Constants for ML ---
SECTORES = ["Comedores", "Salones", "Laboratorios", "Auditorios", "Oficinas"]

# Base features from raw_data.consumo
BASE_FEATURES = [
    "temperatura_exterior_c", "ocupacion_pct", "periodo_academico", # Re-added periodo_academico
    "hora", "dia_semana", "mes", "trimestre", "ano",
    "es_fin_semana", "es_festivo", "es_semana_parciales", "es_semana_finales"
]

CATEGORICAL_FEATURES = [
    "dia_semana", "mes", "trimestre", "ano", # Treat these as categorical for tree models
    "periodo_academico", # Re-added periodo_academico
    "es_fin_semana", "es_festivo", "es_semana_parciales", "es_semana_finales"
]

NUMERICAL_FEATURES = [
    "temperatura_exterior_c", "ocupacion_pct", "hora"
]

# Directory to save trained models and feature lists
MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True) # Ensure directory exists

# --- Helper Function for Feature Engineering (from old main.py, adapted) ---
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates time-based features from the timestamp column.
    The new wide-format data already provides many pre-calculated features.
    This function will primarily ensure consistency if needed or add new ones.
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Ensure categorical features are treated as such for consistency with one-hot encoding
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype('category')
            
    return df

# --- Request Models for Training ---
class TrainPredictorRequest(BaseModel):
    sede_id: str
    sector: str # e.g., "Comedores", "Salones"
    # For now, we assume the target column name is "energia_{sector.lower()}_kwh"

class TrainPredictorResponse(BaseModel):
    message: str
    model_path: str
    features_path: str
    categories_map_path: str
    mae: float
    rmse: float

# --- Request/Response Models for Prediction ---
class PredictionRequest(BaseModel):
    timestamp: datetime
    temperatura_exterior_c: float = 20.0
    ocupacion_pct: float = 50.0
    periodo_academico: str = "semestre_1"
    hora: int
    dia_semana: int
    mes: int
    trimestre: int
    ano: int
    es_fin_semana: bool
    es_festivo: bool
    es_semana_parciales: bool
    es_semana_finales: bool

class PredictionResponse(BaseModel):
    sede_id: str
    sector: str
    timestamp: datetime
    consumo_predicho: float
    model_version: str

# --- Phase 1: Predictive Modeling ---
@app.post("/train-predictor", response_model=TrainPredictorResponse, tags=["Predictive Modeling"])
def train_predictor_model_endpoint(request: TrainPredictorRequest):
    """
    Trains an XGBoost model for a specific sector and campus (sede_id),
    and evaluates its performance.
    """
    print(f"Starting training for Sede: {request.sede_id}, Sector: {request.sector}...")
    
    # Map sector name to database column name
    sector_col_map = {
        "Comedores": "comedor",
        "Salones": "salones",
        "Laboratorios": "laboratorios",
        "Auditorios": "auditorios",
        "Oficinas": "oficinas"
    }
    
    if request.sector not in sector_col_map:
        raise HTTPException(status_code=400, detail=f"Invalid sector: {request.sector}. Must be one of {list(sector_col_map.keys())}")
        
    target_column = f"energia_{sector_col_map[request.sector].lower()}_kwh"
    model_filename = f"{MODEL_DIR}/{request.sede_id}_{request.sector.lower()}_predictor.joblib"
    features_filename = f"{MODEL_DIR}/{request.sede_id}_{request.sector.lower()}_predictor_features.joblib"
    categories_map_filename = f"{MODEL_DIR}/{request.sede_id}_{request.sector.lower()}_categories_map.joblib"

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Fetch data
        query_columns = ", ".join(["timestamp"] + BASE_FEATURES + [target_column])
        query = f"SELECT {query_columns} FROM raw_data.consumo WHERE sede_id = %s AND {target_column} IS NOT NULL;"
        cur.execute(query, (request.sede_id,))
        data = cur.fetchall()
        df = pd.DataFrame(data)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for Sede: {request.sede_id}, Sector: {request.sector}.")
        
        # Feature Engineering
        df = create_features(df)
        all_categories_map = {col: list(df[col].astype('category').cat.categories) for col in CATEGORICAL_FEATURES if col in df.columns}
        joblib.dump(all_categories_map, categories_map_filename)
        df_processed = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True)
        
        # Prepare X and y
        X = df_processed.drop(columns=['timestamp', target_column])
        y = df_processed[target_column]
        
        # --- Split data for evaluation ---
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Save the final list of features for consistent prediction
        joblib.dump(X_train.columns.tolist(), features_filename)
        
        # Train XGBoost model
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # --- Evaluate Model ---
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred)) # Calculate RMSE manually for compatibility
        print(f"Model Evaluation for {request.sede_id} - {request.sector}: MAE={mae:.4f}, RMSE={rmse:.4f}")

        # Save the model
        joblib.dump(model, model_filename)
        
        print(f"Model trained and saved successfully.")
        return TrainPredictorResponse(
            message=f"Model trained and saved successfully for {request.sede_id} - {request.sector}.",
            model_path=model_filename,
            features_path=features_filename,
            categories_map_path=categories_map_filename,
            mae=mae,
            rmse=rmse
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error during training for {request.sede_id} - {request.sector}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during training: {str(e)}")
    finally:
        if conn:
            conn.close()


@app.post("/predict/{sede_id}/{sector}", response_model=PredictionResponse, tags=["Predictive Modeling"])
def predict_consumo_endpoint(sede_id: str, sector: str, request: PredictionRequest):
    """
    Generates a prediction for a specific sector and campus using a trained model.
    """
    print(f"Prediction request for Sede: {sede_id}, Sector: {sector}...")

    # Construct file paths for the specific model
    model_filename = f"{MODEL_DIR}/{sede_id}_{sector.lower()}_predictor.joblib"
    features_filename = f"{MODEL_DIR}/{sede_id}_{sector.lower()}_predictor_features.joblib"
    categories_map_filename = f"{MODEL_DIR}/{sede_id}_{sector.lower()}_categories_map.joblib"

    # Check if model and artifacts exist
    if not all(os.path.exists(p) for p in [model_filename, features_filename, categories_map_filename]):
        raise HTTPException(
            status_code=404,
            detail=f"No trained model found for Sede: {sede_id}, Sector: {sector}. Please train the model first."
        )

    try:
        # Load model and artifacts
        model = joblib.load(model_filename)
        trained_features = joblib.load(features_filename)
        categories_map = joblib.load(categories_map_filename)

        # Create DataFrame from request
        input_data = request.dict()
        prediction_df = pd.DataFrame([input_data])
        
        # --- Data Preprocessing ---
        # Ensure categorical features are handled correctly
        for col, known_categories in categories_map.items():
            if col in prediction_df.columns:
                prediction_df[col] = pd.Categorical(prediction_df[col], categories=known_categories)
        
        # Apply one-hot encoding
        prediction_df_processed = pd.get_dummies(prediction_df, columns=list(categories_map.keys()), drop_first=True)
        
        # Align columns with the features the model was trained on
        X_predict = prediction_df_processed.reindex(columns=trained_features, fill_value=0)

        # --- Prediction ---
        consumo_predicho = model.predict(X_predict)[0]
        consumo_predicho = max(0.0, float(consumo_predicho)) # Ensure non-negative

        # --- Save Prediction to DB ---
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO analytics.predictions (sede_id, sector, timestamp, consumo_predicho, modelo)
            VALUES (%s, %s, %s, %s, %s);
            """,
            (sede_id, sector, request.timestamp, consumo_predicho, f"XGBoost_{sector.lower()}_v2")
        )
        conn.commit()
        cur.close()
        conn.close()

        return PredictionResponse(
            sede_id=sede_id,
            sector=sector,
            timestamp=request.timestamp,
            consumo_predicho=consumo_predicho,
            model_version=f"XGBoost_{sector.lower()}_v2"
        )

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {str(e)}")