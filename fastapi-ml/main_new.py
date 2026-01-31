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
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
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


# --- Request Models for Inefficiency/Anomaly Detection ---
class AnomalyDetectorTrainRequest(BaseModel):
    sede_id: str
    sector: Optional[str] = None # If None, train for total energy consumption
    # Parameters for defining inefficiency (e.g., multiplier for std dev above mean)
    std_dev_multiplier: float = 3.0 # How many std deviations above the mean to flag as inefficient

class AnomalyDetectorTrainResponse(BaseModel):
    message: str
    model_path: str
    features_path: str
    categories_map_path: str
    # Evaluation metrics for classification
    accuracy: float
    precision: float
    recall: float
    f1_score: float

class AnomalyPredictionRequest(BaseModel):
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

class AnomalyPredictionResponse(BaseModel):
    sede_id: str
    sector: str
    timestamp: datetime
    is_inefficient: bool
    inefficiency_score: float # Probability of being inefficient
    model_version: str

# --- Request/Response Models for Recommendation Generation ---
class GenerateRecommendationResponse(BaseModel):
    message: str
    recommendation_id: int
    anomaly_id: int
    accion: str
    ahorro_estimado: Optional[float]
    explicacion: Optional[str]


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
    finally:
        if conn:
            conn.close()


@app.post("/train-inefficiency-detector", response_model=AnomalyDetectorTrainResponse, tags=["Inefficiency Detection"])
def train_inefficiency_detector_endpoint(request: AnomalyDetectorTrainRequest):
    """
    Trains an XGBoost classifier to detect energy inefficiency/anomalies for a specific campus (sede_id)
    and optionally a sector. Inefficiency is defined as consumption significantly above the historical
    mean for similar conditions (hora, dia_semana).
    """
    print(f"Starting inefficiency detector training for Sede: {request.sede_id}, Sector: {request.sector if request.sector else 'Total'}...")

    # Determine target column
    sector_col_map = {
        "Comedores": "comedor",
        "Salones": "salones",
        "Laboratorios": "laboratorios",
        "Auditorios": "auditorios",
        "Oficinas": "oficinas"
    }

    if request.sector:
        if request.sector not in sector_col_map:
            raise HTTPException(status_code=400, detail=f"Invalid sector: {request.sector}. Must be one of {list(sector_col_map.keys())}")
        target_column_prefix = f"energia_{sector_col_map[request.sector].lower()}_kwh"
    else:
        target_column_prefix = "energia_total_kwh"

    model_filename = f"{MODEL_DIR}/{request.sede_id}_{request.sector.lower() if request.sector else 'total'}_inefficiency_detector.joblib"
    features_filename = f"{MODEL_DIR}/{request.sede_id}_{request.sector.lower() if request.sector else 'total'}_inefficiency_features.joblib"
    categories_map_filename = f"{MODEL_DIR}/{request.sede_id}_{request.sector.lower() if request.sector else 'total'}_inefficiency_categories_map.joblib"

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Fetch data
        query_columns = ", ".join(["sede_id", "timestamp"] + BASE_FEATURES + [target_column_prefix])
        query = f"SELECT {query_columns} FROM raw_data.consumo WHERE sede_id = %s AND {target_column_prefix} IS NOT NULL;"
        cur.execute(query, (request.sede_id,))
        data = cur.fetchall()
        df = pd.DataFrame(data)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for Sede: {request.sede_id}, Sector: {request.sector if request.sector else 'Total'}.")
        
        # Feature Engineering (will be used for X)
        df_copy = df.copy() # Use a copy to avoid modifying original df for calculations
        df_copy = create_features(df_copy)

        # --- Generate 'is_inefficient' labels ---
        # Calculate mean and std dev for each unique (sede_id, dia_semana, hora) combination
        # This creates a context-aware threshold for inefficiency
        df_copy['rolling_mean'] = df_copy.groupby(['sede_id', 'dia_semana', 'hora'])[target_column_prefix].transform('mean')
        df_copy['rolling_std'] = df_copy.groupby(['sede_id', 'dia_semana', 'hora'])[target_column_prefix].transform('std').fillna(0) # Fill NaN for single observations

        # Define inefficiency: actual consumption > (mean + std_dev_multiplier * df_copy['rolling_std'])) & (df_copy[target_column_prefix] > 0)).astype(int)
        df_copy['is_inefficient'] = ((df_copy[target_column_prefix] > (df_copy['rolling_mean'] + request.std_dev_multiplier * df_copy['rolling_std'])) & (df_copy[target_column_prefix] > 0)).astype(int)

        # Drop temporary columns used for labeling
        df_copy.drop(columns=['rolling_mean', 'rolling_std'], inplace=True)

        # Check if there are any inefficient labels to train on
        if df_copy['is_inefficient'].sum() == 0:
            raise HTTPException(status_code=400, detail=f"No inefficiency instances found with std_dev_multiplier={request.std_dev_multiplier}. Consider lowering the multiplier.")

        # Handle class imbalance (optional but recommended for anomaly detection)
        # For simplicity, we'll proceed without explicit resampling for now, but a warning can be issued
        if df_copy['is_inefficient'].mean() < 0.1: # Less than 10% positive class
            print("Warning: Highly imbalanced dataset for inefficiency detection. Consider using techniques like SMOTE or adjusting class weights.")

        # Process features for model training
        all_categories_map = {col: list(df_copy[col].astype('category').cat.categories) for col in CATEGORICAL_FEATURES if col in df_copy.columns}
        joblib.dump(all_categories_map, categories_map_filename)
        df_processed = pd.get_dummies(df_copy, columns=CATEGORICAL_FEATURES, drop_first=True)
        
        # Prepare X and y for the classifier
        # Explicitly drop 'sede_id' as it's used for filtering, not as a feature in X for training.
        X = df_processed.drop(columns=['sede_id', 'timestamp', target_column_prefix, 'is_inefficient'])
        y = df_processed['is_inefficient']
        
        # --- Split data for evaluation ---
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Save the final list of features for consistent prediction
        joblib.dump(X_train.columns.tolist(), features_filename)
        
        # Train XGBoost Classifier model
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum() # Handle imbalance
        )
        model.fit(X_train, y_train)

        # --- Evaluate Model ---
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Inefficiency Detector Evaluation for {request.sede_id} - {request.sector if request.sector else 'Total'}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}")

        # Save the model
        joblib.dump(model, model_filename)
        
        print(f"Inefficiency Detector trained and saved successfully.")
        return AnomalyDetectorTrainResponse(
            message=f"Inefficiency Detector trained and saved successfully for {request.sede_id} - {request.sector if request.sector else 'Total'}.",
            model_path=model_filename,
            features_path=features_filename,
            categories_map_path=categories_map_filename,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error during inefficiency detector training for {request.sede_id} - {request.sector if request.sector else 'Total'}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during inefficiency detector training: {str(e)}")
    finally:
        if conn:
            conn.close()


@app.post("/predict-inefficiency/{sede_id}/{sector_or_total}", response_model=AnomalyPredictionResponse, tags=["Inefficiency Detection"])
def predict_inefficiency_endpoint(sede_id: str, sector_or_total: str, request: AnomalyPredictionRequest):
    """
    Generates an inefficiency prediction for a specific campus (sede_id) and sector (or total).
    """
    print(f"Inefficiency prediction request for Sede: {sede_id}, Target: {sector_or_total}...")

    # Determine target (sector or total) for filename consistency
    target_lower = sector_or_total.lower() if sector_or_total in SECTORES else 'total'

    # Construct file paths for the specific model
    model_filename = f"{MODEL_DIR}/{sede_id}_{target_lower}_inefficiency_detector.joblib"
    features_filename = f"{MODEL_DIR}/{sede_id}_{target_lower}_inefficiency_features.joblib"
    categories_map_filename = f"{MODEL_DIR}/{sede_id}_{target_lower}_inefficiency_categories_map.joblib"

    # Check if model and artifacts exist
    if not all(os.path.exists(p) for p in [model_filename, features_filename, categories_map_filename]):
        raise HTTPException(
            status_code=404,
            detail=f"No trained inefficiency detector found for Sede: {sede_id}, Target: {target_lower}. Please train the model first."
        )

    conn = None
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
        is_inefficient_pred = model.predict(X_predict)[0]
        inefficiency_score_pred = model.predict_proba(X_predict)[0][1] # Probability of being the positive class (inefficient)

        # --- Save detected inefficiency to DB if applicable ---
        if is_inefficient_pred == 1:
            conn = get_db_connection()
            cur = conn.cursor()
            
            # Placeholder for calculating impact_kwh - ideally, this would come from a prediction model
            # For now, we can use a fixed value or a placeholder.
            # In a real scenario, this would be `actual_consumption - expected_normal_consumption`
            impact_kwh_placeholder = 0.0 # This needs to be calculated based on actual data vs. normal
            
            # Fetch actual consumption for the given timestamp and target
            target_column_for_db = f"energia_{sector_or_total.lower()}_kwh" if sector_or_total in SECTORES else "energia_total_kwh"
            query_actual_consumption = f"SELECT {target_column_for_db} FROM raw_data.consumo WHERE sede_id = %s AND timestamp = %s;"
            cur.execute(query_actual_consumption, (sede_id, request.timestamp))
            actual_consumption_row = cur.fetchone()
            
            actual_consumption = None
            if actual_consumption_row and actual_consumption_row[target_column_for_db] is not None:
                actual_consumption = actual_consumption_row[target_column_for_db]
                # To calculate impact, we need a baseline of "expected normal" consumption.
                # This could come from a trained predictor model or a simple rolling mean.
                # For this implementation, we'll use a simplified impact calculation if actual is available.
                # If there's a corresponding predictive model, we could use its prediction here for comparison.
                # Since we don't have that directly linked in this endpoint, let's keep it simple.
                # For a more robust solution, the predictive model's output could be part of the request context.
                impact_kwh_placeholder = float(actual_consumption) * 0.1 # Example: 10% of actual consumption as impact

            cur.execute(
                """
                INSERT INTO anomalies.detected (sede_id, sector, timestamp, tipo, severidad, impacto_kwh, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW());
                """,
                (sede_id, sector_or_total, request.timestamp, "Ineficiencia por Alto Consumo", f"{inefficiency_score_pred:.2f}", impact_kwh_placeholder)
            )
            conn.commit()
            cur.close()
            conn.close()

        return AnomalyPredictionResponse(
            sede_id=sede_id,
            sector=sector_or_total,
            timestamp=request.timestamp,
            is_inefficient=bool(is_inefficient_pred),
            inefficiency_score=float(inefficiency_score_pred),
            model_version=f"XGBoost_Inefficiency_{target_lower}_v1"
        )

    except Exception as e:
        print(f"Error during inefficiency prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during inefficiency prediction: {str(e)}")
    finally:
        if conn:
            conn.close()


# --- Phase 3: Recommendation Engine ---
def _create_recommendation_logic(anomaly: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates a recommendation based on a rule-based system.
    Input: Anomaly details from the database.
    Output: A dictionary with recommendation details.
    """
    sector = anomaly["sector"]
    anomaly_type = anomaly["tipo"]
    timestamp = anomaly["timestamp"]
    impact = float(anomaly.get("impacto_kwh", 0.0) or 0.0)

    # Default recommendation
    rec = {
        "accion": "Investigar la causa del consumo energético elevado.",
        "explicacion": f"Se ha detectado una anomalía de tipo '{anomaly_type}' en el sector '{sector}'. Se recomienda una inspección para identificar la fuente.",
        "ahorro_estimado": impact * 0.3,
        "confianza": 0.6
    }

    if anomaly_type == "Ineficiencia por Alto Consumo":
        is_weekend = timestamp.weekday() >= 5
        hour = timestamp.hour

        if sector == "Salones" and (hour < 7 or hour > 22 or is_weekend):
            rec = {
                "accion": "Apagar luces y equipos en salones no utilizados.",
                "explicacion": "Se detectó un consumo energético elevado en los salones fuera del horario académico habitual o durante el fin de semana. Es probable que luces o equipos hayan quedado encendidos.",
                "ahorro_estimado": impact * 0.8,
                "confianza": 0.9
            }
        elif sector == "Oficinas" and (hour > 18 or hour < 8 or is_weekend):
            rec = {
                "accion": "Verificar y apagar equipos de oficina y climatización.",
                "explicacion": "Consumo elevado detectado en oficinas fuera del horario laboral. Revisar que los computadores, impresoras y sistemas de aire acondicionado estén apagados.",
                "ahorro_estimado": impact * 0.75,
                "confianza": 0.85
            }
        elif sector == "Laboratorios":
             rec = {
                "accion": "Revisar equipos especializados y sistemas de ventilación en laboratorios.",
                "explicacion": "Los laboratorios muestran un consumo anómalo. Verificar que los equipos de alta demanda y los sistemas de extracción de aire no estén operando innecesariamente.",
                "ahorro_estimado": impact * 0.6,
                "confianza": 0.8
            }

    return rec


@app.post("/generate-recommendation/{anomaly_id}", response_model=GenerateRecommendationResponse, tags=["Recommendation Engine"])
def generate_recommendation_endpoint(anomaly_id: int):
    """
    Generates and stores a recommendation for a given anomaly ID.
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # 1. Fetch anomaly details
        cur.execute("SELECT * FROM anomalies.detected WHERE id = %s;", (anomaly_id,))
        anomaly = cur.fetchone()

        if not anomaly:
            raise HTTPException(status_code=404, detail=f"Anomaly with ID {anomaly_id} not found.")

        # 2. Generate recommendation logic
        recommendation_logic = _create_recommendation_logic(anomaly)

        # 3. Store the new recommendation in the database
        insert_query = """
            INSERT INTO recommendations.generated
            (anomaly_id, sede_id, sector, timestamp, tipo_anomalia, accion, ahorro_estimado, explicacion, confianza, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            RETURNING id;
        """
        cur.execute(
            insert_query,
            (
                anomaly_id,
                anomaly["sede_id"],
                anomaly["sector"],
                anomaly["timestamp"],
                anomaly["tipo"],
                recommendation_logic["accion"],
                recommendation_logic["ahorro_estimado"],
                recommendation_logic["explicacion"],
                recommendation_logic["confianza"],
            ),
        )
        new_recommendation_id = cur.fetchone()["id"]
        conn.commit()

        return GenerateRecommendationResponse(
            message="Recommendation generated successfully.",
            recommendation_id=new_recommendation_id,
            anomaly_id=anomaly_id,
            accion=recommendation_logic["accion"],
            ahorro_estimado=recommendation_logic["ahorro_estimado"],
            explicacion=recommendation_logic["explicacion"],
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error during recommendation generation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        if conn:
            cur.close()
            conn.close()

