from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib # For saving/loading models
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os
import requests # Para hacer llamadas HTTP a otros endpoints (aunque lo usaremos solo para el concepto interno en MVP)
from datetime import datetime, timedelta

# --- Configuración de Datos (Duplicado de generate_data.py para consistencia) ---
SEDED = {
    "Tunja": 18000,
    "Duitama": 5500,
    "Sogamoso": 6000,
    "Chiquinquira": 2000
}
SECTORES = ["Comedores", "Salones", "Laboratorios", "Auditorios", "Oficinas"]

# Cargar variables de entorno
load_dotenv()

# --- Configuración de la Base de Datos ---
DB_HOST = os.getenv("DB_HOST", "postgres") # Usamos 'postgres' como host dentro de Docker
DB_NAME = os.getenv("POSTGRES_DB_CORE")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")

# OpenAI API Key (para futuras implementaciones del LLM)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            cursor_factory=RealDictCursor # Para obtener resultados como diccionarios
        )
        # Establecer el search_path para la sesión
        cursor = conn.cursor()
        cursor.execute("SET search_path TO raw_data, analytics, anomalies, recommendations, chat, public;")
        conn.commit()
        cursor.close()
        return conn
    except Exception as e:
        print(f"Error al conectar con la base de datos: {e}")
        raise HTTPException(status_code=500, detail="Error de conexión a la base de datos")

# --- FastAPI App ---
app = FastAPI(title="Energy AI Platform Core API", version="1.0.0")

# --- Rutas de Salud ---
@app.get("/health")
def health():
    """Verifica el estado del servicio."""
    return {"status": "ok", "message": "FastAPI service is healthy"}

@app.get("/")
def read_root():
    """Punto de entrada de la API."""
    return {"message": "Welcome to the Energy AI Platform Core API!"}

# --- Modelos de Datos para Predicción ---
class PredictionRequest(BaseModel):
    sede: str
    sector: str
    timestamp: datetime # Fecha y hora para la cual se quiere la predicción
    temperatura_c: float = 20.0 # Valor por defecto o promedio
    ocupacion_pct: float = 50.0 # Valor por defecto o promedio
    tarifa_cop_kwh: float = 500.0 # Valor por defecto
    factor_co2_kg_kwh: float = 0.3 # Valor por defecto
    periodo_academico: str = "Semestre 1" # Valor por defecto
    dia_especial: str = "Normal" # Valor por defecto


class PredictionResponse(BaseModel):
    sede: str
    sector: str
    timestamp: datetime
    consumo_predicho: float
    model_version: str = "XGBoost v1"

# --- Modelos de Datos para Detección de Anomalías ---
class AnomalyDetectionRequest(BaseModel):
    sede: str
    sector: str
    timestamp: datetime # Fecha y hora del evento a analizar
    consumo_kwh: float # Consumo real observado
    temperatura_c: float = 20.0
    ocupacion_pct: float = 50.0
    tarifa_cop_kwh: float = 500.0
    factor_co2_kg_kwh: float = 0.3
    periodo_academico: str = "Semestre 1"
    dia_especial: str = "Normal"

class AnomalyDetectionResponse(BaseModel):
    sede: str
    sector: str
    timestamp: datetime
    consumo_kwh: float
    is_anomaly: bool
    anomaly_type: Optional[str] = None
    severity: Optional[str] = None
    impacto_kwh: Optional[float] = None
    anomaly_probability: float # Ahora siempre se devuelve
    model_version: str = "XGBoost Anomaly Detector v1"

# --- Modelos de Datos para Recomendaciones (Placeholders) ---
class RecommendationRequest(BaseModel):
    sede: str
    sector: str
    anomaly_type: str
    severity: str
    impacto_kwh: float
    timestamp: datetime
    consumo_kwh: float
    temperatura_c: float = 20.0
    ocupacion_pct: float = 50.0
    anomaly_probability: Optional[float] = None # Nuevo campo

class RecommendationResponse(BaseModel):
    sede: str
    sector: str
    accion: str
    ahorro_estimado: float
    explicacion: str
    confianza: float = 0.8 # Valor fijo para MVP
    model_version: str = "Rule-based Recommender v1"


# --- Variables Globales para los Modelos ---
MODEL_PATH = "xgboost_predictor_model.joblib"
FEATURES_PATH = "predictor_features.joblib"
ALL_CATEGORIES_MAP_PATH = "all_categories_map.joblib" # Nuevo path para el mapa de categorías
TARGET_COLUMN = "consumo_kwh"
CATEGORICAL_FEATURES = ["sede", "sector", "periodo_academico", "dia_especial"]
NUMERICAL_FEATURES = [
    "temperatura_c", "ocupacion_pct", "tarifa_cop_kwh", "factor_co2_kg_kwh",
    "hour", "day_of_week", "day_of_month", "day_of_year", "month", "week_of_year", "year"
]

# Paths específicos para el modelo de anomalías
ANOMALY_MODEL_PATH = "xgboost_anomaly_model.joblib"
ANOMALY_FEATURES_PATH = "anomaly_features.joblib"
ANOMALY_TARGET_COLUMN = "es_anomalia_simulada"

# Nuevas features numéricas para el detector de anomalías (incluye consumo_kwh, consumo_predicho, diff, ratio)
# Nota: Consumo_kwh también se incluye como feature numérica para el detector
ANOMALY_NUMERICAL_FEATURES = NUMERICAL_FEATURES + ["consumo_kwh", "consumo_predicho", "consumo_kwh_diff", "consumo_kwh_ratio"]


# --- Funciones Auxiliares para Feature Engineering ---
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek # Lunes=0, Domingo=6
    df['day_of_month'] = df['timestamp'].dt.day
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['month'] = df['timestamp'].dt.month
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)
    df['year'] = df['timestamp'].dt.year
    
    return df

# --- Endpoint de Entrenamiento Predictor ---
@app.post("/train-predictor")
def train_predictor_model():
    """
    Entrena el modelo XGBoost para predecir el consumo energético.
    Carga datos históricos desde PostgreSQL, realiza ingeniería de características
    y guarda el modelo entrenado.
    """
    print("Iniciando entrenamiento del modelo predictivo...")
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Cargar datos de raw_data.consumo
        cur.execute("SELECT * FROM raw_data.consumo WHERE consumo_kwh IS NOT NULL;")
        data = cur.fetchall()
        df = pd.DataFrame(data)

        if df.empty:
            raise HTTPException(status_code=404, detail="No hay datos para entrenar el modelo.")
        
        # Ingeniería de características
        df = create_features(df)
        
        # Uno-hot encoding para características categóricas
        # Guardar el mapa de categorías para usarlo en la predicción
        all_categories_map = {}
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                all_categories_map[col] = list(df[col].unique())
        joblib.dump(all_categories_map, ALL_CATEGORIES_MAP_PATH) 

        df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True)
        
        # Asegurarse de que solo las features numéricas y las categóricas codificadas se usen
        final_features = [col for col in df.columns if col in NUMERICAL_FEATURES or any(col.startswith(cat_col + "_") for cat_col in CATEGORICAL_FEATURES)]
        
        X = df[final_features]
        y = df[TARGET_COLUMN]

        # Guardar la lista final de características para usar en la predicción
        joblib.dump(X.columns.tolist(), FEATURES_PATH)
        
        # Entrenar modelo XGBoost
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100, # Pequeño para MVP, se puede ajustar
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)

        # Guardar el modelo
        joblib.dump(model, MODEL_PATH)
        
        print("Modelo predictivo entrenado y guardado exitosamente.")
        return {"message": "Modelo predictivo entrenado y guardado exitosamente.", "model_path": MODEL_PATH}

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor durante el entrenamiento: {str(e)}")
    finally:
        if conn:
            conn.close()

# --- Endpoint de Predicción ---
@app.post("/predict", response_model=PredictionResponse)
def predict_consumo(request: PredictionRequest):
    """
    Genera una predicción de consumo energético para una sede, sector y timestamp dados.
    Carga el modelo entrenado y aplica las mismas transformaciones de características.
    """
    print(f"Solicitud de predicción para {request.sede}, {request.sector} en {request.timestamp}")
    conn = None
    try:
        # Cargar modelo, lista de características y mapa de categorías
        if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH) or not os.path.exists(ALL_CATEGORIES_MAP_PATH):
            raise HTTPException(status_code=404, detail="Modelo no entrenado o mapa de categorías faltante. Por favor, ejecute /train-predictor primero.")
        
        model = joblib.load(MODEL_PATH)
        trained_features = joblib.load(FEATURES_PATH)
        all_categories_map = joblib.load(ALL_CATEGORIES_MAP_PATH) # Cargar el mapa de categorías

        # Crear DataFrame para la predicción con los valores de la solicitud
        prediction_df = pd.DataFrame([{
            'timestamp': request.timestamp,
            'sede': request.sede,
            'sector': request.sector,
            'temperatura_c': request.temperatura_c,
            'ocupacion_pct': request.ocupacion_pct,
            'tarifa_cop_kwh': request.tarifa_cop_kwh,
            'factor_co2_kg_kwh': request.factor_co2_kg_kwh,
            'periodo_academico': request.periodo_academico,
            'dia_especial': request.dia_especial
        }])

        # Ingeniería de características
        prediction_df = create_features(prediction_df)

        # Aplicar One-hot encoding de forma consistente con el entrenamiento
        # Primero, aplicar get_dummies a la solicitud actual
        prediction_df_processed = pd.get_dummies(prediction_df, columns=CATEGORICAL_FEATURES, drop_first=True)
        
        # Alinear las columnas con las trained_features usando reindex
        # Esto asegura que el DataFrame tenga exactamente las mismas columnas
        # en el mismo orden que el modelo espera, rellenando con 0 las que falten.
        X_predict = prediction_df_processed.reindex(columns=trained_features, fill_value=0)
        
        # Realizar la predicción
        consumo_predicho = model.predict(X_predict)[0]
        consumo_predicho = max(0.0, float(consumo_predicho)) # Asegurar no negativos

        # Opcional: Guardar la predicción en la base de datos
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO analytics.predictions (sede, timestamp, consumo_predicho, modelo)
            VALUES (%s, %s, %s, %s);
            """,
            (request.sede, request.timestamp, consumo_predicho, "XGBoost Predictor v1")
        )
        conn.commit()
        
        return PredictionResponse(
            sede=request.sede,
            sector=request.sector,
            timestamp=request.timestamp,
            consumo_predicho=consumo_predicho
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error durante la predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor durante la predicción: {str(e)}")
    finally:
        if conn:
            conn.close()

# --- Endpoint de Entrenamiento Anomalias ---
@app.post("/train-detector")
def train_anomaly_detector_model():
    """
    Entrena el modelo XGBoost para detectar anomalías energéticas.
    Carga datos históricos (con anomalías simuladas) desde PostgreSQL,
    realiza ingeniería de características y guarda el modelo entrenado.
    """
    print("Iniciando entrenamiento del modelo detector de anomalías...")
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Cargar datos de raw_data.consumo
        # Y las predicciones para calcular las features de diferencia/ratio
        cur.execute("""
            SELECT
                r.*,
                p.consumo_predicho
            FROM raw_data.consumo r
            LEFT JOIN analytics.predictions p ON r.sede = p.sede AND r.timestamp = p.timestamp
            WHERE r.consumo_kwh IS NOT NULL
        """)
        data = cur.fetchall()
        df = pd.DataFrame(data)

        if df.empty:
            raise HTTPException(status_code=404, detail="No hay datos para entrenar el modelo de anomalías.")
        
        # Ingeniería de características
        df = create_features(df)
        
        # Calcular features de diferencia y ratio si consumo_predicho existe
        df['consumo_kwh_diff'] = df['consumo_kwh'] - df['consumo_predicho'].fillna(df['consumo_kwh']) # Si no hay prediccion, diff es 0
        df['consumo_kwh_ratio'] = df['consumo_kwh'] / df['consumo_predicho'].replace(0, np.nan).fillna(1) # Evitar div por cero, si no hay prediccion, ratio es 1
        
        # One-hot encoding para características categóricas
        # Usar el mapa de categorías guardado del predictor para consistencia.
        if os.path.exists(ALL_CATEGORIES_MAP_PATH):
            all_categories_map = joblib.load(ALL_CATEGORIES_MAP_PATH)
        else: # En caso de que se entrene el detector antes que el predictor, se recrea
            all_categories_map = {}
            for col in CATEGORICAL_FEATURES:
                if col in df.columns:
                    all_categories_map[col] = list(df[col].unique())
            joblib.dump(all_categories_map, ALL_CATEGORIES_MAP_PATH)
            
        df_processed = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True)
        
        # Las features para el detector de anomalías
        final_anomaly_features = [col for col in df_processed.columns if col in ANOMALY_NUMERICAL_FEATURES or any(col.startswith(cat_col + "_") for cat_col in CATEGORICAL_FEATURES)]
        
        X_anomaly = df_processed[final_anomaly_features]
        y_anomaly = df_processed[ANOMALY_TARGET_COLUMN]

        # Log de balance de clases
        true_count = y_anomaly.sum()
        false_count = len(y_anomaly) - true_count
        scale_pos_weight_value = false_count / true_count if true_count > 0 else 1
        print(f"Balance de clases para entrenamiento de anomalías: True={true_count}, False={false_count}")
        print(f"scale_pos_weight calculado: {scale_pos_weight_value}")

        # Guardar la lista final de características para usar en la detección
        joblib.dump(X_anomaly.columns.tolist(), ANOMALY_FEATURES_PATH)

        # Entrenar modelo XGBoost Classifier
        model = xgb.XGBClassifier(
            objective='binary:logistic', # Para clasificación binaria
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight_value
        )
        model.fit(X_anomaly, y_anomaly)

        # Guardar el modelo
        joblib.dump(model, ANOMALY_MODEL_PATH)
        
        print("Modelo detector de anomalías entrenado y guardado exitosamente.")
        return {"message": "Modelo detector de anomalías entrenado y guardado exitosamente.", "model_path": ANOMALY_MODEL_PATH}

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error durante el entrenamiento del detector de anomalías: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor durante el entrenamiento del detector: {str(e)}")
    finally:
        if conn:
            conn.close()

# --- Endpoint de Detección de Anomalías ---
@app.post("/detect-anomaly", response_model=AnomalyDetectionResponse)
def detect_anomaly(request: AnomalyDetectionRequest):
    """
    Detecta si un evento de consumo dado es una anomalía.
    Carga el modelo clasificador entrenado y aplica las mismas transformaciones.
    """
    print(f"Solicitud de detección de anomalía para {request.sede}, {request.sector} en {request.timestamp}")
    conn = None
    try:
        # Cargar modelo y lista de características
        if not os.path.exists(ANOMALY_MODEL_PATH) or not os.path.exists(ANOMALY_FEATURES_PATH) or not os.path.exists(ALL_CATEGORIES_MAP_PATH):
            raise HTTPException(status_code=404, detail="Modelo detector de anomalías no entrenado o mapa de categorías faltante. Por favor, ejecute /train-detector primero.")
        
        model_anomaly = joblib.load(ANOMALY_MODEL_PATH) # Renombrar a model_anomaly
        trained_anomaly_features = joblib.load(ANOMALY_FEATURES_PATH)
        all_categories_map = joblib.load(ALL_CATEGORIES_MAP_PATH) # Cargar el mapa de categorías

        # Primero, obtener la predicción del consumo para este punto de datos
        # Crear un PredictionRequest a partir del AnomalyDetectionRequest
        predictor_request = PredictionRequest(
            sede=request.sede,
            sector=request.sector,
            timestamp=request.timestamp,
            temperatura_c=request.temperatura_c,
            ocupacion_pct=request.ocupacion_pct,
            tarifa_cop_kwh=request.tarifa_cop_kwh,
            factor_co2_kg_kwh=request.factor_co2_kg_kwh,
            periodo_academico=request.periodo_academico,
            dia_especial=request.dia_especial
        )
        
        # --- Hack temporal: duplicar la lógica de predicción interna ---
        # En un sistema de producción con microservicios, harías una llamada HTTP a /predict
        # Aquí, simplemente duplicamos la lógica mínima para obtener el consumo_predicho
        if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
             raise HTTPException(status_code=500, detail="Modelo predictor no cargado para detección de anomalías.")
        
        predictor_model = joblib.load(MODEL_PATH)
        predictor_trained_features = joblib.load(FEATURES_PATH)
        
        pred_df_for_anomaly = pd.DataFrame([{
            'timestamp': request.timestamp,
            'sede': request.sede,
            'sector': request.sector,
            'temperatura_c': request.temperatura_c,
            'ocupacion_pct': request.ocupacion_pct,
            'tarifa_cop_kwh': request.tarifa_cop_kwh,
            'factor_co2_kg_kwh': request.factor_co2_kg_kwh,
            'periodo_academico': request.periodo_academico,
            'dia_especial': request.dia_especial
        }])
        pred_df_for_anomaly = create_features(pred_df_for_anomaly)
        pred_df_for_anomaly_processed = pd.get_dummies(pred_df_for_anomaly, columns=CATEGORICAL_FEATURES, drop_first=True)
        X_pred_for_anomaly = pred_df_for_anomaly_processed.reindex(columns=predictor_trained_features, fill_value=0)
        consumo_predicho = predictor_model.predict(X_pred_for_anomaly)[0]
        # --- Fin del hack temporal ---


        # Crear DataFrame para la detección con los valores de la solicitud
        detection_df = pd.DataFrame([{
            'timestamp': request.timestamp,
            'sede': request.sede,
            'sector': request.sector,
            'consumo_kwh': request.consumo_kwh, # Es el consumo real
            'temperatura_c': request.temperatura_c,
            'ocupacion_pct': request.ocupacion_pct,
            'tarifa_cop_kwh': request.tarifa_cop_kwh,
            'factor_co2_kg_kwh': request.factor_co2_kg_kwh,
            'periodo_academico': request.periodo_academico,
            'dia_especial': request.dia_especial,
            'consumo_predicho': consumo_predicho # Añadimos la predicción como feature
        }])

        # Ingeniería de características
        detection_df = create_features(detection_df)

        # Calcular features de diferencia y ratio
        detection_df['consumo_kwh_diff'] = detection_df['consumo_kwh'] - detection_df['consumo_predicho']
        # Evitar división por cero, si predicho es 0, el ratio es indefinido o muy alto, lo tratamos como np.nan o un valor grande
        detection_df['consumo_kwh_ratio'] = detection_df['consumo_kwh'] / detection_df['consumo_predicho'].replace(0, np.nan).fillna(detection_df['consumo_kwh'] + 1) # Fallback seguro
        
        # Aplicar One-hot encoding de forma consistente con el entrenamiento
        detection_df_processed = pd.get_dummies(detection_df, columns=CATEGORICAL_FEATURES, drop_first=True)
        
        # Alinear las columnas con las trained_anomaly_features
        X_detect = pd.DataFrame(0, index=detection_df_processed.index, columns=trained_anomaly_features)
        
        # Copiar los valores del df de detección a X_detect
        for col in detection_df_processed.columns:
            if col in X_detect.columns:
                X_detect[col] = detection_df_processed[col]
        
        X_detect = X_detect[trained_anomaly_features]

        # Realizar la detección
        is_anomaly_proba = model_anomaly.predict_proba(X_detect)[:, 1][0] # Probabilidad de ser anomalía
        is_anomaly = (is_anomaly_proba > 0.5) # Umbral de 0.5 para clasificar

        anomaly_type = None
        severity = None
        impacto_kwh = None

        if is_anomaly:
            anomaly_type = "Pico de Consumo Inesperado"
            severity = "Alta" # Podría ser más inteligente, ej. basarse en consumo_kwh - consumo_predicho
            impacto_kwh = float(request.consumo_kwh - consumo_predicho) # Convertir a float estándar
            if impacto_kwh < 0: impacto_kwh = float(request.consumo_kwh) # Si es anomalia por consumo bajo, el impacto es el consumo itself
            
            # Buscar el consumo_id correspondiente en raw_data.consumo
            consumo_id_found = None
            conn_search = get_db_connection() # Nueva conexión para la búsqueda
            cur_search = conn_search.cursor()
            
            # Truncar el timestamp de la solicitud a la hora en Python y hacerlo naive
            truncated_timestamp = request.timestamp.replace(minute=0, second=0, microsecond=0, tzinfo=None)
            print(f"DEBUG: truncated_timestamp (Python) = {truncated_timestamp} (type: {type(truncated_timestamp)})")

            cur_search.execute(
                """
                SELECT id, timestamp FROM raw_data.consumo
                WHERE sede = %s AND sector = %s AND timestamp = %s;
                """,
                (request.sede, request.sector, truncated_timestamp)
            )
            result = cur_search.fetchone()
            print(f"DEBUG: Result from consumo_id search: {result}") # Nuevo print
            if result:
                consumo_id_found = result['id']
                print(f"DEBUG: Found matching record with ID={consumo_id_found}, timestamp_db={result['timestamp']} (type: {type(result['timestamp'])})")
            else:
                print(f"DEBUG: No matching record found for sede={request.sede}, sector={request.sector}, timestamp={truncated_timestamp}")
                # Add a more robust search fallback here for debugging
                cur_search.execute(
                    """
                    SELECT id, timestamp FROM raw_data.consumo
                    WHERE sede = %s AND sector = %s
                    ORDER BY ABS(EXTRACT(EPOCH FROM (timestamp - %s::timestamp)))
                    LIMIT 1;
                    """,
                    (request.sede, request.sector, truncated_timestamp)
                )
                nearby_result = cur_search.fetchone()
                if nearby_result:
                    print(f"DEBUG: Found closest record: ID={nearby_result['id']}, timestamp_db={nearby_result['timestamp']} (used for insertion fallback)")
                    consumo_id_found = nearby_result['id']
                
            conn_search.close() # Cerrar la conexión temporal para la búsqueda
            
            # Reabrir conexión para la inserción de anomalía
            conn_insert = get_db_connection() # Nueva conexión para la inserción
            cur_insert = conn_insert.cursor()
            cur_insert.execute(
                """
                INSERT INTO anomalies.detected (consumo_id, tipo, severidad, impacto_kwh)
                VALUES (%s, %s, %s, %s);
                """,
                (consumo_id_found, anomaly_type, severity, impacto_kwh)
            )
            conn_insert.commit()
            conn_insert.close() # Cerrar la conexión de inserción


        return AnomalyDetectionResponse(
            sede=request.sede,
            sector=request.sector,
            timestamp=request.timestamp,
            consumo_kwh=request.consumo_kwh,
            is_anomaly=is_anomaly,
            anomaly_type=anomaly_type,
            severity=severity,
            impacto_kwh=impacto_kwh,
            anomaly_probability=is_anomaly_proba # Siempre devolver la probabilidad
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error durante la detección de anomalías: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor durante la detección de anomalías: {str(e)}")
    finally:
        if conn:
            conn.close()

# --- Endpoint de Recomendaciones ---
@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendation(request: RecommendationRequest):
    """
    Genera una recomendación basada en el tipo de anomalía y su severidad.
    Para MVP, utiliza un sistema basado en reglas.
    """
    print(f"Solicitud de recomendación para {request.sede}, {request.sector} - Anomalía: {request.anomaly_type}")

    accion = "No se requiere acción específica."
    ahorro_estimado = 0.0
    explicacion = "No se encontró una regla de recomendación para esta anomalía."
    
    # Usar la probabilidad de la anomalía si está disponible, si no, usar el valor por defecto
    confianza = request.anomaly_probability if request.anomaly_probability is not None else 0.8

    if request.anomaly_type == "Pico de Consumo Inesperado":
        if request.severity == "Alta":
            accion = "Revisar equipos de alto consumo y desconectar los no esenciales. Optimizar horarios de operación."
            ahorro_estimado = request.impacto_kwh * 0.3 # Estimar un 30% del impacto
            explicacion = (
                f"Se detectó un pico de consumo inesperado de {request.impacto_kwh:.2f} kWh en {request.sector} "
                f"en {request.sede} a las {request.timestamp.strftime('%H:%M')}. "
                "Considerar inspeccionar equipos como sistemas de climatización, iluminación o maquinaria de laboratorio "
                "fuera de los horarios habituales. Un 30% de ahorro estimado se puede lograr con la optimización."
            )
        elif request.severity == "Media":
            accion = "Verificar patrones de uso y ajustar configuraciones de equipos."
            ahorro_estimado = request.impacto_kwh * 0.15 # Estimar un 15%
            explicacion = (
                f"Se detectó un consumo elevado de {request.impacto_kwh:.2f} kWh en {request.sector} "
                f"en {request.sede} a las {request.timestamp.strftime('%H:%M')}. "
                "Se sugiere revisar el uso de la energía en este periodo. Un 15% de ahorro estimado."
            )
    
    # Aquí se podrían añadir más reglas para otros tipos de anomalías o severidades

    # Guardar la recomendación en la base de datos
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO recommendations.generated (sede, sector, timestamp, tipo_anomalia, accion, ahorro_estimado, explicacion, confianza)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
            """,
            (request.sede, request.sector, request.timestamp, request.anomaly_type, accion, ahorro_estimado, explicacion, confianza)
        )
        conn.commit()
    except Exception as e:
        print(f"Error al guardar la recomendación en la base de datos: {e}")
        # No se eleva la excepción para no detener el flujo de la recomendación
    finally:
        if conn:
            conn.close()

    return RecommendationResponse(
        sede=request.sede,
        sector=request.sector,
        accion=accion,
        ahorro_estimado=ahorro_estimado,
        explicacion=explicacion,
        confianza=confianza
    )