\c core_db;

-- ============================
-- Schemas
-- ============================
CREATE SCHEMA IF NOT EXISTS raw_data;
GRANT USAGE ON SCHEMA raw_data TO energy_user;
CREATE SCHEMA IF NOT EXISTS analytics;
GRANT USAGE ON SCHEMA analytics TO energy_user;
CREATE SCHEMA IF NOT EXISTS anomalies;
GRANT USAGE ON SCHEMA anomalies TO energy_user;
CREATE SCHEMA IF NOT EXISTS recommendations;
GRANT USAGE ON SCHEMA recommendations TO energy_user;
CREATE SCHEMA IF NOT EXISTS chat;
GRANT USAGE ON SCHEMA chat TO energy_user;

-- ============================
-- Sedes (Campuses)
-- ============================
CREATE TABLE IF NOT EXISTS raw_data.sedes (
  sede_id TEXT PRIMARY KEY,
  sede TEXT NOT NULL,
  nombre_completo TEXT,
  ciudad TEXT,
  area_m2 INT,
  num_estudiantes INT,
  num_empleados INT,
  num_edificios INT,
  tiene_residencias BOOLEAN,
  tiene_laboratorios_pesados BOOLEAN,
  altitud_msnm INT,
  temp_promedio_c NUMERIC,
  pct_comedores NUMERIC,
  pct_salones NUMERIC,
  pct_laboratorios NUMERIC,
  pct_auditorios NUMERIC,
  pct_oficinas NUMERIC
);
GRANT ALL PRIVILEGES ON TABLE raw_data.sedes TO energy_user;


-- ============================
-- Datos Crudos (Wide Format)
-- ============================
CREATE TABLE IF NOT EXISTS raw_data.consumo (
  reading_id BIGSERIAL PRIMARY KEY,
  timestamp TIMESTAMP NOT NULL,
  sede_id TEXT REFERENCES raw_data.sedes(sede_id),

  -- Energía Total
  energia_total_kwh NUMERIC,
  potencia_total_kw NUMERIC,
  co2_kg NUMERIC,

  -- Energía por Sector
  energia_comedor_kwh NUMERIC,
  energia_salones_kwh NUMERIC,
  energia_laboratorios_kwh NUMERIC,
  energia_auditorios_kwh NUMERIC,
  energia_oficinas_kwh NUMERIC,

  -- Agua y Contexto
  agua_litros NUMERIC,
  temperatura_exterior_c NUMERIC,
  ocupacion_pct NUMERIC,

  -- Variables Temporales Pre-calculadas
  hora INT,
  dia_semana INT,
  dia_nombre TEXT,
  mes INT,
  trimestre INT,
  ano INT,
  periodo_academico TEXT,
  es_fin_semana BOOLEAN,
  es_festivo BOOLEAN,
  es_semana_parciales BOOLEAN,
  es_semana_finales BOOLEAN
);
GRANT ALL PRIVILEGES ON TABLE raw_data.consumo TO energy_user;
GRANT ALL PRIVILEGES ON SEQUENCE raw_data.consumo_reading_id_seq TO energy_user;


-- ============================
-- Predicciones ML (Long Format)
-- ============================
CREATE TABLE IF NOT EXISTS analytics.predictions (
  id BIGSERIAL PRIMARY KEY,
  sede_id TEXT REFERENCES raw_data.sedes(sede_id),
  sector TEXT NOT NULL, -- e.g., 'comedores', 'salones'
  timestamp TIMESTAMP,
  consumo_predicho NUMERIC,
  modelo TEXT,
  created_at TIMESTAMP DEFAULT now()
);
GRANT ALL PRIVILEGES ON TABLE analytics.predictions TO energy_user;
GRANT ALL PRIVILEGES ON SEQUENCE analytics.predictions_id_seq TO energy_user;


-- ============================
-- Anomalías (Long Format)
-- ============================
CREATE TABLE IF NOT EXISTS anomalies.detected (
  id BIGSERIAL PRIMARY KEY,
  prediction_id BIGINT REFERENCES analytics.predictions(id),
  sede_id TEXT REFERENCES raw_data.sedes(sede_id),
  sector TEXT NOT NULL,
  timestamp TIMESTAMP,
  tipo TEXT,
  severidad TEXT,
  impacto_kwh NUMERIC,
  created_at TIMESTAMP DEFAULT now()
);
GRANT ALL PRIVILEGES ON TABLE anomalies.detected TO energy_user;
GRANT ALL PRIVILEGES ON SEQUENCE anomalies.detected_id_seq TO energy_user;


-- ============================
-- Recomendaciones (Long Format)
-- ============================
CREATE TABLE IF NOT EXISTS recommendations.generated (
  id BIGSERIAL PRIMARY KEY,
  anomaly_id BIGINT REFERENCES anomalies.detected(id),
  sede_id TEXT REFERENCES raw_data.sedes(sede_id),
  sector TEXT NOT NULL,
  timestamp TIMESTAMP,
  tipo_anomalia TEXT,
  accion TEXT,
  ahorro_estimado NUMERIC,
  explicacion TEXT,
  confianza NUMERIC,
  created_at TIMESTAMP DEFAULT now()
);
GRANT ALL PRIVILEGES ON TABLE recommendations.generated TO energy_user;
GRANT ALL PRIVILEGES ON SEQUENCE recommendations.generated_id_seq TO energy_user;


-- ============================
-- Chat
-- ============================
CREATE TABLE IF NOT EXISTS chat.interactions (
  id BIGSERIAL PRIMARY KEY,
  pregunta TEXT,
  intencion TEXT,
  respuesta TEXT,
  confianza NUMERIC,
  created_at TIMESTAMP DEFAULT now()
);
GRANT ALL PRIVILEGES ON TABLE chat.interactions TO energy_user;
GRANT ALL PRIVILEGES ON SEQUENCE chat.interactions_id_seq TO energy_user;

