-- Vistas de Analytics v2 (para esquema "ancho")
-- Estas vistas están diseñadas para ser consumidas por Power BI
-- y permitir un análisis exploratorio de los datos crudos.

\c core_db;

-- ============================
-- 1. Vista de Consumo por Hora y Sector
-- Transforma los datos a un formato "largo" para facilitar la agregación por sector.
-- Útil para identificar patrones horarios y picos de consumo.
-- ============================
CREATE OR REPLACE VIEW analytics.consumo_horario_sector AS
WITH consumo_largo AS (
    -- Desanchoar (unpivot) la tabla de consumo
    SELECT
        c.timestamp,
        c.sede_id,
        s.sede,
        'Comedores' AS sector,
        c.energia_comedor_kwh AS consumo_kwh
    FROM raw_data.consumo c
    JOIN raw_data.sedes s ON c.sede_id = s.sede_id

    UNION ALL

    SELECT
        c.timestamp,
        c.sede_id,
        s.sede,
        'Salones' AS sector,
        c.energia_salones_kwh AS consumo_kwh
    FROM raw_data.consumo c
    JOIN raw_data.sedes s ON c.sede_id = s.sede_id

    UNION ALL

    SELECT
        c.timestamp,
        c.sede_id,
        s.sede,
        'Laboratorios' AS sector,
        c.energia_laboratorios_kwh AS consumo_kwh
    FROM raw_data.consumo c
    JOIN raw_data.sedes s ON c.sede_id = s.sede_id

    UNION ALL

    SELECT
        c.timestamp,
        c.sede_id,
        s.sede,
        'Auditorios' AS sector,
        c.energia_auditorios_kwh AS consumo_kwh
    FROM raw_data.consumo c
    JOIN raw_data.sedes s ON c.sede_id = s.sede_id

    UNION ALL

    SELECT
        c.timestamp,
        c.sede_id,
        s.sede,
        'Oficinas' AS sector,
        c.energia_oficinas_kwh AS consumo_kwh
    FROM raw_data.consumo c
    JOIN raw_data.sedes s ON c.sede_id = s.sede_id
)
SELECT
    l.sede,
    l.sector,
    c.hora,
    c.dia_nombre,
    c.es_fin_semana,
    c.es_festivo,
    AVG(l.consumo_kwh) AS consumo_promedio_kwh,
    SUM(l.consumo_kwh) AS consumo_total_kwh,
    MAX(l.consumo_kwh) AS consumo_maximo_kwh
FROM consumo_largo l
JOIN raw_data.consumo c ON l.timestamp = c.timestamp AND l.sede_id = c.sede_id
WHERE l.consumo_kwh IS NOT NULL AND l.consumo_kwh > 0
GROUP BY
    l.sede,
    l.sector,
    c.hora,
    c.dia_nombre,
    c.es_fin_semana,
    c.es_festivo
ORDER BY
    l.sede,
    l.sector,
    c.hora;

-- ============================
-- 2. Vista de Métricas por Sede
-- Agrega métricas clave a nivel de sede para un dashboard de alto nivel.
-- ============================
CREATE OR REPLACE VIEW analytics.metricas_sede AS
SELECT
    s.sede_id,
    s.sede,
    s.ciudad,
    s.num_estudiantes,
    s.area_m2,
    SUM(c.energia_total_kwh) AS consumo_total_sede_kwh,
    AVG(c.energia_total_kwh) AS consumo_promedio_hora_kwh,
    SUM(c.co2_kg) AS co2_total_kg,
    SUM(c.agua_litros) AS agua_total_litros,
    (SUM(c.energia_total_kwh) / s.num_estudiantes) AS kwh_por_estudiante,
    (SUM(c.energia_total_kwh) / s.area_m2) AS kwh_por_m2
FROM raw_data.sedes s
JOIN raw_data.consumo c ON s.sede_id = c.sede_id
WHERE c.energia_total_kwh IS NOT NULL AND c.energia_total_kwh > 0
GROUP BY
    s.sede_id,
    s.sede,
    s.ciudad,
    s.num_estudiantes,
    s.area_m2;

-- ============================
-- 3. Vista de Consumo por Contexto
-- Analiza cómo diferentes eventos y periodos afectan el consumo energético.
-- ============================
CREATE OR REPLACE VIEW analytics.consumo_contexto AS
SELECT
    c.periodo_academico,
    c.es_festivo,
    c.es_fin_semana,
    c.es_semana_parciales,
    c.es_semana_finales,
    s.sede,
    SUM(c.energia_total_kwh) AS consumo_total_kwh,
    AVG(c.energia_total_kwh) AS consumo_promedio_kwh,
    AVG(c.ocupacion_pct) AS ocupacion_promedio_pct
FROM raw_data.consumo c
JOIN raw_data.sedes s ON c.sede_id = s.sede_id
WHERE c.energia_total_kwh IS NOT NULL AND c.energia_total_kwh > 0
GROUP BY
    c.periodo_academico,
    c.es_festivo,
    c.es_fin_semana,
    c.es_semana_parciales,
    c.es_semana_finales,
    s.sede
ORDER BY
    s.sede,
    c.periodo_academico;

-- Grant permissions for all views to the user
GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO energy_user;