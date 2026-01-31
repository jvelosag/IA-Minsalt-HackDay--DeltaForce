CREATE SCHEMA IF NOT EXISTS analytics;

CREATE OR REPLACE VIEW analytics.vw_energia_por_sector AS
SELECT
    c.reading_id,
    c."timestamp",
    c.sede_id,
    'total'::text AS sector,
    c.energia_total_kwh::numeric AS kwh,

    -- Variables que sirven para Power BI / ML (las dejo “passthrough”)
    c.potencia_total_kw,
    c.temperatura_exterior_c,
    c.ocupacion_pct,
    c.hora,
    c.dia_semana,
    c.dia_nombre,
    c.mes,
    c.trimestre,
    c.ano,
    c.periodo_academico,
    c.es_fin_semana,
    c.es_festivo,
    c.es_semana_parciales,
    c.es_semana_finales
FROM raw_data.consumo c
WHERE c.energia_total_kwh IS NOT NULL

UNION ALL
SELECT
    c.reading_id,
    c."timestamp",
    c.sede_id,
    'comedores'::text AS sector,
    c.energia_comedor_kwh::numeric AS kwh,
    c.potencia_total_kw,
    c.temperatura_exterior_c,
    c.ocupacion_pct,
    c.hora,
    c.dia_semana,
    c.dia_nombre,
    c.mes,
    c.trimestre,
    c.ano,
    c.periodo_academico,
    c.es_fin_semana,
    c.es_festivo,
    c.es_semana_parciales,
    c.es_semana_finales
FROM raw_data.consumo c
WHERE c.energia_comedor_kwh IS NOT NULL

UNION ALL
SELECT
    c.reading_id,
    c."timestamp",
    c.sede_id,
    'salones'::text AS sector,
    c.energia_salones_kwh::numeric AS kwh,
    c.potencia_total_kw,
    c.temperatura_exterior_c,
    c.ocupacion_pct,
    c.hora,
    c.dia_semana,
    c.dia_nombre,
    c.mes,
    c.trimestre,
    c.ano,
    c.periodo_academico,
    c.es_fin_semana,
    c.es_festivo,
    c.es_semana_parciales,
    c.es_semana_finales
FROM raw_data.consumo c
WHERE c.energia_salones_kwh IS NOT NULL

UNION ALL
SELECT
    c.reading_id,
    c."timestamp",
    c.sede_id,
    'laboratorios'::text AS sector,
    c.energia_laboratorios_kwh::numeric AS kwh,
    c.potencia_total_kw,
    c.temperatura_exterior_c,
    c.ocupacion_pct,
    c.hora,
    c.dia_semana,
    c.dia_nombre,
    c.mes,
    c.trimestre,
    c.ano,
    c.periodo_academico,
    c.es_fin_semana,
    c.es_festivo,
    c.es_semana_parciales,
    c.es_semana_finales
FROM raw_data.consumo c
WHERE c.energia_laboratorios_kwh IS NOT NULL

UNION ALL
SELECT
    c.reading_id,
    c."timestamp",
    c.sede_id,
    'auditorios'::text AS sector,
    c.energia_auditorios_kwh::numeric AS kwh,
    c.potencia_total_kw,
    c.temperatura_exterior_c,
    c.ocupacion_pct,
    c.hora,
    c.dia_semana,
    c.dia_nombre,
    c.mes,
    c.trimestre,
    c.ano,
    c.periodo_academico,
    c.es_fin_semana,
    c.es_festivo,
    c.es_semana_parciales,
    c.es_semana_finales
FROM raw_data.consumo c
WHERE c.energia_auditorios_kwh IS NOT NULL

UNION ALL
SELECT
    c.reading_id,
    c."timestamp",
    c.sede_id,
    'oficinas'::text AS sector,
    c.energia_oficinas_kwh::numeric AS kwh,
    c.potencia_total_kw,
    c.temperatura_exterior_c,
    c.ocupacion_pct,
    c.hora,
    c.dia_semana,
    c.dia_nombre,
    c.mes,
    c.trimestre,
    c.ano,
    c.periodo_academico,
    c.es_fin_semana,
    c.es_festivo,
    c.es_semana_parciales,
    c.es_semana_finales
FROM raw_data.consumo c
WHERE c.energia_oficinas_kwh IS NOT NULL;


---------------------------------------------------

CREATE OR REPLACE VIEW analytics.vw_agua_co2 AS
SELECT
    reading_id,
    "timestamp",
    sede_id,
    agua_litros,
    co2_kg,
    energia_total_kwh,
    potencia_total_kw,
    temperatura_exterior_c,
    ocupacion_pct,
    hora,
    dia_semana,
    dia_nombre,
    mes,
    trimestre,
    ano,
    periodo_academico,
    es_fin_semana,
    es_festivo,
    es_semana_parciales,
    es_semana_finales
FROM raw_data.consumo;

--------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_consumo_ts
ON raw_data.consumo ("timestamp");

CREATE INDEX IF NOT EXISTS idx_consumo_sede_ts
ON raw_data.consumo (sede_id, "timestamp");

