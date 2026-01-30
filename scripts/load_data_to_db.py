# scripts/load_data_to_db.py

import pandas as pd
import psycopg2
from io import StringIO
import os
from dotenv import load_dotenv

# --- Configuration ---
SEDES_CSV_PATH = "consumos_uptc_hackday/sedes_uptc.csv"
CONSUMOS_CSV_PATH = "consumos_uptc_hackday/consumos_uptc.csv"

def get_db_connection():
    """Establishes and returns a database connection."""
    load_dotenv()
    db_host = os.getenv("DB_HOST", "localhost")
    db_name = os.getenv("POSTGRES_DB_CORE")
    db_user = os.getenv("POSTGRES_USER")
    db_password = os.getenv("POSTGRES_PASSWORD")

    if not all([db_name, db_user, db_password]):
        raise ValueError("Missing database environment variables.")

    print(f"Connecting to database '{db_name}' on host '{db_host}'...")
    conn = psycopg2.connect(
        host=db_host,
        database=db_name,
        user=db_user,
        password=db_password
    )
    print("Connection successful.")
    return conn

def copy_from_dataframe(conn, df, table):
    """
    Efficiently loads a pandas DataFrame into a PostgreSQL table using COPY.
    
    :param conn: Active psycopg2 connection.
    :param df: Pandas DataFrame to load.
    :param table: Target table name in 'schema.table' format.
    """
    buffer = StringIO()
    df.to_csv(buffer, index=False, header=False, sep='|')
    buffer.seek(0)
    
    cursor = conn.cursor()
    try:
        print(f"Truncating table {table}...")
        cursor.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE;")
        print(f"Loading data into {table} using COPY...")
        # Use a | separator as it's less likely to be in the data
        cursor.copy_expert(f"COPY {table} FROM STDIN WITH (FORMAT CSV, DELIMITER '|')", buffer)
        conn.commit()
        print(f"Successfully loaded {len(df)} rows into {table}.")
    except Exception as e:
        conn.rollback()
        print(f"Error loading data into {table}: {e}")
        raise
    finally:
        cursor.close()

def load_sedes_data(conn):
    """Loads data from the sedes CSV into the raw_data.sedes table."""
    print(f"\n--- Loading Sedes Data from {SEDES_CSV_PATH} ---")
    df = pd.read_csv(SEDES_CSV_PATH)
    
    # Ensure column order matches the database schema
    db_columns = [
        'sede_id', 'sede', 'nombre_completo', 'ciudad', 'area_m2', 'num_estudiantes', 
        'num_empleados', 'num_edificios', 'tiene_residencias', 'tiene_laboratorios_pesados', 
        'altitud_msnm', 'temp_promedio_c', 'pct_comedores', 'pct_salones', 
        'pct_laboratorios', 'pct_auditorios', 'pct_oficinas'
    ]
    df = df[db_columns]
    
    copy_from_dataframe(conn, df, 'raw_data.sedes')

def load_consumos_data(conn):
    """Loads data from the consumos CSV into the raw_data.consumo table."""
    print(f"\n--- Loading Consumos Data from {CONSUMOS_CSV_PATH} ---")
    df = pd.read_csv(CONSUMOS_CSV_PATH)

    # Rename columns to match database schema (e.g., 'año' -> 'ano')
    df.rename(columns={'año': 'ano'}, inplace=True)

    # FIX: Standardize 'sede_id' to match the format in 'sedes' table (e.g., 'uptc-dui' -> 'UPTC_DUI')
    df['sede_id'] = df['sede_id'].str.replace('-', '_').str.upper()

    # Ensure column order matches the database schema
    db_columns = [
        'reading_id', 'timestamp', 'sede_id', 'energia_total_kwh', 'potencia_total_kw', 
        'co2_kg', 'energia_comedor_kwh', 'energia_salones_kwh', 'energia_laboratorios_kwh', 
        'energia_auditorios_kwh', 'energia_oficinas_kwh', 'agua_litros', 
        'temperatura_exterior_c', 'ocupacion_pct', 'hora', 'dia_semana', 'dia_nombre', 
        'mes', 'trimestre', 'ano', 'periodo_academico', 'es_fin_semana', 
        'es_festivo', 'es_semana_parciales', 'es_semana_finales'
    ]
    df = df[db_columns]

    copy_from_dataframe(conn, df, 'raw_data.consumo')


def main():
    """Main function to orchestrate the data loading process."""
    conn = None
    try:
        conn = get_db_connection()
        # The order is important due to foreign key constraints
        load_sedes_data(conn)
        load_consumos_data(conn)
        print("\n✅ Data loading process completed successfully.")
    except Exception as e:
        print(f"\n❌ An error occurred during the data loading process: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()