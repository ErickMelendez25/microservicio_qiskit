#!/usr/bin/env python3
"""
train_qsvc_local.py

Entrena un QSVC por zona con FidelityQuantumKernel usando un sampler local.
"""

import os, sys, json, logging
from datetime import datetime
import pandas as pd, numpy as np
from joblib import dump, load
import mysql.connector
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Qiskit
try:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit.primitives import Sampler
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.algorithms.classifiers import QSVC
    from qiskit.algorithms.state_fidelities import ComputeUncompute
    QISKIT_AVAILABLE = True
except Exception:
    QISKIT_AVAILABLE = False

try:
    from qiskit import Aer, transpile
    from qiskit.visualization import plot_bloch_multivector
    AER_AVAILABLE = True
except Exception:
    AER_AVAILABLE = False

# ---------- Config ----------
TRAIN_SIZE = 50
RANDOM_STATE = 42
FEATURE_COLUMNS = ["temperatura", "humedad", "ph", "nitrógeno", "fósforo", "potasio", "conductividad"]
OUTPUT_DIR = "outputs"
# ----------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ----------- DB -----------
def conectar_bd():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306))
    )

def leer_datos(conn, zone_id=None, limit=TRAIN_SIZE*10):
    if zone_id is None:
        query = """
            SELECT s.tipo_sensor, l.valor, l.fecha_lectura
            FROM lecturas_sensor l
            JOIN dispositivos_sensor ds ON l.id_dispositivo_sensor = ds.id_dispositivo_sensor
            JOIN sensores s ON ds.id_sensor = s.id_sensor
            JOIN dispositivos d ON ds.id_dispositivo = d.id_dispositivo
            ORDER BY l.fecha_lectura ASC
            LIMIT %s
        """
        df = pd.read_sql(query, conn, params=(limit,))
    else:
        query = f"""
            SELECT s.tipo_sensor, l.valor, l.fecha_lectura
            FROM lecturas_sensor l
            JOIN dispositivos_sensor ds ON l.id_dispositivo_sensor = ds.id_dispositivo_sensor
            JOIN sensores s ON ds.id_sensor = s.id_sensor
            JOIN dispositivos d ON ds.id_dispositivo = d.id_dispositivo
            WHERE d.zona_agricola_id = %s
            ORDER BY l.fecha_lectura ASC
            LIMIT {limit}
        """
        df = pd.read_sql(query, conn, params=(zone_id,))
    return df

def preparar_dataset(df):
    df["fecha_lectura"] = pd.to_datetime(df["fecha_lectura"])
    df["fecha_minuto"] = df["fecha_lectura"].dt.floor("1min")

    df_pivot = df.pivot_table(
        index="fecha_minuto", columns="tipo_sensor", values="valor", aggfunc="mean"
    ).reset_index()

    rename_map = {"pH": "ph", "nitrogeno": "nitrógeno", "fosforo": "fósforo"}
    df_pivot = df_pivot.rename(columns=rename_map)

    df_clean = df_pivot.dropna(how="all", subset=FEATURE_COLUMNS).copy()
    df_clean[FEATURE_COLUMNS] = df_clean[FEATURE_COLUMNS].fillna(df_clean[FEATURE_COLUMNS].mean())
    df_clean = df_clean.head(TRAIN_SIZE)

    X = df_clean[FEATURE_COLUMNS].astype(float).values
    y = np.floor(np.mean(X, axis=1) / 10).astype(int)
    return X, y, df_clean

def escalar_y_guardar(X, zone_dir, zone_id):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    dump(scaler, os.path.join(zone_dir, f"scaler_qsvc_zone_{zone_id}.joblib"))
    return X_scaled

# ----------- Quantum -----------
def entrenar_qsvc(X_train, y_train):
    feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2)
    sampler = Sampler()
    fidelity = ComputeUncompute(sampler)
    qkernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    model = QSVC(quantum_kernel=qkernel)
    model.fit(X_train, y_train)
    return model, qkernel, feature_map

# ----------- Interpretación Agronómica -----------
def interpretacion_agronomica(values_dict):
    temp = float(values_dict.get("temperatura", np.nan))
    hum = float(values_dict.get("humedad", np.nan))
    ph = float(values_dict.get("ph", np.nan))
    nitr = float(values_dict.get("nitrógeno", np.nan))
    fosf = float(values_dict.get("fósforo", np.nan))
    pot = float(values_dict.get("potasio", np.nan))

    lines, cultivo = [], "General"

    # Riego
    if hum < 40:
        lines.append("💧 Riego recomendado: Sí (humedad baja).")
    elif hum < 55:
        lines.append("💧 Riego recomendado: Monitorear (humedad moderada).")
    else:
        lines.append("💧 Riego no necesario (humedad adecuada).")

    # pH
    if ph < 5.5:
        lines.append("🧪 pH ácido — apto para papa, arándanos.")
        cultivo = "Papa"
    elif ph > 7.5:
        lines.append("🧪 pH alcalino — apto para alfalfa, cebada.")
        cultivo = "Cebada"
    else:
        lines.append("🧪 pH óptimo para maíz, trigo, hortalizas.")
        cultivo = "Maíz"

    # Nutrientes
    nutrient_avg = np.nanmean([nitr, fosf, pot])
    if nutrient_avg < 10:
        lines.append("🌾 Nivel de nutrientes bajo — aplicar fertilización.")
    elif nutrient_avg < 25:
        lines.append("🌾 Nutrientes moderados — suficiente para cultivos medianos.")
    else:
        lines.append("🌾 Nutrientes adecuados.")

    lines.append(f"🌱 Cultivo recomendado: {cultivo}")
    return "\n".join(lines), cultivo

# ----------- Training Zone -----------
def train_zone(zone_id):
    conn = conectar_bd()
    df = leer_datos(conn, zone_id=zone_id)
    conn.close()

    X, y, df_clean = preparar_dataset(df)
    zone_dir = os.path.join(OUTPUT_DIR, f"zone_{zone_id}")
    os.makedirs(zone_dir, exist_ok=True)

    X_scaled = escalar_y_guardar(X, zone_dir, zone_id)
    model, qkernel, feature_map = entrenar_qsvc(X_scaled, y)
    dump(model, os.path.join(zone_dir, f"modelo_qsvc_zone_{zone_id}.joblib"))

    # metadata
    last_ts = df["fecha_lectura"].max()
    meta = {"last_trained_at": str(last_ts), "trained_on": datetime.utcnow().isoformat()}
    with open(os.path.join(zone_dir, "metadata.json"), "w") as f:
        json.dump(meta, f)

    return {"model": True, "last_trained_at": str(last_ts)}

def load_model_for_zone(zone_id):
    zone_dir = os.path.join(OUTPUT_DIR, f"zone_{zone_id}")
    model = load(os.path.join(zone_dir, f"modelo_qsvc_zone_{zone_id}.joblib"))
    scaler = load(os.path.join(zone_dir, f"scaler_qsvc_zone_{zone_id}.joblib"))
    return model, scaler

def predict_from_values(zone_id, values_dict):
    X = np.array([[float(values_dict.get(col, np.nan)) for col in FEATURE_COLUMNS]], dtype=float)
    model, scaler = load_model_for_zone(zone_id)
    X_scaled = scaler.transform(X)
    pred = int(model.predict(X_scaled)[0])
    return pred
