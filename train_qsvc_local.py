#!/usr/bin/env python3
"""
train_qsvc_local.py

Entrena un QSVC con FidelityQuantumKernel usando un sampler local.
Genera:
 - Modelo entrenado (modelo_qsvc_local.joblib)
 - Scaler (scaler_qsvc_local.joblib)
 - Estadísticas resumen en CSV
 - Gráfico PCA de superposición cuántica (PNG)
 - Métricas de accuracy y correlación sensores–clase
 - Gráfico de clustering emergente
 - Gráfico de importancia de sensores
"""

import os
import sys
import logging
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from joblib import dump
import mysql.connector
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Qiskit
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms.classifiers import QSVC
from qiskit.algorithms.state_fidelities import ComputeUncompute

# ---------- Config ----------
TRAIN_SIZE = 300   # ⚡ ajusta entre 200–500
RANDOM_STATE = 42
FEATURE_COLUMNS = [
    "temperatura", "humedad", "ph",
    "nitrógeno", "fósforo", "potasio", "conductividad"
]
MODEL_OUT = "modelo_qsvc_local.joblib"
SCALER_OUT = "scaler_qsvc_local.joblib"
STATS_CSV = "estadisticas_entrenamiento.csv"
SUPERPOSICION_PNG = "superposicion_pca.png"
CLUSTER_PNG = "clustering_emergente.png"
IMPORTANCIA_PNG = "importancia_sensores.png"
# ----------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ----------- Helpers -----------

def cargar_env():
    load_dotenv()
    logging.info("Variables de entorno cargadas desde .env")

def conectar_bd():
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306))
    )
    return conn

def leer_datos(conn):
    query = f"""
        SELECT s.tipo_sensor AS tipo_sensor, l.valor, l.fecha_lectura
        FROM lecturas_sensor l
        JOIN dispositivos_sensor d ON l.id_dispositivo_sensor = d.id_dispositivo_sensor
        JOIN sensores s ON d.id_sensor = s.id_sensor
        ORDER BY l.fecha_lectura ASC
        LIMIT {TRAIN_SIZE * 10}
    """
    df = pd.read_sql(query, conn)
    logging.info("Registros leídos: %d", len(df))
    return df

def preparar_dataset(df):
    df["fecha_lectura"] = pd.to_datetime(df["fecha_lectura"])
    df["fecha_minuto"] = df["fecha_lectura"].dt.floor("1min")

    df_pivot = df.pivot_table(
        index="fecha_minuto",
        columns="tipo_sensor",
        values="valor",
        aggfunc="mean"
    ).reset_index()

    rename_map = {"pH": "ph", "nitrogeno": "nitrógeno", "fosforo": "fósforo"}
    df_pivot = df_pivot.rename(columns=rename_map)

    df_clean = df_pivot.dropna(how="all", subset=FEATURE_COLUMNS).copy()
    df_clean[FEATURE_COLUMNS] = df_clean[FEATURE_COLUMNS].fillna(df_clean[FEATURE_COLUMNS].mean())

    df_clean = df_clean.head(TRAIN_SIZE)

    X = df_clean[FEATURE_COLUMNS].astype(float).values
    y = np.floor(np.mean(X, axis=1) / 10).astype(int)

    if len(X) == 0:
        raise ValueError("❌ No hay datos suficientes después del pivot y limpieza.")

    return X, y, df_clean

def escalar_y_dividir(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

# ----------- Quantum Helpers -----------

def entrenar_qsvc(X_train, y_train):
    feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2)
    sampler = Sampler()
    fidelity = ComputeUncompute(sampler)
    qkernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    model = QSVC(quantum_kernel=qkernel)
    model.fit(X_train, y_train)
    return model, qkernel

# ----------- Output Helpers -----------

def generar_estadisticas(df, y, filename):
    df_stats = df[FEATURE_COLUMNS].describe().T
    df_stats["clase_media"] = pd.Series(y).mean()
    df_stats.to_csv(filename, index=True)
    logging.info("📊 Estadísticas guardadas en %s", filename)

def graficar_superposicion(X_scaled, y, filename):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="plasma", alpha=0.7)
    plt.colorbar(scatter, label="Clases (colapso esperado)")
    plt.title("🌌 Superposición cuántica (PCA de embeddings)")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    logging.info("🌌 Gráfico de superposición guardado en %s", filename)

def graficar_clustering(X_scaled, filename):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE)
    clusters = kmeans.fit_predict(X_pca)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label="Clusters emergentes")
    plt.title("🌱 Clustering emergente en embeddings cuánticos")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    logging.info("🌱 Gráfico de clustering guardado en %s", filename)

def graficar_importancia_sensores(X_scaled, y, filename):
    corr = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS).corrwith(pd.Series(y))
    corr.plot(kind="bar", figsize=(8,5), color="teal", alpha=0.7)
    plt.title("📊 Correlación de sensores con clases (colapso)")
    plt.ylabel("Correlación")
    plt.grid(True, axis="y")
    plt.savefig(filename)
    plt.close()
    logging.info("📊 Gráfico de importancia de sensores guardado en %s", filename)

# ----------- Main -----------

def main():
    try:
        cargar_env()
        conn = conectar_bd()
        df = leer_datos(conn)
        conn.close()

        X, y, df_clean = preparar_dataset(df)
        X_scaled, y, scaler = escalar_y_dividir(X, y)

        model, qkernel = entrenar_qsvc(X_scaled, y)

        dump(model, MODEL_OUT)
        dump(scaler, SCALER_OUT)
        logging.info("✅ Modelo y scaler guardados")

        # --- Métricas ---
        y_pred = model.predict(X_scaled)
        acc = accuracy_score(y, y_pred)
        logging.info("🎯 Accuracy del modelo: %.2f%%", acc*100)
        logging.info("\n" + classification_report(y, y_pred))

        # --- Salidas gráficas ---
        generar_estadisticas(df_clean, y, STATS_CSV)
        graficar_superposicion(X_scaled, y, SUPERPOSICION_PNG)
        graficar_clustering(X_scaled, CLUSTER_PNG)
        graficar_importancia_sensores(X_scaled, y, IMPORTANCIA_PNG)

        logging.info("🎉 Entrenamiento y análisis completados. Archivos listos.")

    except Exception as e:
        logging.exception("❌ Error en entrenamiento: %s", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
