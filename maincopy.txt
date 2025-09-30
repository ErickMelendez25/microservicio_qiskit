from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import numpy as np
import pandas as pd
import mysql.connector
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel, QuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit_aer import Aer
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import random
from collections import Counter
from joblib import load


# Cargar variables de entorno
load_dotenv()

# 🔧 Inicializar FastAPI
app = FastAPI()

# ✅ CORS middleware (🔒 muy importante que esté justo después de app = FastAPI())


origins = [
    "https://qiskit.academionlinegpt.com",
    "http://localhost:5173", 
    "https://tinka.academionlinegpt.com",
    "https://microservicioqiskit-production.up.railway.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



#app.add_middleware(
  #  CORSMiddleware,
   # allow_origins=["https://tinka.academionlinegpt.com","http://localhost:5173"],
   # allow_credentials=True,
   # allow_methods=["*"],
    #allow_headers=["*"],
#)



# 📁 Crear carpetas necesarias
os.makedirs("static", exist_ok=True)
os.makedirs("graficos", exist_ok=True)
os.makedirs("interpretaciones", exist_ok=True)

# 🖼️ Montar archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")


# 🧾 Modelos
class ZonaRequest(BaseModel):
    zona_id: int

class DummyRequest(BaseModel):
    ejecutar: bool = True
    
RANGOS_CULTIVOS = {
    "papa": {
        "temperatura": (15, 18),
        "humedad": (60, 80),
        "ph": (5.0, 6.5),
        "nitrógeno": (30, 60),
        "fósforo": (20, 40),
        "potasio": (150, 300),
        "conductividad": (0.2, 1.5)
    },
    "maiz": {
        "temperatura": (20, 30),
        "humedad": (50, 70),
        "ph": (5.5, 7.5),
        "nitrógeno": (30, 50),
        "fósforo": (15, 30),
        "potasio": (100, 200),
        "conductividad": (0.2, 1.8)
    },
    "oca": {
        "temperatura": (12, 18),
        "humedad": (60, 80),
        "ph": (5.0, 6.5),
        "nitrógeno": (25, 50),
        "fósforo": (15, 35),
        "potasio": (100, 250),
        "conductividad": (0.1, 1.5)
    },
    "haba": {
        "temperatura": (12, 20),
        "humedad": (60, 75),
        "ph": (6.0, 7.0),
        "nitrógeno": (20, 40),
        "fósforo": (15, 30),
        "potasio": (120, 250),
        "conductividad": (0.1, 2.0)
    }
}



# ✅ Endpoint de prueba
@app.get("/")
def root():
    return {"mensaje": "✅ API de Tinka está corriendo correctamente"}

@app.options("/api/ejecutarmodelos")
def preflight_handler():
    return {"message": "Preflight received"}



# 🔍 MODELO 1: Análisis por zona
@app.post("/ejecut")
def ejecutar_modelo(data: ZonaRequest):
    zona_id = data.zona_id

    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", 3306)),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT s.tipo AS tipo_sensor, l.valor
        FROM lecturas_sensor l
        JOIN dispositivos_sensor d ON l.dispositivo_id = d.id
        JOIN sensores s ON d.sensor_id = s.id
        WHERE d.zona_id = %s
    """, (zona_id,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    datos = {}
    for row in rows:
        tipo = row['tipo_sensor'].lower()
        valor = float(row['valor'])
        datos.setdefault(tipo, []).append(valor)

    tipos = ['temperatura', 'humedad', 'ph', 'nitrógeno', 'fósforo', 'potasio', 'conductividad']

    num = len(datos.get('temperatura', []))
    if num == 0:
        return {"error": "No hay datos suficientes para ejecutar el modelo."}

    X = [[datos.get(tipo, [0]*num)[i] for tipo in tipos] for i in range(num)]
    X = StandardScaler().fit_transform(np.array(X))

    feature_map = ZZFeatureMap(feature_dimension=X.shape[1], reps=2)
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    matrix = kernel.evaluate(x_vec=X)

    clustering = SpectralClustering(n_clusters=2, affinity='precomputed')
    labels = clustering.fit_predict(matrix)

    graficar_clusters(X, labels, zona_id)
    graficar_matriz_kernel(matrix, zona_id)
    graficar_heatmap(X, tipos, zona_id)
    graficar_matriz_confusion(labels, X, tipos, zona_id)

    generar_interpretacion(zona_id, labels, X, tipos)



    return {
        "zona_id": zona_id,
        "clusters": labels.tolist(),
        "tipos": tipos,
        "valores": X.tolist(),
        "imagenes": {
            "clusters": f"/grafico/cluster/{zona_id}",
            "kernel": f"/grafico/kernel/{zona_id}",
            "heatmap": f"/grafico/heatmap/{zona_id}",
            "confusion": f"/grafico/confusion/{zona_id}"
        },
        "interpretacion": f"/interpretacion/{zona_id}"
    }



# 🔮 MODELO 2: Predicción cuántica de La Tinka
@app.post("/api/ejecutarmodelos")
def ejecutar_modelo_loteria(_: DummyRequest):
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            port=int(os.getenv("DB_PORT", 3306))
        )
        cursor = conn.cursor(dictionary=True)

        # Traer sorteos y procesar datos
        cursor.execute("SELECT * FROM sorteos ORDER BY fecha ASC")
        sorteos = cursor.fetchall()
        df = pd.DataFrame(sorteos).dropna(subset=[f"bola{i}" for i in range(1, 7)])
        
        # Análisis estadístico
        todas_bolas = df[[f"bola{i}" for i in range(1, 7)]].values.flatten()
        frecuencia = Counter(todas_bolas)
        pares, trios = Counter(), Counter()
        for _, row in df.iterrows():
            bolas = [row[f"bola{i}"] for i in range(1, 7)]
            pares.update(combinations(sorted(bolas), 2))
            trios.update(combinations(sorted(bolas), 3))

        # Codificador directo (6 bolas → vector de 6 números)
        def quantum_encode(bolas):
            return [float(b) for b in bolas]  # <-- Compatible con el modelo entrenado y serialización
        
        import symengine
        print("✅ Versión de SymEngine:", symengine.__version__)

        # Cargar modelo
        print("✅ Intentando cargar modelo y scaler...")
        model = load("modelo_qsvc_fast.joblib")
        scaler = load("scaler_qsvc_tinka.joblib")
        print("✅ Modelo y scaler cargados correctamente.")

        # Generar combinaciones
        top_bolas = [n for n, _ in frecuencia.most_common(25)]
        combinaciones = random.sample(list(combinations(top_bolas, 6)), 100)

        # Eliminar predicciones anteriores
        cursor.execute("DELETE FROM predicciones")

        predicciones = []

        for bolas in combinaciones:
            bolas = sorted(bolas)
            vec = quantum_encode(bolas)
            vec_scaled = scaler.transform([vec])
            pred = model.predict(vec_scaled)[0]
            media = sum(bolas) / 6
            clase_real = media // 10
            probabilidad = 1.0 if pred == clase_real else (0.9 if abs(pred - clase_real) <= 1 else 0.5)

            top_pares = sum([pares[p] for p in combinations(bolas, 2)])
            top_trios = sum([trios[t] for t in combinations(bolas, 3)])
            boliyapa = random.choice([n for n in range(1, 51) if n not in bolas])

            # Conversión explícita a tipos nativos
            registro = {
                **{f"bola{i+1}": int(bolas[i]) for i in range(6)},
                "boliyapa": int(boliyapa),
                "probabilidad": float(round(probabilidad, 2)),
                "modelo_version": "Qiskit-v3",
                "pares": int(top_pares),
                "trios": int(top_trios)
            }

            cursor.execute("""
                INSERT INTO predicciones (
                    bola1, bola2, bola3, bola4, bola5, bola6,
                    boliyapa, probabilidad, modelo_version, pares, trios
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, tuple(registro.values()))
            
            predicciones.append(registro)

        conn.commit()

        # Retornar las 15 mejores combinaciones al frontend
        predicciones_ordenadas = sorted(
            predicciones,
            key=lambda x: (-x["probabilidad"], -x["pares"], -x["trios"])
        )[:15]

        return {
            "status": "✅ Predicciones generadas con modelo cuántico",
            "predicciones": predicciones_ordenadas
        }

    except Exception as e:
        if conn:
            conn.rollback()
        print("❌ ERROR AL EJECUTAR MODELO:", e)  # 👈 esto te da el error exacto en consola
        return {"error": str(e)}

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()



# 🖼️ Rutas para ver imágenes y textos
@app.get("/grafico/cluster/{zona_id}")
def ver_cluster(zona_id: int):
    return FileResponse(os.path.join("graficos", f"clusters_zona_{zona_id}.png"), media_type="image/png")

@app.get("/grafico/kernel/{zona_id}")
def ver_kernel(zona_id: int):
    return FileResponse(os.path.join("graficos", f"kernel_zona_{zona_id}.png"), media_type="image/png")
@app.get("/grafico/heatmap/{zona_id}")
def ver_heatmap(zona_id: int):
    return FileResponse(os.path.join("graficos", f"heatmap_zona_{zona_id}.png"), media_type="image/png")

@app.get("/grafico/confusion/{zona_id}")
def ver_confusion(zona_id: int):
    return FileResponse(os.path.join("graficos", f"confusion_zona_{zona_id}.png"), media_type="image/png")

@app.get("/interpretacion/{zona_id}")
def ver_interpretacion(zona_id: int):
    return FileResponse(os.path.join("interpretaciones", f"interpretacion_zona_{zona_id}.txt"), media_type="text/plain")


# 📊 Funciones auxiliares
def graficar_clusters(X, labels, zona_id):
    pca = PCA(n_components=2)
    X_reducido = pca.fit_transform(X)
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X_reducido[:, 0], X_reducido[:, 1], c=labels, cmap='viridis', s=50)
    plt.title(f"Clústeres - Zona {zona_id}")
    plt.colorbar(scatter)
    plt.grid(True)
    plt.savefig(f"graficos/clusters_zona_{zona_id}.png")
    plt.close()

def graficar_matriz_kernel(matrix, zona_id):
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, cmap='coolwarm', square=True, cbar=True)
    plt.title(f"Matriz Kernel - Zona {zona_id}")
    plt.savefig(f"graficos/kernel_zona_{zona_id}.png")
    plt.close()


def graficar_heatmap(X, tipos, zona_id):
    df = pd.DataFrame(X, columns=tipos)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f"Mapa de Calor de Indicadores - Zona {zona_id}")
    plt.savefig(f"graficos/heatmap_zona_{zona_id}.png")
    plt.close()

def graficar_matriz_confusion(labels, X, tipos, zona_id):
    df = pd.DataFrame(X, columns=tipos)
    df['cluster'] = labels

    # Calculamos el promedio de cada indicador por cluster
    mean_by_cluster = df.groupby('cluster')[tipos].mean()

    # Creamos un heatmap de esos promedios
    plt.figure(figsize=(10, 6))
    sns.heatmap(mean_by_cluster, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(f"Promedios de Indicadores por Clúster - Zona {zona_id}")
    plt.ylabel("Clúster")
    plt.xlabel("Indicadores")
    plt.tight_layout()
    plt.savefig(f"graficos/confusion_zona_{zona_id}.png")
    plt.close()


def generar_interpretacion(zona_id, labels, X, tipos):
    
    total = len(labels)
    grupo_0 = sum(1 for l in labels if l == 0)
    grupo_1 = total - grupo_0

    texto = f"""
INTERPRETACIÓN DE RESULTADOS - ZONA {zona_id}

Se identificaron 2 grupos:
- Grupo 0: {grupo_0} muestras ({(grupo_0 / total) * 100:.1f}%)
- Grupo 1: {grupo_1} muestras ({(grupo_1 / total) * 100:.1f}%)

Variables analizadas: {', '.join(tipos)}.
"""

    # Cálculo de promedios de la zona
    # Supongamos que 'tipos' es la lista de nombres de indicadores
    df = pd.DataFrame(X, columns=tipos)


    # Añade las etiquetas como columna
    df["cluster"] = labels

    # Calcula promedio general por indicador
    promedios = df.groupby("cluster").mean().mean()


    # Comparación con cada cultivo
    def evaluar_cultivo(cultivo, rangos, promedios):
        puntaje = 0
        total = 0
        detalles = []
        for tipo in tipos:
            if tipo in rangos:
                valor = promedios.get(tipo)
                if valor is None:
                    continue
                min_val, max_val = rangos[tipo]
                if min_val <= valor <= max_val:
                    puntaje += 1
                else:
                    detalles.append(f"{tipo}: {valor:.2f} fuera de rango ({min_val}-{max_val})")
                total += 1
        return (puntaje / total) * 100 if total > 0 else 0, detalles



    recomendaciones = []
    for cultivo, rangos in RANGOS_CULTIVOS.items():
        match_pct, detalles = evaluar_cultivo(cultivo, rangos, promedios)
        recomendaciones.append((cultivo, match_pct))

    recomendaciones.sort(key=lambda x: -x[1])
    texto += "\n\nRECOMENDACIÓN DE CULTIVOS SEGÚN INDICADORES:\n"
    for cultivo, pct in recomendaciones:
        estado = "✅ Recomendado" if pct >= 70 else "⚠️ Parcial" if pct >= 40 else "❌ No recomendado"
        texto += f"- {cultivo.upper():<6}: {pct:.1f}% coincidencia → {estado}\n"
        
    texto += "\n\nANÁLISIS POR CLÚSTER:\n"
    for cluster_id in sorted(df["cluster"].unique()):
        cluster_df = df[df["cluster"] == cluster_id]
        promedios_cluster = cluster_df[tipos].mean()
    
        recomendaciones_cluster = []
        for cultivo, rangos in RANGOS_CULTIVOS.items():
            match_pct, detalles = evaluar_cultivo(cultivo, rangos, promedios_cluster)
            recomendaciones_cluster.append((cultivo, match_pct))


        recomendaciones_cluster.sort(key=lambda x: -x[1])
        texto += f"\n➡️ Cluster {cluster_id}:\n"
        for cultivo, pct in recomendaciones_cluster:
            estado = "✅ Recomendado" if pct >= 70 else "⚠️ Parcial" if pct >= 40 else "❌ No recomendado"
            texto += f"   - {cultivo.upper():<6}: {pct:.1f}% → {estado}\n"


    with open(f"interpretaciones/interpretacion_zona_{zona_id}.txt", "w", encoding="utf-8") as f:
        f.write(texto.strip())


