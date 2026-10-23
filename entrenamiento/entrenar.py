"""
Capa de ENTRENAMIENTO
---------------------
Entrena el modelo K-Means de segmentación de clientes y guarda los artefactos
que consume el frontend (Streamlit):

    modelo/kmeans_model.joblib   -> modelo K-Means entrenado
    modelo/scaler.joblib         -> StandardScaler ajustado
    modelo/metadata.json         -> K elegido, métricas (codo y silueta),
                                    centroides y perfiles de negocio
    datos/clientes_segmentados.csv -> dataset original + cluster + perfil

Uso (desde la carpeta raíz del proyecto):
    python entrenamiento/entrenar.py          # K automático (mejor silueta)
    python entrenamiento/entrenar.py --k 5    # K fijo
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Rutas (relativas a la ubicación de este archivo, funciona desde cualquier cwd)
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]
RUTA_DATOS = RAIZ / "datos" / "Mall_Customers.csv"
RUTA_SALIDA_DATOS = RAIZ / "datos" / "clientes_segmentados.csv"
CARPETA_MODELO = RAIZ / "modelo"

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]
RANGO_K = range(1, 11)          # para el método del codo
RANGO_K_SILUETA = range(2, 11)  # la silueta no existe para K=1
RANDOM_STATE = 42
N_INIT = 10

ESTRATEGIAS = {
    "Clientes de alto valor": (
        "Programa VIP y de fidelización: acceso anticipado a colecciones, "
        "eventos exclusivos, atención personalizada. Objetivo: retenerlos."
    ),
    "Clientes con potencial": (
        "Tienen ingreso alto pero gastan poco. Campañas de calidad/premium, "
        "experiencias, beneficios por primera compra en marcas de lujo y "
        "comunicación basada en valor (no en descuento). Objetivo: activarlos."
    ),
    "Clientes promedio": (
        "Segmento más grande. Promociones por temporada, combos, programa de "
        "puntos y venta cruzada para subir el ticket promedio."
    ),
    "Clientes impulsivos": (
        "Ingreso bajo y gasto alto (mayoría jóvenes). Redes sociales, "
        "tendencias, flash sales y pagos a cuotas, cuidando no "
        "sobreendeudarlos. Objetivo: mantener frecuencia."
    ),
    "Clientes de bajo valor": (
        "Ingreso y gasto bajos. Comunicación de bajo costo (email, WhatsApp), "
        "descuentos en productos básicos y ofertas de precio. No priorizar "
        "inversión publicitaria alta."
    ),
}


def asignar_perfil(z_ingreso: float, z_gasto: float, umbral: float = 0.5) -> str:
    """Asigna un perfil de negocio a un centroide según su posición
    (en unidades estandarizadas) respecto a la media del dataset."""
    if abs(z_ingreso) < umbral and abs(z_gasto) < umbral:
        return "Clientes promedio"
    if z_ingreso >= 0 and z_gasto >= 0:
        return "Clientes de alto valor"
    if z_ingreso >= 0 and z_gasto < 0:
        return "Clientes con potencial"
    if z_ingreso < 0 and z_gasto < 0:
        return "Clientes de bajo valor"
    return "Clientes impulsivos"


def evaluar_k(X):
    """Calcula inercia (codo) y coeficiente de silueta para cada K."""
    inercias, siluetas = {}, {}
    for k in RANGO_K:
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit(X)
        inercias[k] = float(km.inertia_)
        if k in RANGO_K_SILUETA:
            siluetas[k] = float(silhouette_score(X, km.labels_))
    return inercias, siluetas


def main():
    parser = argparse.ArgumentParser(description="Entrena K-Means sobre Mall Customers")
    parser.add_argument("--k", type=int, default=None,
                        help="Número de clusters. Si se omite, se usa el K con mejor silueta.")
    args = parser.parse_args()

    # 1. Datos ---------------------------------------------------------------
    print(f"[1/5] Cargando datos desde {RUTA_DATOS}")
    df = pd.read_csv(RUTA_DATOS)
    X_raw = df[FEATURES]

    # 2. Preprocesamiento ----------------------------------------------------
    print("[2/5] Escalando variables con StandardScaler")
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # 3. Selección de K ------------------------------------------------------
    print("[3/5] Evaluando K con método del codo y coeficiente de silueta")
    inercias, siluetas = evaluar_k(X)
    k_silueta = max(siluetas, key=siluetas.get)
    k = args.k if args.k else k_silueta
    for kk in RANGO_K:
        s = f"{siluetas[kk]:.4f}" if kk in siluetas else "  -   "
        marca = "  <- K elegido" if kk == k else ""
        print(f"      K={kk:2d}  inercia={inercias[kk]:8.2f}  silueta={s}{marca}")

    # 4. Modelo final --------------------------------------------------------
    print(f"[4/5] Entrenando K-Means final con K={k}")
    modelo = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit(X)
    df["Cluster"] = modelo.labels_

    centroides_z = modelo.cluster_centers_
    centroides = scaler.inverse_transform(centroides_z)

    clusters = []
    for i in range(k):
        perfil = asignar_perfil(centroides_z[i][0], centroides_z[i][1])
        miembros = df[df["Cluster"] == i]
        clusters.append({
            "cluster": i,
            "perfil": perfil,
            "estrategia": ESTRATEGIAS[perfil],
            "n_clientes": int(len(miembros)),
            "porcentaje": round(100 * len(miembros) / len(df), 1),
            "centroide": {
                FEATURES[0]: round(float(centroides[i][0]), 2),
                FEATURES[1]: round(float(centroides[i][1]), 2),
            },
            "edad_promedio": round(float(miembros["Age"].mean()), 1),
            "pct_mujeres": round(100 * float((miembros["Gender"] == "Female").mean()), 1),
        })
    mapa_perfil = {c["cluster"]: c["perfil"] for c in clusters}
    df["Perfil"] = df["Cluster"].map(mapa_perfil)

    # 5. Guardar artefactos --------------------------------------------------
    print("[5/5] Guardando artefactos")
    CARPETA_MODELO.mkdir(exist_ok=True)
    joblib.dump(modelo, CARPETA_MODELO / "kmeans_model.joblib")
    joblib.dump(scaler, CARPETA_MODELO / "scaler.joblib")

    metadata = {
        "fecha_entrenamiento": datetime.now().isoformat(timespec="seconds"),
        "features": FEATURES,
        "k": k,
        "k_mejor_silueta": k_silueta,
        "silueta_modelo": round(float(silhouette_score(X, modelo.labels_)), 4),
        "inercia_modelo": round(float(modelo.inertia_), 4),
        "inercias": {str(kk): round(v, 4) for kk, v in inercias.items()},
        "siluetas": {str(kk): round(v, 4) for kk, v in siluetas.items()},
        "clusters": clusters,
    }
    with open(CARPETA_MODELO / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    df.to_csv(RUTA_SALIDA_DATOS, index=False)

    print("\nResumen de clusters:")
    for c in clusters:
        cen = c["centroide"]
        print(f"  Cluster {c['cluster']}: {c['perfil']:<24} n={c['n_clientes']:3d} "
              f"ingreso={cen[FEATURES[0]]:6.1f}k$  gasto={cen[FEATURES[1]]:5.1f}")
    print(f"\nListo. Modelo guardado en {CARPETA_MODELO}")


if __name__ == "__main__":
    main()
