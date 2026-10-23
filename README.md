# Taller final Minería de Datos: Segmentación de clientes con K-Means

Segmentación del *Mall Customers Dataset* con K-Means (K = 5, silueta 0.555) usando
ingreso anual y puntaje de gasto, con despliegue interactivo en Streamlit.

## Estructura (por capas)

```
taller_kmeans/
├── datos/
│   ├── Mall_Customers.csv          # dataset original
│   └── clientes_segmentados.csv    # dataset + Cluster + Perfil (se genera)
├── cuaderno/
│   └── taller_kmeans.ipynb         # análisis completo + respuestas (Partes 1 a 7)
├── entrenamiento/
│   └── entrenar.py                 # entrena y exporta el modelo
├── modelo/
│   ├── kmeans_model.joblib         # modelo K-Means
│   ├── scaler.joblib               # StandardScaler ajustado
│   └── metadata.json               # K, inercias, siluetas, centroides y perfiles
├── frontend/
│   └── app.py                      # app Streamlit
├── .streamlit/config.toml          # tema visual de la app
├── requirements.txt
└── README.md
```

Flujo: `datos/` → `entrenamiento/entrenar.py` → `modelo/` → `frontend/app.py`

## Cómo ejecutarlo

Todos los comandos se corren **desde la carpeta raíz** `taller_kmeans/`.

### 1. Crear el entorno e instalar dependencias

Windows (PowerShell o CMD):
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Mac / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> Si PowerShell no deja activar el entorno, ejecuta una vez:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 2. Entrenar el modelo

```bash
python entrenamiento/entrenar.py
```
Elige K automáticamente (mejor silueta = 5). Para forzar otro valor: `python entrenamiento/entrenar.py --k 4`

### 3. Abrir el cuaderno

```bash
jupyter notebook cuaderno/taller_kmeans.ipynb
```
También se puede abrir directamente en VS Code (extensión Jupyter). Ya viene ejecutado con
todas las gráficas; para regenerarlo: *Kernel → Restart & Run All*.

### 4. Lanzar la app

```bash
streamlit run frontend/app.py
```
Se abre en `http://localhost:8501`. Si en la primera ejecución pide un correo, solo presiona Enter.

La app tiene 4 pestañas: método del codo, coeficiente de silueta, clusters con perfiles y
estrategias, y clasificación de un cliente nuevo.

## Resultados principales

| Perfil | Ingreso medio | Gasto medio | Clientes |
|---|---|---|---|
| Clientes de alto valor | ≈ 87 k$ | ≈ 82 | 39 |
| Clientes con potencial | ≈ 88 k$ | ≈ 17 | 35 |
| Clientes promedio | ≈ 55 k$ | ≈ 50 | 81 |
| Clientes impulsivos | ≈ 26 k$ | ≈ 79 | 22 |
| Clientes de bajo valor | ≈ 26 k$ | ≈ 21 | 23 |

Mayor oportunidad de crecimiento: **clientes con potencial** (ingreso alto, gasto bajo).
