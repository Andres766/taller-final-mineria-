"""
Capa de FRONTEND (Parte 7 - Despliegue con Streamlit)
-----------------------------------------------------
- Visualiza el método del codo y el coeficiente de silueta
- Muestra los clusters generados y sus perfiles de negocio
- Permite ingresar un nuevo cliente y lo clasifica automáticamente

Ejecutar desde la carpeta raíz del proyecto:
    streamlit run frontend/app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]
CARPETA_MODELO = RAIZ / "modelo"
RUTA_DATOS = RAIZ / "datos" / "Mall_Customers.csv"

# ---------------------------------------------------------------------------
# Estilo
# ---------------------------------------------------------------------------
PRIMARIO = "#24527A"
TINTA = "#1F2A36"
COLORES = {
    "Clientes de alto valor": "#1B7F5C",
    "Clientes con potencial": "#D08C1A",
    "Clientes promedio": "#6B7C8F",
    "Clientes impulsivos": "#7A4FA3",
    "Clientes de bajo valor": "#B5404E",
}

st.set_page_config(
    page_title="Segmentación de clientes K-Means",
    page_icon="🛍️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1200px;}
      .tarjeta {
          background: #FFFFFF;
          border: 1px solid #E1E6EB;
          border-left-width: 8px;
          border-radius: 6px;
          padding: 1rem 1.2rem;
          margin-bottom: 1rem;
      }
      .tarjeta h3 {margin: 0 0 .25rem 0; font-size: 1.35rem;}
      .tarjeta p {margin: .2rem 0; line-height: 1.5;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Carga de artefactos
# ---------------------------------------------------------------------------
def artefactos_disponibles() -> bool:
    archivos = ["kmeans_model.joblib", "scaler.joblib", "metadata.json"]
    return all((CARPETA_MODELO / a).exists() for a in archivos)


@st.cache_resource
def cargar_modelo():
    modelo = joblib.load(CARPETA_MODELO / "kmeans_model.joblib")
    scaler = joblib.load(CARPETA_MODELO / "scaler.joblib")
    meta = json.loads((CARPETA_MODELO / "metadata.json").read_text(encoding="utf-8"))
    return modelo, scaler, meta


@st.cache_data
def cargar_datos() -> pd.DataFrame:
    return pd.read_csv(RUTA_DATOS)


if not artefactos_disponibles():
    st.error(
        "No encontré el modelo entrenado en la carpeta `modelo/`. "
        "Ejecuta primero el entrenamiento desde la raíz del proyecto:\n\n"
        "`python entrenamiento/entrenar.py`"
    )
    st.stop()

modelo, scaler, meta = cargar_modelo()
df = cargar_datos()
F_ING, F_GAS = meta["features"]
K = meta["k"]
INFO = {c["cluster"]: c for c in meta["clusters"]}

# Asignación de cluster a cada cliente con el modelo cargado
df["Cluster"] = modelo.predict(scaler.transform(df[[F_ING, F_GAS]]))
df["Perfil"] = df["Cluster"].map(lambda c: INFO[c]["perfil"])


def color_de(perfil: str) -> str:
    return COLORES.get(perfil, PRIMARIO)


def layout_base(fig, alto=440):
    fig.update_layout(
        template="plotly_white",
        height=alto,
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(color=TINTA, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def grafica_clusters(nuevo=None):
    fig = go.Figure()
    for c in meta["clusters"]:
        sub = df[df["Cluster"] == c["cluster"]]
        fig.add_trace(go.Scatter(
            x=sub[F_ING], y=sub[F_GAS], mode="markers",
            name=f"{c['cluster']}: {c['perfil']}",
            marker=dict(size=9, color=color_de(c["perfil"]), opacity=0.78,
                        line=dict(width=0.6, color="white")),
            customdata=sub[["CustomerID", "Age", "Gender"]],
            hovertemplate=(
                "Cliente %{customdata[0]}<br>Ingreso: %{x} k$<br>"
                "Gasto: %{y}<br>Edad: %{customdata[1]}<extra>%{fullData.name}</extra>"
            ),
        ))
    fig.add_trace(go.Scatter(
        x=[c["centroide"][F_ING] for c in meta["clusters"]],
        y=[c["centroide"][F_GAS] for c in meta["clusters"]],
        mode="markers", name="Centroides",
        marker=dict(symbol="x-thin", size=18, color=TINTA, line=dict(width=3)),
        hovertemplate="Centroide<br>Ingreso: %{x:.1f} k$<br>Gasto: %{y:.1f}<extra></extra>",
    ))
    if nuevo is not None:
        fig.add_trace(go.Scatter(
            x=[nuevo["ingreso"]], y=[nuevo["gasto"]], mode="markers",
            name="Nuevo cliente",
            marker=dict(symbol="star", size=26, color=color_de(nuevo["perfil"]),
                        line=dict(width=2, color=TINTA)),
            hovertemplate="Nuevo cliente<br>Ingreso: %{x} k$<br>Gasto: %{y}<extra></extra>",
        ))
    fig.update_xaxes(title="Ingreso anual (k$)")
    fig.update_yaxes(title="Puntaje de gasto (1-100)")
    return layout_base(fig, alto=520)


# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.title("🛍️ Segmentación de clientes del centro comercial")
st.caption(
    f"Modelo K-Means con K = {K}, entrenado con ingreso anual y puntaje de gasto "
    f"de {len(df)} clientes (Mall Customers Dataset)."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Clientes analizados", len(df))
m2.metric("Clusters (K)", K)
m3.metric("Coeficiente de silueta", f"{meta['silueta_modelo']:.3f}")
m4.metric("Inercia (WCSS)", f"{meta['inercia_modelo']:.1f}")

with st.sidebar:
    st.header("Sobre el modelo")
    st.write(f"**Variables:** {F_ING} y {F_GAS}")
    st.write("**Escalado:** StandardScaler")
    st.write(f"**K con mejor silueta:** {meta['k_mejor_silueta']}")
    st.write(f"**Entrenado:** {meta['fecha_entrenamiento'].replace('T', ' ')}")
    st.divider()
    st.caption("Para reentrenar: `python entrenamiento/entrenar.py --k 5`")

tab_codo, tab_sil, tab_clusters, tab_nuevo = st.tabs(
    ["📉 Método del codo", "📊 Coeficiente de silueta", "🎯 Clusters", "🧍 Clasificar cliente"]
)

# ---------------------------------------------------------------------------
# Método del codo
# ---------------------------------------------------------------------------
with tab_codo:
    ks = sorted(int(x) for x in meta["inercias"])
    inercias = [meta["inercias"][str(x)] for x in ks]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ks, y=inercias, mode="lines+markers", name="Inercia",
        line=dict(color=PRIMARIO, width=3), marker=dict(size=9),
        hovertemplate="K = %{x}<br>Inercia = %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[K], y=[meta["inercias"][str(K)]], mode="markers", name=f"K elegido = {K}",
        marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(width=3, color="#D08C1A")),
        hoverinfo="skip",
    ))
    fig.update_xaxes(title="Número de clusters (K)", dtick=1)
    fig.update_yaxes(title="Inercia (suma de distancias² al centroide)")
    st.plotly_chart(layout_base(fig))

    reduccion = pd.DataFrame({
        "K": ks,
        "Inercia": np.round(inercias, 2),
        "Reducción vs. K-1 (%)": [None] + [
            round(100 * (inercias[i - 1] - inercias[i]) / inercias[i - 1], 1)
            for i in range(1, len(ks))
        ],
    })
    c1, c2 = st.columns([1, 1.4])
    c1.dataframe(reduccion, hide_index=True)
    c2.markdown(
        f"""
        **Cómo leerlo.** La inercia mide qué tan cerca están los clientes de su
        centroide: entre más baja, más compactos los grupos. Siempre baja al
        aumentar K, por eso se busca el *codo*, el punto desde el cual agregar
        clusters ya casi no mejora.

        Aquí el codo está en **K = {K}**: de {K - 1} a {K} clusters la inercia
        cae bastante, y de {K} en adelante las mejoras son pequeñas.
        """
    )

# ---------------------------------------------------------------------------
# Silueta
# ---------------------------------------------------------------------------
with tab_sil:
    ks_s = sorted(int(x) for x in meta["siluetas"])
    sil = [meta["siluetas"][str(x)] for x in ks_s]
    fig = go.Figure(go.Bar(
        x=ks_s, y=sil,
        marker_color=["#D08C1A" if x == K else "#A9B8C7" for x in ks_s],
        text=[f"{v:.3f}" for v in sil], textposition="outside",
        hovertemplate="K = %{x}<br>Silueta = %{y:.4f}<extra></extra>",
    ))
    fig.update_xaxes(title="Número de clusters (K)", dtick=1)
    fig.update_yaxes(title="Coeficiente de silueta promedio", range=[0, max(sil) * 1.18])
    st.plotly_chart(layout_base(fig))
    st.markdown(
        f"""
        El coeficiente de silueta va de **-1 a 1**: cerca de 1 significa grupos
        bien separados, cerca de 0 grupos que se solapan y negativo clientes mal
        asignados. El máximo se da en **K = {meta['k_mejor_silueta']}**
        ({max(sil):.3f}), el mismo valor que sugiere el método del codo.
        """
    )

# ---------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------
with tab_clusters:
    st.plotly_chart(grafica_clusters())

    resumen = pd.DataFrame([{
        "Cluster": c["cluster"],
        "Perfil": c["perfil"],
        "Clientes": c["n_clientes"],
        "% del total": c["porcentaje"],
        "Ingreso medio (k$)": c["centroide"][F_ING],
        "Gasto medio": c["centroide"][F_GAS],
        "Edad media": c["edad_promedio"],
    } for c in meta["clusters"]])
    st.subheader("Perfiles de los clusters")
    st.dataframe(resumen, hide_index=True)

    st.subheader("Estrategias de marketing sugeridas")
    for c in sorted(meta["clusters"], key=lambda c: -c["centroide"][F_ING] - c["centroide"][F_GAS]):
        st.markdown(
            f"""<div class="tarjeta" style="border-left-color:{color_de(c['perfil'])}">
                <h3>{c['perfil']} <span style="font-weight:400;font-size:1rem">(cluster {c['cluster']}, {c['n_clientes']} clientes)</span></h3>
                <p>{c['estrategia']}</p></div>""",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Clasificar un nuevo cliente
# ---------------------------------------------------------------------------
with tab_nuevo:
    if "historial" not in st.session_state:
        st.session_state.historial = []
    if "ultimo" not in st.session_state:
        st.session_state.ultimo = None

    col_form, col_res = st.columns([1, 1.6])

    with col_form:
        st.subheader("Datos del cliente")
        with st.form("form_cliente"):
            nombre = st.text_input("Nombre o ID (opcional)", placeholder="Ej: Cliente 201")
            ingreso = st.number_input(
                "Ingreso anual (miles de USD)", min_value=1.0, max_value=300.0,
                value=60.0, step=1.0,
            )
            gasto = st.slider("Puntaje de gasto (1-100)", min_value=1, max_value=100, value=50)
            enviado = st.form_submit_button("Clasificar cliente", type="primary")

        if enviado:
            X_nuevo = pd.DataFrame([[ingreso, gasto]], columns=[F_ING, F_GAS])
            z = scaler.transform(X_nuevo)
            cluster = int(modelo.predict(z)[0])
            distancias = np.linalg.norm(modelo.cluster_centers_ - z, axis=1)
            perfil = INFO[cluster]["perfil"]
            st.session_state.ultimo = {
                "nombre": nombre.strip() or f"Cliente {len(st.session_state.historial) + 1}",
                "ingreso": ingreso, "gasto": gasto,
                "cluster": cluster, "perfil": perfil,
                "distancias": distancias.tolist(),
            }
            st.session_state.historial.append({
                "Cliente": st.session_state.ultimo["nombre"],
                "Ingreso (k$)": ingreso, "Gasto": gasto,
                "Cluster": cluster, "Perfil": perfil,
            })

        min_i, max_i = df[F_ING].min(), df[F_ING].max()
        st.caption(
            f"El modelo se entrenó con ingresos entre {min_i} y {max_i} k$. "
            "Fuera de ese rango la clasificación es menos confiable."
        )

    with col_res:
        u = st.session_state.ultimo
        if u is None:
            st.info("Ingresa el ingreso y el puntaje de gasto, y pulsa **Clasificar cliente**.")
            st.plotly_chart(grafica_clusters(), key="graf_vacia")
        else:
            info = INFO[u["cluster"]]
            st.markdown(
                f"""<div class="tarjeta" style="border-left-color:{color_de(u['perfil'])}">
                    <p style="font-size:.95rem">{u['nombre']}: ingreso {u['ingreso']:.0f} k$, gasto {u['gasto']}</p>
                    <h3>Cluster {u['cluster']}: {u['perfil']}</h3>
                    <p><b>Estrategia recomendada:</b> {info['estrategia']}</p></div>""",
                unsafe_allow_html=True,
            )
            if not (min_i <= u["ingreso"] <= max_i):
                st.warning("El ingreso está fuera del rango de entrenamiento.")
            st.plotly_chart(grafica_clusters(nuevo=u), key="graf_nuevo")

            dist = pd.DataFrame({
                "Cluster": [c["cluster"] for c in meta["clusters"]],
                "Perfil": [c["perfil"] for c in meta["clusters"]],
                "Distancia al centroide (escalada)": np.round(u["distancias"], 3),
            }).sort_values("Distancia al centroide (escalada)")
            with st.expander("Ver distancia a cada centroide"):
                st.dataframe(dist, hide_index=True)

    if st.session_state.historial:
        st.subheader("Clientes clasificados en esta sesión")
        st.dataframe(pd.DataFrame(st.session_state.historial), hide_index=True)
        if st.button("Borrar historial"):
            st.session_state.historial = []
            st.session_state.ultimo = None
            st.rerun()
