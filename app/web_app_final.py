# ============================================================
# WEB APP FINAL
# Analítica del Repositorio Institucional UC
# Universidad Central
#
# FRONTEND STREAMLIT
# ============================================================

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# RUTAS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

APP_DIR = ROOT_DIR / "app"
ASSETS_DIR = APP_DIR / "assets"

LOGO_PATH = ASSETS_DIR / "logo_uc.png"


# ============================================================
# BACKEND
# ============================================================

from backend.data_service import (
    cargar_documentos,
)

from backend.analytics_service import (
    filtrar_documentos,
    obtener_indicadores,
    obtener_documentos_por_anio,
    obtener_distribucion_facultades,
    obtener_distribucion_programas,
    obtener_distribucion_topicos,
    obtener_evolucion_topicos,
    obtener_matriz_topicos_anio,
    obtener_matriz_programa_topico,
    obtener_cambio_topicos_ultimos_anios,
    generar_hallazgo_panorama,
    generar_hallazgo_tendencias,
)

from backend.search_service import (
    buscar_tfidf,
    buscar_transformers,
    comparar_busquedas,
    resumir_comparacion_busqueda,
)

from backend.model_service import (
    obtener_comparacion_modelos,
    obtener_mejor_modelo,
    obtener_evaluacion_kmeans,
    obtener_mejor_kmeans,
    obtener_topicos_nmf,
    obtener_resumen_clusters,
    obtener_terminos_clusters,
    obtener_programas_clusters,
    obtener_resumen_modelado,
    obtener_comparacion_vectorizaciones,
)


# ============================================================
# CONFIGURACIÓN STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Repositorio UC | Analítica NLP",
    page_icon=(
        str(LOGO_PATH)
        if LOGO_PATH.exists()
        else "📊"
    ),
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# ESTILOS
# ============================================================

st.markdown(
    """
    <style>


    .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        font-family: Arial, sans-serif;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #E6E6E6;
        padding: 16px;
        border-radius: 12px;
    }

    .titulo-principal {
        font-size: 1.9rem;
        font-weight: 700;
        line-height: 1.2;
        margin-top: 0;
        margin-bottom: 0;
        color: #1B1B1B;
    }

    .subtitulo-principal {
        font-size: 0.95rem;
        color: #555555;
        margin-top: 0.3rem;
    }

    .caja-hallazgo {
        background-color: #F7F7F7;
        border-left: 5px solid #7A8400;
        padding: 16px 18px;
        border-radius: 7px;
        margin-top: 10px;
        margin-bottom: 18px;
        line-height: 1.55;
    }

    .caja-info {
        background-color: #FAFAFA;
        border: 1px solid #E4E4E4;
        padding: 16px 18px;
        border-radius: 8px;
        margin-bottom: 15px;
        line-height: 1.5;
    }


    .metodo-card {
        border: 1px solid #E2E2E2;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        background: #FFFFFF;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UTILIDADES DEL FRONTEND
# ============================================================

def columna_existente(df, candidatos):

    for columna in candidatos:
        if columna in df.columns:
            return columna

    return None


def acortar_texto(texto, limite=80):

    if pd.isna(texto):
        return ""

    texto = str(texto)

    if len(texto) <= limite:
        return texto

    return texto[: limite - 3] + "..."


def formatear_porcentaje(valor):

    try:
        return f"{float(valor) * 100:.1f}%"
    except Exception:
        return "—"


# ============================================================
# CACHE DE DATOS
# ============================================================

@st.cache_data
def obtener_corpus():

    return cargar_documentos()


# ============================================================
# ENCABEZADO
# ============================================================

def mostrar_encabezado():

    col_logo, col_texto = st.columns(
        [1, 7],
        vertical_alignment="center",
    )

    with col_logo:

        if LOGO_PATH.exists():

            st.image(
                str(LOGO_PATH),
                width=95,
            )

    with col_texto:

        st.markdown(
            """
            <div class="titulo-principal">
                Analítica NLP del Repositorio Institucional
            </div>

            <div class="subtitulo-principal">
                Universidad Central · Temáticas, tendencias y recuperación
                de información académica
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <hr style="
            margin-top:0.6rem;
            margin-bottom:1.2rem;
            border:none;
            border-top:1px solid #DDDDDD;
        ">
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FILTROS GLOBALES
# ============================================================

def construir_filtros(df):

    st.sidebar.markdown(
        "## Filtros"
    )

    anios_seleccionados = []
    facultades_seleccionadas = []
    programas_seleccionados = []
    topicos_seleccionados = []

    # --------------------------------------------------------
    # Años
    # --------------------------------------------------------

    if "anio_publicacion" in df.columns:

        anios = sorted(
            df["anio_publicacion"]
            .dropna()
            .astype(int)
            .unique()
        )

        anios_seleccionados = (
            st.sidebar.multiselect(
                "Año",
                anios,
                default=anios,
            )
        )

    # --------------------------------------------------------
    # Facultades
    # --------------------------------------------------------

    if "facultad" in df.columns:

        facultades = sorted(
            df["facultad"]
            .dropna()
            .astype(str)
            .unique()
        )

        facultades_seleccionadas = (
            st.sidebar.multiselect(
                "Facultad / Escuela",
                facultades,
            )
        )

    # --------------------------------------------------------
    # Programas
    # --------------------------------------------------------

    if "programa_academico" in df.columns:

        programas = sorted(
            df["programa_academico"]
            .dropna()
            .astype(str)
            .unique()
        )

        programas_seleccionados = (
            st.sidebar.multiselect(
                "Programa académico",
                programas,
            )
        )

    # --------------------------------------------------------
    # Temáticas
    # --------------------------------------------------------

    if "nombre_topico" in df.columns:

        topicos = sorted(
            [
                x
                for x in (
                    df["nombre_topico"]
                    .dropna()
                    .astype(str)
                    .unique()
                )
                if x != "Sin tópico asignado"
            ]
        )

        topicos_seleccionados = (
            st.sidebar.multiselect(
                "Temática NMF",
                topicos,
            )
        )

    # --------------------------------------------------------
    # Backend de filtros
    # --------------------------------------------------------

    df_filtrado = filtrar_documentos(
        df,
        anios=(
            anios_seleccionados
            if anios_seleccionados
            else None
        ),
        facultades=(
            facultades_seleccionadas
            if facultades_seleccionadas
            else None
        ),
        programas=(
            programas_seleccionados
            if programas_seleccionados
            else None
        ),
        topicos=(
            topicos_seleccionados
            if topicos_seleccionados
            else None
        ),
    )

    st.sidebar.markdown("---")

    st.sidebar.metric(
        "Documentos seleccionados",
        len(df_filtrado),
    )

    return df_filtrado


# ============================================================
# PANORAMA
# ============================================================

def mostrar_panorama(
    df_total,
    df_filtrado,
):

    st.header(
        "Panorama general"
    )

    st.caption(
        "Visión ejecutiva de la producción académica y del corpus "
        "utilizado para el análisis mediante procesamiento "
        "de lenguaje natural."
    )

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    indicadores = obtener_indicadores(
        df_total
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Documentos del repositorio",
        f"{indicadores['total_documentos']:,}",
    )

    col2.metric(
        "Corpus analizado con NLP",
        f"{indicadores['total_modelado']:,}",
    )

    col3.metric(
        "Programas académicos",
        indicadores["programas"],
    )

    col4.metric(
        "Facultades / Escuelas",
        indicadores["facultades"],
    )

    # --------------------------------------------------------
    # Lectura automática
    # --------------------------------------------------------

    hallazgo = generar_hallazgo_panorama(
        df_filtrado
    )

    st.markdown(
        f"""
        <div class="caja-hallazgo">
            <b>Lectura del panorama</b><br><br>
            {hallazgo}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Producción y facultades
    # --------------------------------------------------------

    col_a, col_b = st.columns(
        [1.15, 1]
    )

    with col_a:

        st.subheader(
            "Producción académica por año"
        )

        df_anio = obtener_documentos_por_anio(
            df_filtrado
        )

        if not df_anio.empty:

            fig = px.line(
                df_anio,
                x="anio",
                y="documentos",
                markers=True,
            )

            fig.update_layout(
                xaxis_title="Año",
                yaxis_title="Número de documentos",
                hovermode="x unified",
                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=20,
                ),
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

    with col_b:

        st.subheader(
            "Distribución por Facultad / Escuela"
        )

        facultades = (
            obtener_distribucion_facultades(
                df_filtrado
            )
        )

        if not facultades.empty:

            grafico = (
                facultades
                .sort_values(
                    "documentos",
                    ascending=True,
                )
            )

            fig = px.bar(
                grafico,
                x="documentos",
                y="facultad",
                orientation="h",
                text="documentos",
            )

            fig.update_layout(
                xaxis_title="Documentos",
                yaxis_title="",
                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=20,
                ),
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

    # --------------------------------------------------------
    # Temáticas
    # --------------------------------------------------------

    st.subheader(
        "Distribución de temáticas de investigación"
    )

    topicos = obtener_distribucion_topicos(
        df_filtrado
    )

    if not topicos.empty:

        grafico = (
            topicos
            .sort_values(
                "documentos",
                ascending=True,
            )
        )

        fig = px.bar(
            grafico,
            x="documentos",
            y="tematica",
            orientation="h",
            text="documentos",
            custom_data=[
                "porcentaje"
            ],
        )

        fig.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Documentos: %{x}<br>"
                "Participación: %{customdata[0]:.1f}%"
                "<extra></extra>"
            )
        )

        fig.update_layout(
            xaxis_title="Número de documentos",
            yaxis_title="",
            height=510,
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

    st.divider()

    # --------------------------------------------------------
    # Matriz programa × tema
    # --------------------------------------------------------

    st.subheader(
        "Relación entre programas y temáticas"
    )

    st.caption(
        "La matriz permite observar qué temáticas se concentran "
        "en determinados programas y cuáles atraviesan varias "
        "áreas académicas."
    )

    matriz = obtener_matriz_programa_topico(
        df_filtrado,
        top_n_programas=20,
    )

    if not matriz.empty:

        fig = px.imshow(
            matriz,
            aspect="auto",
            labels={
                "x": "Temática",
                "y": "Programa",
                "color": "Documentos",
            },
        )

        fig.update_layout(
            height=760,
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        st.markdown(
            """
            <div class="caja-info">
            <b>Lectura crítica:</b>
            una concentración alta en una única temática puede reflejar
            especialización disciplinar. La presencia de una temática
            en varios programas puede ser indicio de transversalidad o
            convergencia entre áreas del conocimiento.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# TENDENCIAS
# ============================================================

def mostrar_tendencias(df):

    st.header(
        "Tendencias de investigación"
    )

    st.caption(
        "Análisis de la evolución temporal de la producción "
        "y de las temáticas identificadas en el corpus."
    )
    hallazgo_tendencias = generar_hallazgo_tendencias(df)

    st.markdown(
        f"""
        <div class="caja-hallazgo">
            <b>Lectura de tendencias</b><br><br>
            {hallazgo_tendencias}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df.empty:

        st.warning(
            "No hay documentos para los filtros seleccionados."
        )

        return

    modo = st.radio(
        "Analizar como",
        [
            "Cantidad de documentos",
            "Participación porcentual anual",
        ],
        horizontal=True,
    )

    modo_backend = (
        "porcentaje"
        if modo
        == "Participación porcentual anual"
        else "cantidad"
    )

    # --------------------------------------------------------
    # Producción total
    # --------------------------------------------------------

    st.subheader(
        "Evolución de la producción académica"
    )

    produccion = obtener_documentos_por_anio(
        df
    )

    if not produccion.empty:

        fig = px.line(
            produccion,
            x="anio",
            y="documentos",
            markers=True,
        )

        fig.update_layout(
            xaxis_title="Año",
            yaxis_title="Documentos",
            hovermode="x unified",
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

    st.info(
        "Los extremos temporales deben interpretarse con cautela. "
        "Los años con cobertura parcial no son directamente comparables "
        "con años completos."
    )

    # --------------------------------------------------------
    # Evolución temática
    # --------------------------------------------------------

    st.subheader(
        "Evolución de las temáticas"
    )

    evolucion = obtener_evolucion_topicos(
        df,
        modo=modo_backend,
    )

    if not evolucion.empty:

        eje_y = (
            "Participación anual (%)"
            if modo_backend == "porcentaje"
            else "Número de documentos"
        )

        fig = px.line(
            evolucion,
            x="anio",
            y="valor",
            color="nombre_topico",
            markers=True,
            labels={
                "nombre_topico":
                    "Temática",
            },
        )

        fig.update_layout(
            xaxis_title="Año",
            yaxis_title=eje_y,
            legend_title="Temática",
            hovermode="x unified",
            height=610,
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    st.subheader(
        "Mapa de calor de temáticas por año"
    )

    matriz = obtener_matriz_topicos_anio(
        df
    )

    if not matriz.empty:

        fig = px.imshow(
            matriz,
            text_auto=True,
            aspect="auto",
            labels={
                "x": "Temática",
                "y": "Año",
                "color": "Documentos",
            },
        )

        fig.update_layout(
            height=610,
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

    # --------------------------------------------------------
    # Cambio entre últimos años
    # --------------------------------------------------------

    st.subheader(
        "Cambio entre los dos últimos años disponibles"
    )

    cambios = (
        obtener_cambio_topicos_ultimos_anios(
            df
        )
    )

    if not cambios.empty:

        anio_1 = int(
            cambios[
                "anio_anterior"
            ].iloc[0]
        )

        anio_2 = int(
            cambios[
                "anio_actual"
            ].iloc[0]
        )

        grafico = (
            cambios
            .sort_values(
                "variacion_absoluta"
            )
        )

        fig = px.bar(
            grafico,
            x="variacion_absoluta",
            y="tematica",
            orientation="h",
            text="variacion_absoluta",
        )

        fig.update_layout(
            xaxis_title=(
                f"Variación de documentos "
                f"({anio_1} → {anio_2})"
            ),
            yaxis_title="",
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        st.warning(
            "Una variación negativa en el año más reciente no debe "
            "interpretarse automáticamente como disminución de interés. "
            "Puede estar asociada a cobertura documental incompleta."
        )


# ============================================================
# TEMÁTICAS
# ============================================================

def mostrar_tematicas(df):

    st.header(
        "Temáticas de investigación"
    )

    st.caption(
        "Exploración de los tópicos identificados mediante "
        "Non-negative Matrix Factorization (NMF)."
    )

    distribucion = obtener_distribucion_topicos(
        df
    )

    if distribucion.empty:

        st.warning(
            "No hay información temática para la selección actual."
        )

        return

    # --------------------------------------------------------
    # Tarjetas
    # --------------------------------------------------------

    columnas = st.columns(3)

    for i, fila in (
        distribucion
        .reset_index(drop=True)
        .iterrows()
    ):

        with columnas[i % 3]:

            with st.container(border=True):

                st.markdown(
                    f"### {fila['tematica']}"
                )

                st.metric(
                    "Documentos",
                    int(fila["documentos"]),
                )

                st.caption(
                    f"{fila['porcentaje']:.1f}% "
                    "del corpus seleccionado"
                )
    # --------------------------------------------------------
    # Selector
    # --------------------------------------------------------

    st.subheader(
        "Explorar una temática"
    )

    tematica = st.selectbox(
        "Selecciona una temática",
        distribucion[
            "tematica"
        ].tolist(),
    )

    seleccion = df[
        df[
            "nombre_topico"
        ]
        == tematica
    ].copy()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Documentos",
        len(seleccion),
    )

    col2.metric(
        "Programas representados",
        (
            seleccion[
                "programa_academico"
            ].nunique()
            if "programa_academico"
            in seleccion.columns
            else 0
        ),
    )

    col3.metric(
        "Facultades representadas",
        (
            seleccion[
                "facultad"
            ].nunique()
            if "facultad"
            in seleccion.columns
            else 0
        ),
    )

    col_a, col_b = st.columns(2)

    # --------------------------------------------------------
    # Evolución
    # --------------------------------------------------------

    with col_a:

        st.markdown(
            "#### Evolución temporal"
        )

        temporal = obtener_documentos_por_anio(
            seleccion
        )

        if not temporal.empty:

            fig = px.line(
                temporal,
                x="anio",
                y="documentos",
                markers=True,
            )

            fig.update_layout(
                xaxis_title="Año",
                yaxis_title="Documentos",
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

    # --------------------------------------------------------
    # Programas
    # --------------------------------------------------------

    with col_b:

        st.markdown(
            "#### Programas con mayor presencia"
        )

        programas = obtener_distribucion_programas(
            seleccion,
            top_n=10,
        )

        if not programas.empty:

            fig = px.bar(
                programas.sort_values(
                    "documentos"
                ),
                x="documentos",
                y="programa",
                orientation="h",
                text="documentos",
            )

            fig.update_layout(
                xaxis_title="Documentos",
                yaxis_title="",
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

    # --------------------------------------------------------
    # Documentos
    # --------------------------------------------------------

    st.subheader(
        "Documentos asociados"
    )

    columnas_documentos = [
        columna
        for columna in [
            "titulo",
            "autores_texto",
            "anio_publicacion",
            "facultad",
            "programa_academico",
            "subjects_texto",
            "url_consulta",
        ]
        if columna in seleccion.columns
    ]

    tabla = seleccion[
        columnas_documentos
    ].copy()

    if "anio_publicacion" in tabla.columns:

        tabla = tabla.sort_values(
            "anio_publicacion",
            ascending=False,
        )

    st.dataframe(
        tabla,
        width="stretch",
        hide_index=True,
        height=520,
        column_config={
            "titulo":
                "Título",
            "autores_texto":
                "Autores",
            "anio_publicacion":
                st.column_config.NumberColumn(
                    "Año",
                    format="%d",
                ),
            "facultad":
                "Facultad / Escuela",
            "programa_academico":
                "Programa",
            "subjects_texto":
                "Descriptores",
            "url_consulta":
                st.column_config.LinkColumn(
                    "Repositorio"
                ),
        },
    )


# ============================================================
# RESULTADOS DE BÚSQUEDA
# ============================================================

def mostrar_resultados_busqueda(
    resultados,
    titulo,
):

    st.markdown(
        f"### {titulo}"
    )

    if resultados.empty:

        st.warning(
            "No se encontraron resultados."
        )

        return

    titulo_col = columna_existente(
        resultados,
        [
            "titulo",
            "titulo_documento",
        ],
    )

    programa_col = columna_existente(
        resultados,
        [
            "programa_academico",
            "programa",
        ],
    )

    anio_col = columna_existente(
        resultados,
        [
            "anio_publicacion",
            "anio",
        ],
    )

    columnas = [
        columna
        for columna in [
            titulo_col,
            programa_col,
            anio_col,
            "similitud",
        ]
        if columna is not None
    ]

    tabla = resultados[
        columnas
    ].copy()

    tabla[
        "similitud"
    ] = (
        tabla[
            "similitud"
        ]
        .round(4)
    )

    st.dataframe(
        tabla,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------
    # Gráfico
    # --------------------------------------------------------

    if titulo_col is not None:

        grafico = (
            resultados
            .head(10)
            .copy()
        )

        grafico[
            "titulo_corto"
        ] = (
            grafico[
                titulo_col
            ]
            .apply(
                lambda x:
                    acortar_texto(
                        x,
                        68,
                    )
            )
        )

        grafico = (
            grafico
            .sort_values(
                "similitud"
            )
        )

        fig = px.bar(
            grafico,
            x="similitud",
            y="titulo_corto",
            orientation="h",
            text_auto=".3f",
        )

        fig.update_layout(
            xaxis_title="Similitud coseno",
            yaxis_title="",
            height=500,
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )


# ============================================================
# BÚSQUEDA Y EXPLORACIÓN
# ============================================================

def mostrar_busqueda(df):

    st.header(
        "Explorar y buscar"
    )

    st.caption(
        "Recuperación de información mediante TF-IDF y "
        "representaciones semánticas con Transformers."
    )

    # --------------------------------------------------------
    # Método
    # --------------------------------------------------------

    metodo = st.radio(
        "Método de búsqueda",
        [
            "TF-IDF",
            "Transformers",
            "Comparar ambos métodos",
        ],
        horizontal=True,
    )

    if metodo == "TF-IDF":

        st.markdown(
            """
            <div class="caja-info">
            <b>TF-IDF — método principal</b><br><br>

            Utiliza importancia léxica de términos y bigramas.
            Fue el método con mejor desempeño frente a los
            descriptores temáticos utilizados como referencia.
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif metodo == "Transformers":

        st.markdown(
            """
            <div class="caja-info">
            <b>Transformers — búsqueda semántica complementaria</b><br><br>

            Utiliza embeddings multilingües para identificar
            proximidad semántica incluso cuando los documentos
            no comparten exactamente las mismas palabras.
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="caja-info">
            <b>Comparación de modelos</b><br><br>

            Ejecuta la misma consulta en los dos espacios de
            representación. Las puntuaciones de similitud no deben
            compararse directamente entre modelos porque pertenecen
            a espacios vectoriales distintos.
            </div>
            """,
            unsafe_allow_html=True,
        )

    consulta = st.text_input(
        "Consulta temática",
        placeholder=(
            "Ejemplo: inteligencia artificial aplicada "
            "al análisis de datos"
        ),
    )

    top_n = st.slider(
        "Número de resultados",
        min_value=5,
        max_value=20,
        value=10,
    )

    # --------------------------------------------------------
    # Ejecutar consulta
    # --------------------------------------------------------

    if consulta.strip():

        if metodo == "TF-IDF":

            with st.spinner(
                "Buscando documentos..."
            ):

                resultados = buscar_tfidf(
                    consulta,
                    top_n,
                )

            mostrar_resultados_busqueda(
                resultados,
                "Resultados TF-IDF",
            )

        elif metodo == "Transformers":

            with st.spinner(
                "Calculando similitud semántica..."
            ):

                resultados = buscar_transformers(
                    consulta,
                    top_n,
                )

            mostrar_resultados_busqueda(
                resultados,
                "Resultados semánticos",
            )

        else:

            with st.spinner(
                "Comparando TF-IDF y Transformers..."
            ):

                resultados = comparar_busquedas(
                    consulta,
                    top_n,
                )

                resumen = (
                    resumir_comparacion_busqueda(
                        consulta,
                        top_n,
                    )
                )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Top analizado",
                top_n,
            )

            col2.metric(
                "Documentos coincidentes",
                resumen[
                    "coincidencias"
                ],
            )

            col3.metric(
                "Coincidencia entre métodos",
                f"{resumen['porcentaje_coincidencia']:.1f}%",
            )

            st.markdown(
                """
                <div class="caja-hallazgo">
                <b>Interpretación:</b><br><br>

                La coincidencia indica cuántos documentos aparecen
                simultáneamente en los primeros resultados de TF-IDF
                y Transformers. Una coincidencia baja no implica que
                uno de los métodos sea incorrecto: puede reflejar que
                capturan dimensiones distintas de la relevancia.
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_tfidf, col_trans = st.columns(2)

            with col_tfidf:

                mostrar_resultados_busqueda(
                    resultados["tfidf"],
                    "TF-IDF",
                )

            with col_trans:

                mostrar_resultados_busqueda(
                    resultados[
                        "transformers"
                    ],
                    "Transformers",
                )

    st.divider()

    # --------------------------------------------------------
    # Explorador documental
    # --------------------------------------------------------

    st.subheader(
        "Explorador documental"
    )

    filtro = st.text_input(
        "Filtrar por título, autor, resumen o descriptor",
        key="explorador",
    )

    explorador = df.copy()

    if filtro.strip():

        termino = (
            filtro
            .strip()
            .lower()
        )

        columnas_busqueda = [
            columna
            for columna in [
                "titulo",
                "autores_texto",
                "resumen",
                "subjects_texto",
            ]
            if columna in explorador.columns
        ]

        mascara = pd.Series(
            False,
            index=explorador.index,
        )

        for columna in columnas_busqueda:

            mascara = (
                mascara
                |
                explorador[
                    columna
                ]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains(
                    termino,
                    regex=False,
                )
            )

        explorador = explorador[
            mascara
        ]

    st.caption(
        f"Documentos encontrados: {len(explorador):,}"
    )

    columnas = [
        columna
        for columna in [
            "titulo",
            "autores_texto",
            "anio_publicacion",
            "facultad",
            "programa_academico",
            "nombre_topico",
            "subjects_texto",
            "url_consulta",
        ]
        if columna in explorador.columns
    ]

    st.dataframe(
        explorador[columnas],
        width="stretch",
        hide_index=True,
        height=560,
        column_config={
            "titulo":
                "Título",
            "autores_texto":
                "Autores",
            "anio_publicacion":
                st.column_config.NumberColumn(
                    "Año",
                    format="%d",
                ),
            "facultad":
                "Facultad / Escuela",
            "programa_academico":
                "Programa",
            "nombre_topico":
                "Temática",
            "subjects_texto":
                "Descriptores",
            "url_consulta":
                st.column_config.LinkColumn(
                    "Repositorio"
                ),
        },
    )


# ============================================================
# MODELOS
# ============================================================

def mostrar_modelos():

    st.header(
        "Modelos y evaluación"
    )

    st.caption(
        "Resultados de la comparación de representaciones, "
        "recuperación documental y modelado temático."
    )

    resumen = obtener_resumen_modelado()

    # --------------------------------------------------------
    # KPIs metodología
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Modelo principal",
        "TF-IDF",
    )

    col2.metric(
        "HitRate@10",
        formatear_porcentaje(
            resumen[
                "hitrate_10_principal"
            ]
        ),
    )

    col3.metric(
        "Tópicos NMF",
        resumen[
            "numero_topicos_nmf"
        ],
    )

    col4.metric(
        "KMeans seleccionado",
        f"k = {resumen['kmeans_seleccionado']}",
    )

    # --------------------------------------------------------
    # Comparación modelos
    # --------------------------------------------------------

    st.subheader(
        "Comparación de recuperación"
    )

    comparacion = obtener_comparacion_modelos()

    nombres_metricas = {
        "precision_5":
            "Precision@5",
        "mrr":
            "MRR",
        "ndcg_10":
            "nDCG@10",
        "hitrate_10":
            "HitRate@10",
    }

    metrica = st.selectbox(
        "Métrica",
        list(
            nombres_metricas.keys()
        ),
        format_func=lambda x:
            nombres_metricas[x],
    )

    grafico = (
        comparacion[
            [
                "modelo",
                metrica,
            ]
        ]
        .sort_values(
            metrica
        )
    )

    fig = px.bar(
        grafico,
        x=metrica,
        y="modelo",
        orientation="h",
        text_auto=".3f",
    )

    fig.update_layout(
        xaxis_title=(
            nombres_metricas[
                metrica
            ]
        ),
        yaxis_title="",
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    mejor = obtener_mejor_modelo(
        metrica
    )

    st.markdown(
        f"""
        <div class="caja-hallazgo">
        <b>Resultado:</b><br><br>

        Para <b>{nombres_metricas[metrica]}</b>, el mejor
        resultado corresponde a <b>{mejor['modelo']}</b>,
        con un valor de <b>{mejor['valor']:.4f}</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Tabla completa
    # --------------------------------------------------------

    with st.expander(
        "Ver tabla completa de métricas"
    ):

        tabla = comparacion.rename(
            columns={
                "modelo":
                    "Modelo",
                "precision_5":
                    "Precision@5",
                "mrr":
                    "MRR",
                "ndcg_10":
                    "nDCG@10",
                "hitrate_10":
                    "HitRate@10",
            }
        )

        st.dataframe(
            tabla,
            width="stretch",
            hide_index=True,
        )

    # --------------------------------------------------------
    # Significado métricas
    # --------------------------------------------------------

    with st.expander(
        "¿Cómo interpretar las métricas?"
    ):

        st.markdown(
            """
            **Precision@5**  
            Proporción de documentos relevantes entre los cinco
            primeros resultados.

            **MRR — Mean Reciprocal Rank**  
            Evalúa la posición en la que aparece el primer
            documento relevante.

            **nDCG@10**  
            Considera tanto la relevancia como la posición de los
            documentos dentro de los diez primeros resultados.

            **HitRate@10**  
            Proporción de consultas en las que aparece al menos un
            documento relevante dentro del Top 10.

            **Importante:** HitRate@10 no es *accuracy*.
            """
        )

    st.divider()

    # --------------------------------------------------------
    # Comparación de vectorizaciones
    # --------------------------------------------------------

    st.subheader(
        "Comparación de vectorizaciones"
    )

    vectorizaciones = (
        obtener_comparacion_vectorizaciones()
    )

    if not vectorizaciones.empty:

        st.dataframe(
            vectorizaciones,
            width="stretch",
            hide_index=True,
        )

    # --------------------------------------------------------
    # KMeans
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Clustering KMeans"
    )

    mejor_k = obtener_mejor_kmeans()

    if mejor_k:

        col_k1, col_k2 = st.columns(2)

        col_k1.metric(
            "k seleccionado",
            mejor_k["k"],
        )

        col_k2.metric(
            "Silhouette coseno",
            f"{mejor_k['silhouette']:.6f}",
        )

    evaluacion_k = obtener_evaluacion_kmeans()

    if not evaluacion_k.empty:

        columna_sil = columna_existente(
            evaluacion_k,
            [
                "silhouette_coseno",
                "silhouette",
                "silhouette_score",
            ],
        )

        if (
            "k" in evaluacion_k.columns
            and columna_sil is not None
        ):

            fig = px.line(
                evaluacion_k,
                x="k",
                y=columna_sil,
                markers=True,
            )

            fig.update_layout(
                xaxis_title="Número de clústeres (k)",
                yaxis_title="Coeficiente de silueta",
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

    st.markdown(
        f"""
        <div class="caja-info">
        <b>Interpretación metodológica:</b><br><br>
        {resumen['interpretacion_kmeans']}
        El valor de silueta es bajo, por lo que k = 9 debe
        interpretarse como la mejor configuración dentro del rango
        explorado y no como evidencia de nueve grupos naturales
        completamente separados.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # NMF
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Modelado temático NMF"
    )

    st.write(
        resumen[
            "interpretacion_nmf"
        ]
    )

    topicos = obtener_topicos_nmf()

    if not topicos.empty:

        with st.expander(
            "Ver resultados de los tópicos NMF"
        ):

            st.dataframe(
                topicos,
                width="stretch",
                hide_index=True,
            )

    st.markdown(
        """
        <div class="caja-info">
        Los nombres temáticos utilizados en el dashboard son
        etiquetas interpretativas construidas a partir de los
        términos de mayor peso de cada componente NMF. NMF no
        asigna automáticamente estos nombres.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# METODOLOGÍA
# ============================================================

def mostrar_metodologia():

    st.header(
        "Metodología"
    )

    st.caption(
        "Arquitectura del sistema y proceso utilizado para "
        "analizar los productos del Repositorio Institucional."
    )

    # --------------------------------------------------------
    # Arquitectura
    # --------------------------------------------------------

    st.subheader(
        "Arquitectura de la solución"
    )

    st.code(
        """
Repositorio Institucional
        │
        ▼
Excel + MongoDB
        │
        ▼
Pipeline NLP
src/
        │
        ├── extracción descriptiva
        ├── auditoría
        ├── construcción del corpus
        ├── limpieza
        ├── análisis exploratorio
        ├── TF-IDF
        ├── evaluación
        ├── KMeans / NMF
        └── Transformers
        │
        ▼
data/processed + models/
        │
        ▼
Backend
backend/
        ├── data_service.py
        ├── analytics_service.py
        ├── search_service.py
        └── model_service.py
        │
        ▼
Frontend
app/web_app_final.py
        """,
        language="text",
    )

    st.markdown(
        """
        <div class="caja-info">
        La separación entre backend y frontend permite que la
        interfaz no ejecute directamente toda la lógica analítica.
        El backend centraliza acceso a datos, análisis, recuperación
        documental y evaluación de modelos, mientras Streamlit se
        concentra en interacción y visualización.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    st.subheader(
        "Pipeline analítico"
    )

    etapas = [
        (
            "1. Fuente de información",
            "Base de productos del Repositorio Institucional de "
            "la Universidad Central.",
        ),
        (
            "2. Extracción descriptiva",
            "Recuperación de título, resumen, subjects y "
            "metadatos académicos.",
        ),
        (
            "3. Construcción del corpus",
            "Integración de título, resumen y descriptores "
            "temáticos.",
        ),
        (
            "4. Preprocesamiento",
            "Normalización, tokenización, lematización y "
            "filtrado lingüístico.",
        ),
        (
            "5. Análisis exploratorio",
            "Evaluación de cobertura, longitud, distribución "
            "temporal y comportamiento del vocabulario.",
        ),
        (
            "6. Representación vectorial",
            "Comparación de Count Vectorizer y diferentes "
            "configuraciones TF-IDF.",
        ),
        (
            "7. Recuperación",
            "Búsqueda documental mediante similitud coseno.",
        ),
        (
            "8. Evaluación",
            "Precision, Recall, MAP, MRR, nDCG y HitRate.",
        ),
        (
            "9. Modelado temático",
            "KMeans como agrupamiento exploratorio y NMF "
            "como representación temática interpretable.",
        ),
        (
            "10. Embeddings semánticos",
            "Sentence Transformers como método semántico "
            "complementario.",
        ),
        (
            "11. Visualización",
            "Dashboard interactivo para exploración, análisis "
            "y recuperación de información.",
        ),
    ]

    for titulo, descripcion in etapas:

        st.markdown(
            f"""
            <div class="caja-info">
                <b>{titulo}</b><br><br>
                {descripcion}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # Corpus
    # --------------------------------------------------------

    st.subheader(
        "Construcción del corpus"
    )

    st.write(
        """
        El corpus utilizado para el modelado se construyó
        principalmente a partir de **título, resumen y descriptores
        temáticos (subjects)**.

        Esta decisión permitió trabajar con información documental
        más homogénea y reducir el ruido producido por portadas,
        bibliografías, tablas y otras estructuras presentes en los
        PDF completos.
        """
    )

    # --------------------------------------------------------
    # Modelo principal
    # --------------------------------------------------------

    st.subheader(
        "Selección del modelo de recuperación"
    )

    st.write(
        """
        La comparación experimental mostró que **TF-IDF con
        unigramas y bigramas** presentó el mejor desempeño global
        frente a los descriptores temáticos utilizados como
        referencia.

        Los embeddings de **Sentence Transformers** se conservaron
        como una representación complementaria, útil para explorar
        relaciones semánticas que no dependen exclusivamente de la
        coincidencia de términos.
        """
    )


# ============================================================
# MAIN
# ============================================================

def main():

    mostrar_encabezado()

    df = obtener_corpus()

    if df.empty:

        st.error(
            "No fue posible cargar el corpus consolidado."
        )

        st.stop()

    # ========================================================
    # NAVEGACIÓN
    # ========================================================

    st.sidebar.markdown(
        "# Navegación"
    )

    pagina = st.sidebar.radio(
        "Sección",
        [
            "Panorama",
            "Tendencias",
            "Temáticas",
            "Explorar y buscar",
            "Modelos",
            "Metodología",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")

    # ========================================================
    # FILTROS
    # ========================================================

    paginas_filtrables = [
        "Panorama",
        "Tendencias",
        "Temáticas",
        "Explorar y buscar",
    ]

    if pagina in paginas_filtrables:

        df_filtrado = construir_filtros(
            df
        )

    else:

        df_filtrado = df

    # ========================================================
    # ROUTING
    # ========================================================

    if pagina == "Panorama":

        mostrar_panorama(
            df,
            df_filtrado,
        )

    elif pagina == "Tendencias":

        mostrar_tendencias(
            df_filtrado
        )

    elif pagina == "Temáticas":

        mostrar_tematicas(
            df_filtrado
        )

    elif pagina == "Explorar y buscar":

        mostrar_busqueda(
            df_filtrado
        )

    elif pagina == "Modelos":

        mostrar_modelos()

    elif pagina == "Metodología":

        mostrar_metodologia()


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()
