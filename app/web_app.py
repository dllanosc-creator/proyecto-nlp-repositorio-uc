import sys
from pathlib import Path
from collections import Counter
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
import pandas as pd
import streamlit as st
import plotly.express as px


# Permite importar db_mongo.py desde la carpeta src/
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

from db_mongo import obtener_coleccion

LOGO_PATH = Path("app/assets/logo_uc.png")

if LOGO_PATH.exists():
    logo_icon = Image.open(LOGO_PATH)

st.set_page_config(
    page_title="Repositorio UC - Analítica NLP",
    page_icon=logo_icon,
    layout="wide"
)
def mostrar_encabezado_institucional():
    col_logo, col_texto = st.columns([1, 5])

    with col_logo:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=150)

    with col_texto:
        st.markdown(
            """
            <h1 style="margin-bottom: 0; color: #1B1B1B;">
                Analítica NLP del Repositorio Institucional UC
            </h1>
            <p style="font-size: 18px; margin-top: 6px; color: #444444;">
                Universidad Central · Procesamiento de Lenguaje Natural aplicado a trabajos de grado
            </p>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <hr style="border: 2px solid #7A8400; margin-top: 10px; margin-bottom: 25px;">
        """,
        unsafe_allow_html=True,
    )

@st.cache_data(ttl=300)
def cargar_documentos():
    """
    Carga los documentos principales desde MongoDB.
    """
    coleccion = obtener_coleccion()

    documentos = list(
        coleccion.find(
            {},
            {
                "_id": 0,
                "id_documento": 1,
                "programa_academico": 1,
                "anio_publicacion": 1,
                "autores": 1,
                "titulo": 1,
                "directores": 1,
                "url_consulta": 1,
                "pdf_url": 1,
                "archivo_pdf": 1,
                "estado_procesamiento": 1,
                "cantidad_paginas": 1,
                "cantidad_caracteres": 1,
                "cantidad_palabras_aprox": 1,
                "cantidad_tokens_limpios": 1,
                "texto_original": 1,
                "texto_limpio": 1,
                "tokens": 1,
                "lemmas": 1,
                "version_limpieza": 1,
            },
        )
    )

    return pd.DataFrame(documentos)


@st.cache_data(ttl=300)
def cargar_similitud_consulta_documento():
    """
    Carga desde MongoDB los resultados precomputados de similitud
    entre consultas/palabras clave y documentos.
    """
    coleccion = obtener_coleccion()
    db = coleccion.database

    registros = list(
        db["similitud_consulta_documento"].find(
            {},
            {"_id": 0},
        )
    )

    return pd.DataFrame(registros)


@st.cache_data(ttl=300)
def cargar_similitud_documento_documento():
    """
    Carga desde MongoDB los resultados precomputados de similitud
    entre documentos.
    """
    coleccion = obtener_coleccion()
    db = coleccion.database

    registros = list(
        db["similitud_documento_documento"].find(
            {},
            {"_id": 0},
        )
    )

    return pd.DataFrame(registros)

@st.cache_data(ttl=300)
def cargar_similitud_transformers():
    """
    Carga desde MongoDB los resultados de similitud calculados con embeddings Transformers.
    """
    coleccion = obtener_coleccion()
    db = coleccion.database

    registros = list(
        db["similitud_transformers_consulta_documento"].find(
            {},
            {"_id": 0},
        )
    )

    return pd.DataFrame(registros)

@st.cache_data(ttl=300)
def cargar_topicos_resumen():
    """
    Carga el resumen de tópicos NMF desde MongoDB.
    """
    coleccion = obtener_coleccion()
    db = coleccion.database

    registros = list(
        db["topicos_resumen"].find(
            {},
            {"_id": 0},
        )
    )

    return pd.DataFrame(registros)
def registrar_consulta_usuario(consulta, cantidad_resultados):
    """
    Guarda en MongoDB una consulta realizada por el usuario en la app web.
    """
    if not consulta:
        return

    coleccion = obtener_coleccion()
    db = coleccion.database

    db["consultas_usuarios"].insert_one(
        {
            "consulta": str(consulta).strip().lower(),
            "cantidad_resultados": int(cantidad_resultados),
            "fecha_consulta": datetime.now().isoformat(),
        }
    )


@st.cache_data(ttl=60)
def cargar_tendencias_busqueda():
    """
    Carga las consultas realizadas por usuarios en la app web.
    """
    coleccion = obtener_coleccion()
    db = coleccion.database

    registros = list(
        db["consultas_usuarios"].find(
            {},
            {"_id": 0},
        )
    )

    return pd.DataFrame(registros)


def preparar_documentos_limpios(df):
    """
    Filtra documentos que ya tienen texto limpio.
    """
    if df.empty:
        return df

    if "estado_procesamiento" not in df.columns:
        return pd.DataFrame()

    df_limpios = df[df["estado_procesamiento"] == "texto_limpio"].copy()

    if df_limpios.empty:
        return df_limpios

    if "texto_limpio" not in df_limpios.columns:
        return pd.DataFrame()

    df_limpios["texto_limpio"] = df_limpios["texto_limpio"].fillna("")
    df_limpios["anio_publicacion"] = (
        df_limpios["anio_publicacion"]
        .fillna("Sin año")
        .astype(str)
    )

    return df_limpios


def construir_frecuencias_globales(df_limpios):
    """
    Construye ranking global de términos frecuentes desde tokens.
    """
    contador = Counter()

    if df_limpios.empty or "tokens" not in df_limpios.columns:
        return pd.DataFrame(columns=["termino", "frecuencia"])

    for tokens in df_limpios["tokens"].dropna():
        if isinstance(tokens, list):
            contador.update(tokens)

    df_freq = pd.DataFrame(
        contador.most_common(),
        columns=["termino", "frecuencia"],
    )

    return df_freq


def construir_tendencia_por_termino(df_limpios, termino):
    """
    Calcula frecuencia de un término por año.
    """
    registros = []
    termino = termino.lower().strip()

    if df_limpios.empty or "tokens" not in df_limpios.columns:
        return pd.DataFrame(columns=["anio_publicacion", "frecuencia"])

    for _, fila in df_limpios.iterrows():
        tokens = fila.get("tokens", [])

        if not isinstance(tokens, list):
            tokens = []

        frecuencia = tokens.count(termino)

        registros.append(
            {
                "anio_publicacion": str(fila.get("anio_publicacion", "Sin año")),
                "id_documento": fila.get("id_documento", ""),
                "titulo": fila.get("titulo", ""),
                "frecuencia": frecuencia,
            }
        )

    df_tendencia = pd.DataFrame(registros)

    if df_tendencia.empty:
        return df_tendencia

    df_tendencia = (
        df_tendencia
        .groupby("anio_publicacion", as_index=False)["frecuencia"]
        .sum()
        .sort_values("anio_publicacion")
    )

    return df_tendencia

def construir_documentos_por_anio(df_limpios):
    """
    Calcula la cantidad de documentos procesados por año de publicación.
    """
    if df_limpios.empty or "anio_publicacion" not in df_limpios.columns:
        return pd.DataFrame(columns=["anio_publicacion", "cantidad_documentos"])

    df_anio = (
        df_limpios["anio_publicacion"]
        .fillna("Sin año")
        .astype(str)
        .value_counts()
        .reset_index()
    )

    df_anio.columns = ["anio_publicacion", "cantidad_documentos"]

    df_anio = df_anio.sort_values("anio_publicacion")

    return df_anio
def construir_top_terminos_por_anio_trazabilidad(df_limpios, top_n=5):
    """
    Calcula los términos más frecuentes por año y construye una tabla
    para visualizar su trazabilidad temporal.
    """
    if df_limpios.empty or "tokens" not in df_limpios.columns:
        return pd.DataFrame(columns=["anio_publicacion", "termino", "frecuencia"])

    registros = []

    for anio, grupo in df_limpios.groupby("anio_publicacion"):
        contador = Counter()

        for tokens in grupo["tokens"].dropna():
            if isinstance(tokens, list):
                contador.update(tokens)

        for termino, frecuencia in contador.most_common(top_n):
            registros.append(
                {
                    "anio_publicacion": str(anio),
                    "termino": termino,
                    "frecuencia": frecuencia,
                }
            )

    df_top = pd.DataFrame(registros)

    if df_top.empty:
        return df_top

    # Para trazabilidad: si un término fue top en algún año,
    # se calcula su frecuencia en todos los años.
    terminos_top = sorted(df_top["termino"].unique())

    registros_trazabilidad = []

    for anio, grupo in df_limpios.groupby("anio_publicacion"):
        contador = Counter()

        for tokens in grupo["tokens"].dropna():
            if isinstance(tokens, list):
                contador.update(tokens)

        for termino in terminos_top:
            registros_trazabilidad.append(
                {
                    "anio_publicacion": str(anio),
                    "termino": termino,
                    "frecuencia": contador.get(termino, 0),
                }
            )

    df_trazabilidad = pd.DataFrame(registros_trazabilidad)

    return df_trazabilidad.sort_values(["termino", "anio_publicacion"])
def construir_estado_procesamiento(df):
    """
    Resume estados del pipeline.
    """
    if df.empty or "estado_procesamiento" not in df.columns:
        return pd.DataFrame(columns=["estado_procesamiento", "cantidad"])

    df_estado = (
        df["estado_procesamiento"]
        .fillna("sin_estado")
        .value_counts()
        .reset_index()
    )

    df_estado.columns = ["estado_procesamiento", "cantidad"]

    return df_estado

def generar_nube_palabras(df_limpios, top_n=200):
    """
    Genera una nube de palabras a partir de los tokens limpios del corpus.
    """
    contador = Counter()

    if df_limpios.empty or "tokens" not in df_limpios.columns:
        return None

    for tokens in df_limpios["tokens"].dropna():
        if isinstance(tokens, list):
            contador.update(tokens)

    if not contador:
        return None

    frecuencias = dict(contador.most_common(top_n))

    nube = WordCloud(
        width=1600,
        height=800,
        background_color="white",
        collocations=False,
    ).generate_from_frequencies(frecuencias)

    return nube

@st.cache_data(ttl=300)
def cargar_topicos_documento():
    """
    Carga la asignación de tópico dominante por documento desde MongoDB.
    """
    coleccion = obtener_coleccion()
    db = coleccion.database

    registros = list(
        db["topicos_documento"].find(
            {},
            {"_id": 0},
        )
    )

    return pd.DataFrame(registros)

def construir_heatmap_topicos_por_anio(df_topicos_documento):
    """
    Construye una matriz año × tópico con la cantidad de documentos.
    """
    if df_topicos_documento.empty:
        return pd.DataFrame()

    df_aux = df_topicos_documento.copy()

    if "anio_publicacion" not in df_aux.columns or "topico_dominante" not in df_aux.columns:
        return pd.DataFrame()

    df_aux["anio_publicacion"] = df_aux["anio_publicacion"].fillna("Sin año").astype(str)
    df_aux["topico_dominante"] = df_aux["topico_dominante"].astype(str)

    df_heatmap = (
        df_aux.groupby(["anio_publicacion", "topico_dominante"])
        .size()
        .reset_index(name="cantidad_documentos")
    )

    matriz = df_heatmap.pivot(
        index="anio_publicacion",
        columns="topico_dominante",
        values="cantidad_documentos"
    ).fillna(0)

    # Ordenar años si es posible
    try:
        matriz.index = pd.Index(matriz.index.astype(int), name="anio_publicacion")
        matriz = matriz.sort_index()
        matriz.index = matriz.index.astype(str)
    except:
        matriz = matriz.sort_index()

    # Renombrar columnas para que se vea mejor
    matriz.columns = [f"Tópico {col}" for col in matriz.columns]

    return matriz

def main():
    mostrar_encabezado_institucional()

    st.markdown(
        """
        Prototipo web para explorar trabajos de grado procesados mediante técnicas de 
        procesamiento de lenguaje natural. Permite visualizar similitud entre consultas 
        temáticas y documentos, recomendar documentos similares, analizar tendencias 
        y revisar términos frecuentes del corpus.
        """
    )

    df = cargar_documentos()
    df_limpios = preparar_documentos_limpios(df)
    df_sim_consulta = cargar_similitud_consulta_documento()
    df_sim_documentos = cargar_similitud_documento_documento()
    df_sim_transformers = cargar_similitud_transformers()
    df_topicos_resumen = cargar_topicos_resumen()
    df_topicos_documento = cargar_topicos_documento()

    if df.empty:
        st.error("No se encontraron documentos en MongoDB.")
        return

    if df.empty:
        st.error("No se encontraron documentos en MongoDB.")
        return

    total_registros = len(df)
    total_limpios = len(df_limpios)

    if not df_limpios.empty and "cantidad_paginas" in df_limpios.columns:
        total_paginas = int(df_limpios["cantidad_paginas"].fillna(0).sum())
    else:
        total_paginas = 0

    if not df_limpios.empty and "cantidad_tokens_limpios" in df_limpios.columns:
        total_tokens = int(df_limpios["cantidad_tokens_limpios"].fillna(0).sum())
    else:
        total_tokens = 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Registros en MongoDB", total_registros)
    col2.metric("Documentos limpios", total_limpios)
    col3.metric("Páginas procesadas", total_paginas)
    col4.metric("Tokens limpios", total_tokens)

    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "🔎 Similitud consulta-documento",
            "🤖 Transformers",
            "🧩 Documentos similares",
            "📈 Tendencias por término",
            "🔍 Tendencias de búsqueda",
            "🏷️ Términos frecuentes",
            "📄 Documentos",
            "⚙️ Estado del pipeline",
        ]
    )

    with tab1:
        st.subheader("Similitud entre consulta temática y documentos")

        st.markdown(
            """
            Esta sección utiliza los resultados calculados previamente en el pipeline mediante
            **TF-IDF** y **similitud del coseno**. La aplicación no recalcula los vectores en vivo;
            consulta los resultados almacenados en MongoDB.
            """
        )

        if df_sim_consulta.empty:
            st.warning(
                "No hay resultados de similitud consulta-documento. "
                "Ejecuta primero: python src/05_vectorizar_y_calcular_similitud.py"
            )
        else:
            consultas_disponibles = sorted(
                df_sim_consulta["consulta"].dropna().unique()
            )

            consulta_seleccionada = st.selectbox(
                "Selecciona una consulta o palabra clave",
                consultas_disponibles,
            )

            top_n = st.slider(
                "Número de documentos a mostrar",
                min_value=3,
                max_value=20,
                value=10,
                key="top_consulta_documento",
            )

            resultados = (
                df_sim_consulta[
                    df_sim_consulta["consulta"] == consulta_seleccionada
                ]
                .copy()
                .sort_values("similitud_coseno", ascending=False)
                .head(top_n)
            )

            registrar_consulta_usuario(
                consulta=consulta_seleccionada,
                cantidad_resultados=len(resultados),
            )

            resultados["similitud_coseno"] = resultados["similitud_coseno"].round(4)

            columnas_mostrar = [
                "consulta",
                "titulo",
                "autores",
                "anio_publicacion",
                "programa_academico",
                "cantidad_tokens_limpios",
                "similitud_coseno",
                "url_consulta",
            ]

            columnas_existentes = [
                col for col in columnas_mostrar if col in resultados.columns
            ]

            st.dataframe(
                resultados[columnas_existentes],
                width="stretch",
                hide_index=True,
            )

            grafico = resultados.sort_values("similitud_coseno", ascending=True)

            fig = px.bar(
                grafico,
                x="similitud_coseno",
                y="titulo",
                orientation="h",
                title=f"Documentos más similares a: {consulta_seleccionada}",
                labels={
                    "similitud_coseno": "Similitud coseno",
                    "titulo": "Documento",
                },
            )

            st.plotly_chart(fig, width="stretch")
    
        with tab2:
            st.subheader("Similitud semántica con embeddings Transformers")

            st.markdown(
                """
                Esta sección usa embeddings contextuales generados con un modelo Transformer multilingüe.
                A diferencia de TF-IDF, esta representación busca capturar cercanía semántica entre
                consultas y documentos, incluso cuando no comparten exactamente las mismas palabras.
                """
            )

            if df_sim_transformers.empty:
                st.warning(
                    "No hay resultados de similitud con Transformers. "
                    "Ejecuta primero: python src/10_embeddings_transformers.py"
                )
            else:
                consultas_transformers = sorted(
                    df_sim_transformers["consulta"].dropna().unique()
                )

                consulta_transformer = st.selectbox(
                    "Selecciona una consulta",
                    consultas_transformers,
                    key="consulta_transformer",
                )

                top_n_transformer = st.slider(
                    "Número de documentos a mostrar",
                    min_value=3,
                    max_value=20,
                    value=10,
                    key="top_transformer",
                )

                resultados_transformer = (
                    df_sim_transformers[
                        df_sim_transformers["consulta"] == consulta_transformer
                    ]
                    .copy()
                    .sort_values("similitud_coseno", ascending=False)
                    .head(top_n_transformer)
                )

                resultados_transformer["similitud_coseno"] = (
                    resultados_transformer["similitud_coseno"].round(4)
                )

                columnas_transformer = [
                    "consulta",
                    "titulo",
                    "autores",
                    "anio_publicacion",
                    "programa_academico",
                    "cantidad_tokens_limpios",
                    "similitud_coseno",
                    "url_consulta",
                ]

                columnas_existentes = [
                    col for col in columnas_transformer
                    if col in resultados_transformer.columns
                ]

                st.dataframe(
                    resultados_transformer[columnas_existentes],
                    width="stretch",
                    hide_index=True,
                )

                grafico_transformer = resultados_transformer.sort_values(
                    "similitud_coseno",
                    ascending=True,
                )

                fig = px.bar(
                    grafico_transformer,
                    x="similitud_coseno",
                    y="titulo",
                    orientation="h",
                    title=f"Documentos más similares con Transformers: {consulta_transformer}",
                    labels={
                        "similitud_coseno": "Similitud coseno",
                        "titulo": "Documento",
                    },
                )

                st.plotly_chart(fig, width="stretch")

    with tab3:
        st.subheader("Recomendación de documentos similares")

        st.markdown(
            """
            Esta sección compara documentos entre sí usando los vectores TF-IDF generados
            a partir del texto limpio y la similitud del coseno.
            """
        )

        if df_sim_documentos.empty:
            st.warning(
                "No hay resultados de similitud documento-documento. "
                "Ejecuta primero: python src/05_vectorizar_y_calcular_similitud.py"
            )
        else:
            documentos_origen = (
                df_sim_documentos[["id_documento_origen", "titulo_origen"]]
                .drop_duplicates()
                .sort_values("titulo_origen")
            )

            opciones = {
                f"{row['titulo_origen']} ({row['id_documento_origen']})": row[
                    "id_documento_origen"
                ]
                for _, row in documentos_origen.iterrows()
            }

            seleccion = st.selectbox(
                "Selecciona un documento",
                list(opciones.keys()),
            )

            id_origen = opciones[seleccion]

            top_n_doc = st.slider(
                "Número de documentos similares a mostrar",
                min_value=3,
                max_value=20,
                value=10,
                key="top_documentos_similares",
            )

            similares = (
                df_sim_documentos[
                    df_sim_documentos["id_documento_origen"] == id_origen
                ]
                .copy()
                .sort_values("similitud_coseno", ascending=False)
                .head(top_n_doc)
            )

            similares["similitud_coseno"] = similares["similitud_coseno"].round(4)

            columnas_similares = [
                "titulo_origen",
                "titulo_destino",
                "anio_destino",
                "similitud_coseno",
            ]

            columnas_existentes = [
                col for col in columnas_similares if col in similares.columns
            ]

            st.dataframe(
                similares[columnas_existentes],
                width="stretch",
                hide_index=True,
            )

            grafico_similares = similares.sort_values(
                "similitud_coseno",
                ascending=True,
            )

            fig = px.bar(
                grafico_similares,
                x="similitud_coseno",
                y="titulo_destino",
                orientation="h",
                title="Documentos más similares",
                labels={
                    "similitud_coseno": "Similitud coseno",
                    "titulo_destino": "Documento recomendado",
                },
            )

            st.plotly_chart(fig, width="stretch")

    with tab4:
        st.subheader("Tendencias del corpus")
        st.markdown("### Trazabilidad del Top 5 de términos por año")

        df_top_trazabilidad = construir_top_terminos_por_anio_trazabilidad(
            df_limpios,
            top_n=5,
        )

        if df_top_trazabilidad.empty:
            st.warning("No hay datos suficientes para calcular el top de términos por año.")
        else:
            fig_top_trazabilidad = px.line(
                df_top_trazabilidad,
                x="anio_publicacion",
                y="frecuencia",
                color="termino",
                markers=True,
                title="Trazabilidad del Top 5 de términos por año",
                labels={
                    "anio_publicacion": "Año de publicación",
                    "frecuencia": "Frecuencia",
                    "termino": "Término",
                },
            )

            st.plotly_chart(fig_top_trazabilidad, width="stretch")

            with st.expander("Ver datos de trazabilidad del Top 5"):
                st.dataframe(
                    df_top_trazabilidad,
                    width="stretch",
                    hide_index=True,
                )

        st.divider()

        st.markdown("### Cantidad de documentos procesados por año")

        df_documentos_anio = construir_documentos_por_anio(df_limpios)

        if df_documentos_anio.empty:
            st.warning("No hay datos suficientes para graficar documentos por año.")
        else:
            fig_docs_anio = px.bar(
                df_documentos_anio,
                x="anio_publicacion",
                y="cantidad_documentos",
                title="Cantidad de documentos procesados por año",
                labels={
                    "anio_publicacion": "Año de publicación",
                    "cantidad_documentos": "Cantidad de documentos",
                },
                text="cantidad_documentos",
            )

            fig_docs_anio.update_traces(textposition="outside")
            st.plotly_chart(fig_docs_anio, width="stretch")

        st.divider()

        st.markdown("### Tendencia temática por término")

        termino_tendencia = st.text_input(
            "Término para analizar tendencia",
            value="aprendizaje",
        )

        if termino_tendencia.strip():
            df_tendencia = construir_tendencia_por_termino(
                df_limpios,
                termino_tendencia,
            )

            if df_tendencia.empty:
                st.warning("No hay datos suficientes para generar la tendencia.")
            else:
                fig = px.line(
                    df_tendencia,
                    x="anio_publicacion",
                    y="frecuencia",
                    markers=True,
                    title=f"Tendencia del término: {termino_tendencia}",
                    labels={
                        "anio_publicacion": "Año de publicación",
                        "frecuencia": "Frecuencia",
                    },
                )

                st.plotly_chart(fig, width="stretch")
                st.dataframe(df_tendencia, width="stretch", hide_index=True)
    with tab5:
        st.subheader("Tendencias de búsqueda de usuarios")

        st.markdown(
            """
            Esta sección muestra las consultas realizadas por los usuarios dentro del prototipo web.
            A diferencia de la tendencia temática, aquí se analiza qué términos buscan las personas
            al interactuar con la aplicación.
            """
        )

        df_busquedas = cargar_tendencias_busqueda()

        if df_busquedas.empty:
            st.warning("Todavía no hay búsquedas registradas.")
        else:
            conteo_busquedas = (
                df_busquedas["consulta"]
                .value_counts()
                .reset_index()
            )

            conteo_busquedas.columns = ["consulta", "cantidad_busquedas"]

            fig = px.bar(
                conteo_busquedas.sort_values("cantidad_busquedas", ascending=True),
                x="cantidad_busquedas",
                y="consulta",
                orientation="h",
                title="Consultas más frecuentes en la aplicación",
                labels={
                    "cantidad_busquedas": "Cantidad de búsquedas",
                    "consulta": "Consulta",
                },
            )

            st.plotly_chart(fig, width="stretch")
            st.dataframe(conteo_busquedas, width="stretch", hide_index=True)

            st.subheader("Histórico de consultas")

            st.dataframe(
                df_busquedas.sort_values("fecha_consulta", ascending=False),
                width="stretch",
                hide_index=True,
            )

    with tab6:
        st.subheader("Términos frecuentes del corpus procesado")
        
        st.markdown("### Temáticas más abordadas")

        st.markdown("### Mapa de calor de temáticas por año")

        matriz_heatmap = construir_heatmap_topicos_por_anio(df_topicos_documento)

        if matriz_heatmap.empty:
            st.warning(
                "No hay datos suficientes para construir el mapa de calor. "
                "Ejecuta primero: python src/09_modelado_tematico_clustering.py"
            )
        else:
            fig_heatmap = px.imshow(
                matriz_heatmap,
                text_auto=True,
                aspect="auto",
                labels={
                    "x": "Temática / tópico",
                    "y": "Año de publicación",
                    "color": "Cantidad de documentos",
                },
                title="Distribución de temáticas por año",
            )

            st.plotly_chart(fig_heatmap, width="stretch")

            with st.expander("Ver tabla base del mapa de calor"):
                st.dataframe(
                    matriz_heatmap.reset_index(),
                    width="stretch",
                    hide_index=True,
                )

        st.divider()

        if df_topicos_resumen.empty:
            st.warning(
                "No hay tópicos disponibles. Ejecuta primero: "
                "python src/09_modelado_tematico_clustering.py"
            )
        else:
            tabla_tematicas = df_topicos_resumen.copy()

            columnas_tematicas = [
                "topico",
                "cantidad_documentos",
                "terminos_principales",
            ]

            columnas_existentes = [
                col for col in columnas_tematicas
                if col in tabla_tematicas.columns
            ]

            tabla_tematicas = tabla_tematicas[columnas_existentes]

            tabla_tematicas = tabla_tematicas.rename(
                columns={
                    "topico": "Tópico",
                    "cantidad_documentos": "Cantidad de documentos",
                    "terminos_principales": "Términos representativos",
                }
            )

            st.dataframe(
                tabla_tematicas,
                width="stretch",
                hide_index=True,
            )

            fig_tematicas = px.bar(
                tabla_tematicas,
                x="Tópico",
                y="Cantidad de documentos",
                title="Cantidad de documentos por temática",
                text="Cantidad de documentos",
                labels={
                    "Tópico": "Temática / tópico",
                    "Cantidad de documentos": "Cantidad de documentos",
                },
            )

            fig_tematicas.update_traces(textposition="outside")
            st.plotly_chart(fig_tematicas, width="stretch")

        st.divider()
        df_freq = construir_frecuencias_globales(df_limpios)

        if df_freq.empty:
            st.warning("No hay tokens disponibles.")
        else:
            top_frecuencias = st.slider(
                "Top términos",
                min_value=10,
                max_value=100,
                value=30,
            )

            df_top = df_freq.head(top_frecuencias)

            fig = px.bar(
                df_top.sort_values("frecuencia", ascending=True),
                x="frecuencia",
                y="termino",
                orientation="h",
                title=f"Top {top_frecuencias} términos frecuentes",
                labels={
                    "frecuencia": "Frecuencia",
                    "termino": "Término",
                },
            )

            st.plotly_chart(fig, width="stretch")
            st.dataframe(df_top, width="stretch", hide_index=True)
            st.subheader("☁️ Nube de palabras del corpus")

            nube = generar_nube_palabras(df_limpios, top_n=200)

            if nube is None:
                st.warning("No fue posible generar la nube de palabras.")
            else:
                fig, ax = plt.subplots(figsize=(14, 7))
                ax.imshow(nube, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

    with tab7:
        st.subheader("Documentos del corpus")

        columnas_mostrar = [
            "id_documento",
            "titulo",
            "autores",
            "anio_publicacion",
            "programa_academico",
            "estado_procesamiento",
            "cantidad_paginas",
            "cantidad_palabras_aprox",
            "cantidad_tokens_limpios",
            "url_consulta",
        ]

        columnas_existentes = [col for col in columnas_mostrar if col in df.columns]

        st.dataframe(
            df[columnas_existentes],
            width="stretch",
            hide_index=True,
        )

    with tab8:
        st.subheader("Estado del procesamiento")

        df_estado = construir_estado_procesamiento(df)

        if df_estado.empty:
            st.warning("No hay estados disponibles.")
        else:
            fig = px.pie(
                df_estado,
                names="estado_procesamiento",
                values="cantidad",
                title="Distribución de estados del pipeline",
            )

            st.plotly_chart(fig, width="stretch")
            st.dataframe(df_estado, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()