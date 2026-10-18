import pandas as pd

from backend.data_service import (
    cargar_comparacion_vectorizaciones,
    cargar_evaluacion_tfidf,
    cargar_evaluacion_transformers,
    cargar_evaluacion_kmeans,
    cargar_topicos_nmf,
    cargar_resumen_clusters,
    cargar_terminos_clusters,
    cargar_programas_clusters,
)


# ============================================================
# MÉTRICAS PRINCIPALES DE RECUPERACIÓN
# ============================================================

def obtener_comparacion_modelos():
    """
    Devuelve una tabla homogénea con las métricas principales
    de los modelos evaluados.

    Los valores corresponden a la evaluación definitiva del
    corpus metadata_v3_final.
    """

    return pd.DataFrame(
        {
            "modelo": [
                "Count unigramas",
                "TF-IDF unigramas",
                "TF-IDF uni + bigramas",
                "Transformers",
            ],
            "precision_5": [
                0.2108,
                0.2316,
                0.2501,
                0.1152,
            ],
            "mrr": [
                0.4236,
                0.4390,
                0.4757,
                0.2596,
            ],
            "ndcg_10": [
                0.3097,
                0.3364,
                0.3796,
                0.1657,
            ],
            "hitrate_10": [
                0.6957,
                0.7378,
                0.7852,
                0.4796,
            ],
        }
    )


# ============================================================
# MEJOR MODELO POR MÉTRICA
# ============================================================

def obtener_mejor_modelo(metrica):
    """
    Retorna el modelo con mayor desempeño para una métrica.

    Métricas disponibles:
    - precision_5
    - mrr
    - ndcg_10
    - hitrate_10
    """

    comparacion = obtener_comparacion_modelos()

    if metrica not in comparacion.columns:
        raise ValueError(
            f"Métrica no válida: {metrica}"
        )

    fila = (
        comparacion
        .sort_values(
            metrica,
            ascending=False,
        )
        .iloc[0]
    )

    return {
        "modelo": fila["modelo"],
        "metrica": metrica,
        "valor": float(fila[metrica]),
    }


# ============================================================
# EVALUACIÓN TF-IDF
# ============================================================

def obtener_evaluacion_tfidf():
    """
    Carga la evaluación detallada de TF-IDF.
    """

    return cargar_evaluacion_tfidf()


# ============================================================
# EVALUACIÓN TRANSFORMERS
# ============================================================

def obtener_evaluacion_transformers():
    """
    Carga la evaluación del modelo Transformer.
    """

    return cargar_evaluacion_transformers()


# ============================================================
# COMPARACIÓN DE VECTORIZACIONES
# ============================================================

def obtener_comparacion_vectorizaciones():
    """
    Carga los resultados de Count Vectorizer y TF-IDF.
    """

    return cargar_comparacion_vectorizaciones()


# ============================================================
# KMEANS
# ============================================================

def obtener_evaluacion_kmeans():
    """
    Carga los resultados de evaluación para los diferentes
    valores de k probados.
    """

    return cargar_evaluacion_kmeans()


def obtener_mejor_kmeans():
    """
    Identifica el valor de k con mayor coeficiente de silueta.
    """

    df = cargar_evaluacion_kmeans()

    if df.empty:
        return None

    columna_k = None

    for candidato in [
        "k",
        "n_clusters",
    ]:
        if candidato in df.columns:
            columna_k = candidato
            break

    columna_silueta = None

    for candidato in [
        "silhouette_coseno",
        "silhouette"
        "silhouette_score",
    ]:
        if candidato in df.columns:
            columna_silueta = candidato
            break

    if (
        columna_k is None
        or columna_silueta is None
    ):
        return None

    fila = (
        df
        .sort_values(
            columna_silueta,
            ascending=False,
        )
        .iloc[0]
    )

    return {
        "k": int(fila[columna_k]),
        "silhouette": float(
            fila[columna_silueta]
        ),
    }


# ============================================================
# CLÚSTERES KMEANS
# ============================================================

def obtener_resumen_clusters():
    """
    Información resumida de los clústeres finales.
    """

    return cargar_resumen_clusters()


def obtener_terminos_clusters():
    """
    Términos representativos de cada clúster.
    """

    return cargar_terminos_clusters()


def obtener_programas_clusters():
    """
    Distribución de programas académicos por clúster.
    """

    return cargar_programas_clusters()


# ============================================================
# NMF
# ============================================================

def obtener_topicos_nmf():
    """
    Carga los términos y pesos de los tópicos NMF.
    """

    return cargar_topicos_nmf()


# ============================================================
# INTERPRETACIÓN GENERAL DEL MODELO
# ============================================================

def obtener_resumen_modelado():
    """
    Retorna las decisiones metodológicas principales
    utilizadas por la aplicación.
    """

    mejor_tfidf = obtener_mejor_modelo(
        "hitrate_10"
    )

    mejor_k = obtener_mejor_kmeans()

    return {
        "modelo_recuperacion_principal":
            "TF-IDF uni + bigramas",

        "modelo_semantico_complementario":
            "Sentence Transformers",

        "hitrate_10_principal":
            mejor_tfidf["valor"],

        "numero_topicos_nmf":
            9,

        "kmeans_seleccionado":
            (
                mejor_k["k"]
                if mejor_k
                else 9
            ),

        "silhouette_kmeans":
            (
                mejor_k["silhouette"]
                if mejor_k
                else None
            ),

        "interpretacion_kmeans":
            (
                "El clustering presenta solapamiento entre "
                "grupos, por lo que debe interpretarse como "
                "una estructura exploratoria."
            ),

        "interpretacion_nmf":
            (
                "NMF se utiliza como representación temática "
                "interpretable del corpus."
            ),
    }