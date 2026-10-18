from pathlib import Path

import pandas as pd


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"


# ============================================================
# ARCHIVOS PRINCIPALES
# ============================================================

WEBAPP_FILE = DATA_DIR / "webapp_documentos_final.csv"

COMPARACION_VECTOR_FILE = DATA_DIR / "comparacion_vectorizaciones.csv"
EVALUACION_TFIDF_FILE = DATA_DIR / "evaluacion_tfidf_resumen.csv"
EVALUACION_TRANSFORMERS_FILE = DATA_DIR / "evaluacion_transformers_resumen.csv"

EVALUACION_KMEANS_FILE = DATA_DIR / "evaluacion_kmeans.csv"

TERMINOS_CLUSTER_FILE = DATA_DIR / "terminos_por_cluster.csv"
RESUMEN_CLUSTER_FILE = DATA_DIR / "resumen_clusters_kmeans.csv"
PROGRAMAS_CLUSTER_FILE = DATA_DIR / "programas_por_cluster.csv"

TOPICOS_NMF_FILE = DATA_DIR / "topicos_nmf.csv"
DOCUMENTOS_TOPICOS_FILE = DATA_DIR / "documentos_topicos_nmf.csv"

TFIDF_DOCS_FILE = DATA_DIR / "tfidf_documentos_metadata_v3.csv"
TFIDF_SIMILARES_FILE = DATA_DIR / "tfidf_documentos_similares_top10.csv"

TRANSFORMERS_DOCS_FILE = DATA_DIR / "embeddings_documentos_metadata_v3.csv"
TRANSFORMERS_SIMILARES_FILE = (
    DATA_DIR / "transformers_documentos_similares_top10.csv"
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def normalizar_anio(valor):
    try:
        return int(float(valor))
    except Exception:
        return pd.NA


def cargar_csv_seguro(ruta, dtype=None):
    """
    Carga un CSV si existe.
    Si no existe o presenta error, retorna un DataFrame vacío.
    """

    if not ruta.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            ruta,
            dtype=dtype,
        )

    except Exception as error:
        print(
            f"Error cargando {ruta.name}: {error}"
        )

        return pd.DataFrame()


# ============================================================
# CORPUS PRINCIPAL PARA LA WEB APP
# ============================================================

def cargar_documentos():
    """
    Carga el dataset consolidado utilizado por la aplicación.
    """

    df = cargar_csv_seguro(
        WEBAPP_FILE,
        dtype={
            "id_documento": str,
            "mms_id": str,
        },
    )

    if df.empty:
        return df

    if "anio_publicacion" in df.columns:

        df["anio_publicacion"] = (
            df["anio_publicacion"]
            .apply(normalizar_anio)
        )

    if "apto_modelado" in df.columns:

        if df["apto_modelado"].dtype != bool:

            df["apto_modelado"] = (
                df["apto_modelado"]
                .astype(str)
                .str.lower()
                .isin(
                    [
                        "true",
                        "1",
                        "si",
                        "sí",
                    ]
                )
            )

    return df


# ============================================================
# RESULTADOS DE EVALUACIÓN
# ============================================================

def cargar_comparacion_vectorizaciones():
    return cargar_csv_seguro(
        COMPARACION_VECTOR_FILE
    )


def cargar_evaluacion_tfidf():
    return cargar_csv_seguro(
        EVALUACION_TFIDF_FILE
    )


def cargar_evaluacion_transformers():
    return cargar_csv_seguro(
        EVALUACION_TRANSFORMERS_FILE
    )


def cargar_evaluacion_kmeans():
    return cargar_csv_seguro(
        EVALUACION_KMEANS_FILE
    )


# ============================================================
# RESULTADOS DE CLUSTERING
# ============================================================

def cargar_terminos_clusters():
    return cargar_csv_seguro(
        TERMINOS_CLUSTER_FILE
    )


def cargar_resumen_clusters():
    return cargar_csv_seguro(
        RESUMEN_CLUSTER_FILE
    )


def cargar_programas_clusters():
    return cargar_csv_seguro(
        PROGRAMAS_CLUSTER_FILE
    )


# ============================================================
# RESULTADOS NMF
# ============================================================

def cargar_topicos_nmf():
    return cargar_csv_seguro(
        TOPICOS_NMF_FILE
    )


def cargar_documentos_topicos_nmf():
    return cargar_csv_seguro(
        DOCUMENTOS_TOPICOS_FILE
    )


# ============================================================
# TF-IDF
# ============================================================

def cargar_documentos_tfidf():
    return cargar_csv_seguro(
        TFIDF_DOCS_FILE,
        dtype={
            "id_documento": str
        },
    )


def cargar_similares_tfidf():
    return cargar_csv_seguro(
        TFIDF_SIMILARES_FILE
    )


# ============================================================
# TRANSFORMERS
# ============================================================

def cargar_documentos_transformers():
    return cargar_csv_seguro(
        TRANSFORMERS_DOCS_FILE,
        dtype={
            "id_documento": str
        },
    )


def cargar_similares_transformers():
    return cargar_csv_seguro(
        TRANSFORMERS_SIMILARES_FILE
    )
