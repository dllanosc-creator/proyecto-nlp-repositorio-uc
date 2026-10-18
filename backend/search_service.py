from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from backend.data_service import (
    cargar_documentos_tfidf,
    cargar_documentos_transformers,
)


# ============================================================
# RUTAS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data" / "processed"

TFIDF_MATRIX_FILE = (
    MODELS_DIR
    / "tfidf_matrix_metadata_v3.npz"
)

TFIDF_VECTORIZER_FILE = (
    MODELS_DIR
    / "tfidf_vectorizer_metadata_v3.joblib"
)

TRANSFORMERS_MATRIX_FILE = (
    MODELS_DIR
    / "embeddings_transformers_metadata_v3.npy"
)


# ============================================================
# MODELO TRANSFORMER
# ============================================================

TRANSFORMER_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# ============================================================
# CARGA TF-IDF
# ============================================================
@lru_cache(maxsize=1)
def cargar_modelo_tfidf():
    """
    Carga vectorizador y matriz TF-IDF.
    """

    if not TFIDF_VECTORIZER_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró: {TFIDF_VECTORIZER_FILE}"
        )

    if not TFIDF_MATRIX_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró: {TFIDF_MATRIX_FILE}"
        )

    vectorizador = joblib.load(
        TFIDF_VECTORIZER_FILE
    )

    matriz = load_npz(
        TFIDF_MATRIX_FILE
    )

    return vectorizador, matriz


# ============================================================
# CARGA TRANSFORMERS
# ============================================================
@lru_cache(maxsize=1)
def cargar_embeddings_transformers():
    """
    Carga la matriz de embeddings ya generada.
    """

    if not TRANSFORMERS_MATRIX_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró: {TRANSFORMERS_MATRIX_FILE}"
        )

    return np.load(
        TRANSFORMERS_MATRIX_FILE
    )


@lru_cache(maxsize=1)
def cargar_modelo_transformer():
    """
    Carga el modelo Transformer una sola vez durante
    la ejecución de la aplicación.
    """

    return SentenceTransformer(
        TRANSFORMER_MODEL_NAME
    )


# ============================================================
# BÚSQUEDA TF-IDF
# ============================================================

def buscar_tfidf(
    consulta,
    top_n=10,
):
    """
    Recupera documentos mediante TF-IDF y similitud coseno.
    """

    consulta = str(
        consulta
    ).strip()

    if not consulta:
        return pd.DataFrame()

    vectorizador, matriz = (
        cargar_modelo_tfidf()
    )

    documentos = (
        cargar_documentos_tfidf()
    )

    if documentos.empty:
        return pd.DataFrame()

    vector_consulta = (
        vectorizador.transform(
            [consulta]
        )
    )

    similitudes = (
        cosine_similarity(
            vector_consulta,
            matriz,
        )
        .flatten()
    )

    cantidad = min(
        len(similitudes),
        len(documentos),
    )

    resultados = (
        documentos
        .iloc[:cantidad]
        .copy()
    )

    resultados[
        "similitud"
    ] = similitudes[
        :cantidad
    ]

    resultados[
        "modelo"
    ] = "TF-IDF"

    return (
        resultados
        .sort_values(
            "similitud",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )


# ============================================================
# BÚSQUEDA TRANSFORMERS
# ============================================================

def buscar_transformers(
    consulta,
    top_n=10,
):
    """
    Recupera documentos mediante embeddings semánticos
    y similitud coseno.
    """

    consulta = str(
        consulta
    ).strip()

    if not consulta:
        return pd.DataFrame()

    documentos = (
        cargar_documentos_transformers()
    )

    if documentos.empty:
        return pd.DataFrame()

    embeddings = (
        cargar_embeddings_transformers()
    )

    modelo = (
        cargar_modelo_transformer()
    )

    embedding_consulta = (
        modelo.encode(
            [consulta],
            normalize_embeddings=True,
        )
    )

    cantidad = min(
        len(documentos),
        embeddings.shape[0],
    )

    documentos = (
        documentos
        .iloc[:cantidad]
        .copy()
    )

    embeddings = (
        embeddings[:cantidad]
    )

    similitudes = (
        cosine_similarity(
            embedding_consulta,
            embeddings,
        )
        .flatten()
    )

    documentos[
        "similitud"
    ] = similitudes

    documentos[
        "modelo"
    ] = "Transformers"

    return (
        documentos
        .sort_values(
            "similitud",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )


# ============================================================
# COMPARACIÓN DE MÉTODOS
# ============================================================

def comparar_busquedas(
    consulta,
    top_n=10,
):
    """
    Ejecuta la misma consulta con TF-IDF y Transformers.
    """

    resultados_tfidf = buscar_tfidf(
        consulta,
        top_n=top_n,
    )

    resultados_transformers = (
        buscar_transformers(
            consulta,
            top_n=top_n,
        )
    )

    return {
        "tfidf": resultados_tfidf,
        "transformers":
            resultados_transformers,
    }


# ============================================================
# DOCUMENTOS COINCIDENTES ENTRE MÉTODOS
# ============================================================

def obtener_coincidencias_metodos(
    consulta,
    top_n=10,
):
    """
    Identifica documentos que aparecen dentro del Top-N
    de ambos métodos.
    """

    resultados = comparar_busquedas(
        consulta,
        top_n=top_n,
    )

    tfidf = resultados["tfidf"].copy()
    transformers = (
        resultados["transformers"].copy()
    )

    if (
        tfidf.empty
        or transformers.empty
    ):
        return pd.DataFrame()

    # --------------------------------------------------------
    # Detectar identificador común
    # --------------------------------------------------------

    posibles_ids = [
        "id_documento",
        "mms_id",
    ]

    id_comun = None

    for candidato in posibles_ids:

        if (
            candidato in tfidf.columns
            and candidato
            in transformers.columns
        ):
            id_comun = candidato
            break

    if id_comun is None:
        return pd.DataFrame()

    tfidf = tfidf.copy()
    transformers = transformers.copy()

    tfidf[
        id_comun
    ] = (
        tfidf[
            id_comun
        ]
        .astype(str)
    )

    transformers[
        id_comun
    ] = (
        transformers[
            id_comun
        ]
        .astype(str)
    )

    coincidencias = (
        tfidf.merge(
            transformers,
            on=id_comun,
            how="inner",
            suffixes=(
                "_tfidf",
                "_transformers",
            ),
        )
    )

    if coincidencias.empty:
        return coincidencias

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    tfidf_ranking = {
        str(valor): posicion + 1
        for posicion, valor
        in enumerate(
            tfidf[
                id_comun
            ].tolist()
        )
    }

    transformer_ranking = {
        str(valor): posicion + 1
        for posicion, valor
        in enumerate(
            transformers[
                id_comun
            ].tolist()
        )
    }

    coincidencias[
        "ranking_tfidf"
    ] = (
        coincidencias[
            id_comun
        ]
        .map(
            tfidf_ranking
        )
    )

    coincidencias[
        "ranking_transformers"
    ] = (
        coincidencias[
            id_comun
        ]
        .map(
            transformer_ranking
        )
    )

    return (
        coincidencias
        .sort_values(
            [
                "ranking_tfidf",
                "ranking_transformers",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# RESUMEN DE COMPARACIÓN
# ============================================================

def resumir_comparacion_busqueda(
    consulta,
    top_n=10,
):
    """
    Retorna un resumen simple de la coincidencia entre los
    dos métodos para una consulta.
    """

    coincidencias = (
        obtener_coincidencias_metodos(
            consulta,
            top_n=top_n,
        )
    )

    cantidad = len(
        coincidencias
    )

    porcentaje = (
        cantidad
        / top_n
        * 100
    )

    return {
        "consulta": consulta,
        "top_n": top_n,
        "coincidencias": cantidad,
        "porcentaje_coincidencia":
            porcentaje,
    }