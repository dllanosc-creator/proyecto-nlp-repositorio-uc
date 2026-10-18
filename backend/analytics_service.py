import pandas as pd


# ============================================================
# UTILIDADES
# ============================================================

def filtrar_documentos(
    df,
    anios=None,
    facultades=None,
    programas=None,
    topicos=None,
):
    """
    Aplica filtros al corpus principal.
    """

    resultado = df.copy()

    if anios:
        resultado = resultado[
            resultado["anio_publicacion"].isin(anios)
        ]

    if facultades:
        resultado = resultado[
            resultado["facultad"].isin(facultades)
        ]

    if programas:
        resultado = resultado[
            resultado["programa_academico"].isin(programas)
        ]

    if topicos:
        resultado = resultado[
            resultado["nombre_topico"].isin(topicos)
        ]

    return resultado


# ============================================================
# INDICADORES GENERALES
# ============================================================

def obtener_indicadores(df):
    """
    Retorna indicadores generales del corpus.
    """

    total_documentos = len(df)

    total_modelado = (
        int(df["apto_modelado"].sum())
        if "apto_modelado" in df.columns
        else 0
    )

    programas = (
        df["programa_academico"].nunique()
        if "programa_academico" in df.columns
        else 0
    )

    facultades = (
        df["facultad"].nunique()
        if "facultad" in df.columns
        else 0
    )

    return {
        "total_documentos": total_documentos,
        "total_modelado": total_modelado,
        "programas": programas,
        "facultades": facultades,
    }


# ============================================================
# PRODUCCIÓN POR AÑO
# ============================================================

def obtener_documentos_por_anio(df):

    if (
        df.empty
        or "anio_publicacion" not in df.columns
    ):
        return pd.DataFrame()

    resultado = (
        df
        .dropna(subset=["anio_publicacion"])
        .groupby("anio_publicacion")
        .size()
        .reset_index(name="documentos")
        .sort_values("anio_publicacion")
    )

    resultado["anio"] = (
        resultado["anio_publicacion"]
        .astype(int)
        .astype(str)
    )

    return resultado


# ============================================================
# DISTRIBUCIÓN POR FACULTAD
# ============================================================

def obtener_distribucion_facultades(df):

    if (
        df.empty
        or "facultad" not in df.columns
    ):
        return pd.DataFrame()

    resultado = (
        df["facultad"]
        .fillna("Sin información")
        .value_counts()
        .reset_index()
    )

    resultado.columns = [
        "facultad",
        "documentos",
    ]

    return resultado.sort_values(
        "documentos",
        ascending=False,
    )


# ============================================================
# DISTRIBUCIÓN POR PROGRAMA
# ============================================================

def obtener_distribucion_programas(
    df,
    top_n=None,
):

    if (
        df.empty
        or "programa_academico" not in df.columns
    ):
        return pd.DataFrame()

    resultado = (
        df["programa_academico"]
        .fillna("Sin información")
        .value_counts()
        .reset_index()
    )

    resultado.columns = [
        "programa",
        "documentos",
    ]

    if top_n is not None:
        resultado = resultado.head(top_n)

    return resultado


# ============================================================
# DISTRIBUCIÓN TEMÁTICA
# ============================================================

def obtener_distribucion_topicos(df):

    if (
        df.empty
        or "nombre_topico" not in df.columns
    ):
        return pd.DataFrame()

    resultado = df[
        df["nombre_topico"]
        != "Sin tópico asignado"
    ]

    resultado = (
        resultado["nombre_topico"]
        .value_counts()
        .reset_index()
    )

    resultado.columns = [
        "tematica",
        "documentos",
    ]

    total = resultado["documentos"].sum()

    if total > 0:
        resultado["porcentaje"] = (
            resultado["documentos"]
            / total
            * 100
        )

    else:
        resultado["porcentaje"] = 0

    return resultado


# ============================================================
# EVOLUCIÓN DE TÓPICOS
# ============================================================

def obtener_evolucion_topicos(
    df,
    modo="cantidad",
):

    if (
        df.empty
        or "anio_publicacion" not in df.columns
        or "nombre_topico" not in df.columns
    ):
        return pd.DataFrame()

    base = df[
        df["nombre_topico"]
        != "Sin tópico asignado"
    ].copy()

    base = base.dropna(
        subset=["anio_publicacion"]
    )

    resultado = (
        base
        .groupby(
            [
                "anio_publicacion",
                "nombre_topico",
            ]
        )
        .size()
        .reset_index(
            name="documentos"
        )
    )

    if modo == "porcentaje":

        totales = (
            resultado
            .groupby("anio_publicacion")[
                "documentos"
            ]
            .transform("sum")
        )

        resultado["valor"] = (
            resultado["documentos"]
            / totales
            * 100
        )

    else:

        resultado["valor"] = (
            resultado["documentos"]
        )

    resultado["anio"] = (
        resultado["anio_publicacion"]
        .astype(int)
        .astype(str)
    )

    return resultado


# ============================================================
# HEATMAP TÓPICO × AÑO
# ============================================================

def obtener_matriz_topicos_anio(df):

    if (
        df.empty
        or "anio_publicacion" not in df.columns
        or "nombre_topico" not in df.columns
    ):
        return pd.DataFrame()

    base = df[
        df["nombre_topico"]
        != "Sin tópico asignado"
    ].copy()

    base = base.dropna(
        subset=["anio_publicacion"]
    )

    matriz = (
        base
        .groupby(
            [
                "anio_publicacion",
                "nombre_topico",
            ]
        )
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    if not matriz.empty:
        matriz.index = (
            matriz.index
            .astype(int)
            .astype(str)
        )

    return matriz


# ============================================================
# MATRIZ PROGRAMA × TÓPICO
# ============================================================

def obtener_matriz_programa_topico(
    df,
    top_n_programas=20,
):

    if (
        df.empty
        or "programa_academico" not in df.columns
        or "nombre_topico" not in df.columns
    ):
        return pd.DataFrame()

    base = df[
        (
            df["nombre_topico"]
            != "Sin tópico asignado"
        )
        &
        (
            df["programa_academico"]
            .notna()
        )
    ].copy()

    matriz = (
        base
        .groupby(
            [
                "programa_academico",
                "nombre_topico",
            ]
        )
        .size()
        .unstack(fill_value=0)
    )

    if matriz.empty:
        return matriz

    totales = (
        matriz
        .sum(axis=1)
        .sort_values(
            ascending=False
        )
    )

    programas_top = (
        totales
        .head(top_n_programas)
        .index
    )

    return matriz.loc[
        programas_top
    ]


# ============================================================
# CAMBIOS ENTRE LOS DOS ÚLTIMOS AÑOS
# ============================================================

def obtener_cambio_topicos_ultimos_anios(df):

    if (
        df.empty
        or "anio_publicacion" not in df.columns
        or "nombre_topico" not in df.columns
    ):
        return pd.DataFrame()

    base = df[
        df["nombre_topico"]
        != "Sin tópico asignado"
    ].copy()

    base = base.dropna(
        subset=["anio_publicacion"]
    )

    anios = sorted(
        base["anio_publicacion"]
        .astype(int)
        .unique()
    )

    if len(anios) < 2:
        return pd.DataFrame()

    anio_anterior = anios[-2]
    anio_actual = anios[-1]

    conteo_anterior = (
        base[
            base["anio_publicacion"]
            == anio_anterior
        ]["nombre_topico"]
        .value_counts()
    )

    conteo_actual = (
        base[
            base["anio_publicacion"]
            == anio_actual
        ]["nombre_topico"]
        .value_counts()
    )

    resultado = pd.concat(
        [
            conteo_anterior.rename(
                str(anio_anterior)
            ),
            conteo_actual.rename(
                str(anio_actual)
            ),
        ],
        axis=1,
    ).fillna(0)

    resultado[
        "variacion_absoluta"
    ] = (
        resultado[str(anio_actual)]
        - resultado[str(anio_anterior)]
    )

    resultado = (
        resultado
        .reset_index()
        .rename(
            columns={
                "nombre_topico":
                    "tematica"
            }
        )
    )

    resultado[
        "anio_anterior"
    ] = anio_anterior

    resultado[
        "anio_actual"
    ] = anio_actual

    return resultado


# ============================================================
# HALLAZGO GENERAL
# ============================================================

def generar_hallazgo_panorama(df):

    if df.empty:
        return (
            "No hay documentos para los filtros seleccionados."
        )

    hallazgos = []

    # ========================================================
    # AÑO CON MAYOR PRODUCCIÓN
    # ========================================================

    if "anio_publicacion" in df.columns:

        conteo_anios = (
            df["anio_publicacion"]
            .dropna()
            .astype(int)
            .value_counts()
        )

        if not conteo_anios.empty:

            anio_max = int(
                conteo_anios.idxmax()
            )

            cantidad_max = int(
                conteo_anios.max()
            )

            hallazgos.append(
                f"El año con mayor número de registros es "
                f"{anio_max}, con {cantidad_max} documentos."
            )

    # ========================================================
    # DISTRIBUCIÓN TEMÁTICA
    # ========================================================

    if "nombre_topico" in df.columns:

        base_topicos = df[
            df["nombre_topico"]
            != "Sin tópico asignado"
        ].copy()

        conteo_topicos = (
            base_topicos[
                "nombre_topico"
            ]
            .value_counts()
        )

        if not conteo_topicos.empty:

            # ------------------------------------------------
            # Tópico principal
            # ------------------------------------------------

            topico_principal = (
                conteo_topicos.index[0]
            )

            cantidad_principal = int(
                conteo_topicos.iloc[0]
            )

            porcentaje_principal = (
                cantidad_principal
                / len(base_topicos)
                * 100
            )

            hallazgos.append(
                f"La temática con mayor presencia es "
                f"«{topico_principal}», con "
                f"{cantidad_principal} documentos "
                f"({porcentaje_principal:.1f}% del corpus "
                f"modelado seleccionado)."
            )

            # ------------------------------------------------
            # Concentración de las tres principales temáticas
            # ------------------------------------------------

            top_3 = (
                conteo_topicos
                .head(3)
            )

            total_top_3 = int(
                top_3.sum()
            )

            porcentaje_top_3 = (
                total_top_3
                / len(base_topicos)
                * 100
            )

            nombres_top_3 = (
                list(
                    top_3.index
                )
            )

            if len(nombres_top_3) == 3:

                hallazgos.append(
                    f"Las tres temáticas con mayor presencia "
                    f"son «{nombres_top_3[0]}», "
                    f"«{nombres_top_3[1]}» y "
                    f"«{nombres_top_3[2]}»; en conjunto "
                    f"representan {porcentaje_top_3:.1f}% "
                    f"del corpus modelado seleccionado."
                )

            # ------------------------------------------------
            # Diversidad temática
            # ------------------------------------------------

            numero_topicos = (
                conteo_topicos.shape[0]
            )

            hallazgos.append(
                f"La selección contiene documentos distribuidos "
                f"en {numero_topicos} temáticas identificadas "
                f"mediante NMF."
            )

    return " ".join(
        hallazgos
    )

def generar_hallazgo_tendencias(df):
    """
    Genera una lectura interpretativa simple sobre la evolución
    temática entre los dos últimos años disponibles.
    """

    if (
        df.empty
        or "anio_publicacion" not in df.columns
        or "nombre_topico" not in df.columns
    ):
        return (
            "No hay información suficiente para analizar tendencias."
        )

    base = df[
        df["nombre_topico"]
        != "Sin tópico asignado"
    ].copy()

    base = base.dropna(
        subset=["anio_publicacion"]
    )

    anios = sorted(
        base["anio_publicacion"]
        .astype(int)
        .unique()
    )

    if len(anios) < 2:
        return (
            "Se requieren al menos dos años con información "
            "para analizar cambios temáticos."
        )

    anio_anterior = anios[-2]
    anio_actual = anios[-1]

    conteo_anterior = (
        base[
            base["anio_publicacion"]
            == anio_anterior
        ]["nombre_topico"]
        .value_counts()
    )

    conteo_actual = (
        base[
            base["anio_publicacion"]
            == anio_actual
        ]["nombre_topico"]
        .value_counts()
    )

    comparacion = pd.concat(
        [
            conteo_anterior.rename("anterior"),
            conteo_actual.rename("actual"),
        ],
        axis=1,
    ).fillna(0)

    comparacion["variacion"] = (
        comparacion["actual"]
        - comparacion["anterior"]
    )

    if comparacion.empty:
        return (
            "No fue posible establecer variaciones temáticas "
            "entre los años disponibles."
        )

    crecimiento = (
        comparacion
        .sort_values(
            "variacion",
            ascending=False,
        )
        .iloc[0]
    )

    descenso = (
        comparacion
        .sort_values(
            "variacion",
            ascending=True,
        )
        .iloc[0]
    )

    topico_crece = (
        comparacion[
            comparacion["variacion"]
            == comparacion["variacion"].max()
        ]
        .index[0]
    )

    topico_desciende = (
        comparacion[
            comparacion["variacion"]
            == comparacion["variacion"].min()
        ]
        .index[0]
    )

    texto = (
        f"Entre {anio_anterior} y {anio_actual}, la temática con "
        f"mayor incremento absoluto es «{topico_crece}», con una "
        f"variación de {int(crecimiento['variacion'])} documentos. "
        f"La mayor disminución corresponde a "
        f"«{topico_desciende}», con una variación de "
        f"{int(descenso['variacion'])} documentos. "
    )

    if anio_actual == max(anios):
        texto += (
            "Estas variaciones deben interpretarse con cautela, "
            "especialmente si el año más reciente tiene cobertura "
            "parcial en el repositorio."
        )

    return texto