"""Balanceo de clases y calibración para el problema multi-etiqueta."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score


def estadisticas_y_pos_weight(
    items: Sequence[tuple[Path, list[int]]],
    etiquetas: Sequence[str],
) -> tuple[torch.Tensor, dict]:
    """Calcula ``negativos / positivos`` usando exclusivamente el split train.

    Es la ponderación recomendada por ``BCEWithLogitsLoss`` para que cada
    positivo raro contribuya tanto como el conjunto de negativos de su clase.
    No se inspeccionan validación ni test para evitar fuga de información.
    """
    if not items:
        raise ValueError("No se puede calcular el balance de un train vacío.")

    matriz = np.asarray([vector for _, vector in items], dtype=np.float64)
    if matriz.ndim != 2 or matriz.shape[1] != len(etiquetas):
        raise ValueError(
            f"Se esperaban vectores de {len(etiquetas)} etiquetas; "
            f"se obtuvo una matriz {matriz.shape}."
        )
    if not np.isin(matriz, (0, 1)).all():
        raise ValueError("Las etiquetas de train deben contener solamente 0 o 1.")

    positivos = matriz.sum(axis=0)
    if np.any(positivos == 0):
        ausentes = [etiquetas[i] for i, valor in enumerate(positivos) if valor == 0]
        raise ValueError(f"Train no contiene positivos para: {', '.join(ausentes)}")

    negativos = len(items) - positivos
    pesos = negativos / positivos
    reporte = {
        "imagenes_train": len(items),
        "positivos": {etiqueta: int(positivos[i]) for i, etiqueta in enumerate(etiquetas)},
        "negativos": {etiqueta: int(negativos[i]) for i, etiqueta in enumerate(etiquetas)},
        "pos_weight": {etiqueta: float(pesos[i]) for i, etiqueta in enumerate(etiquetas)},
        "formula": "negativos / positivos, calculada solo con train",
    }
    return torch.tensor(pesos, dtype=torch.float32), reporte


def optimizar_umbrales(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    etiquetas: Sequence[str],
) -> dict[str, float]:
    """Elige por etiqueta el umbral con mayor F1 en validación.

    Se usa una malla fija para que el resultado sea estable y no dependa de
    detalles de ordenamiento de probabilidades idénticas. En empates se elige
    el umbral más cercano a 0.5.
    """
    reales = np.asarray(y_true, dtype=np.int8)
    probabilidades = np.asarray(y_prob, dtype=np.float64)
    forma_esperada = (reales.shape[0], len(etiquetas))
    if reales.shape != forma_esperada or probabilidades.shape != forma_esperada:
        raise ValueError(
            f"Formas incompatibles: y_true={reales.shape}, y_prob={probabilidades.shape}, "
            f"esperada={forma_esperada}."
        )

    candidatos = np.linspace(0.05, 0.95, 181)
    resultado: dict[str, float] = {}
    for indice, etiqueta in enumerate(etiquetas):
        puntuaciones = np.asarray([
            f1_score(
                reales[:, indice],
                probabilidades[:, indice] >= umbral,
                zero_division=0,
            )
            for umbral in candidatos
        ])
        mejor_f1 = puntuaciones.max()
        empatados = candidatos[np.isclose(puntuaciones, mejor_f1)]
        mejor_umbral = min(empatados, key=lambda valor: (abs(valor - 0.5), valor))
        resultado[etiqueta] = round(float(mejor_umbral), 3)
    return resultado


def aplicar_umbrales(probabilidades: np.ndarray, umbrales: np.ndarray) -> np.ndarray:
    """Probabilidades (n, etiquetas) -> predicción 0/1 con la misma regla que la app.

    Todo platillo tiene al menos un sabor: si en una foto ninguna probabilidad
    cruza su umbral, se marca el sabor más probable. Se usa igual al evaluar y
    al predecir para que las métricas midan lo que de verdad hace la interfaz.
    """
    probabilidades = np.atleast_2d(np.asarray(probabilidades, dtype=np.float64))
    prediccion = (probabilidades >= np.asarray(umbrales)).astype(int)
    vacias = ~prediccion.any(axis=1)
    prediccion[vacias, probabilidades[vacias].argmax(axis=1)] = 1
    return prediccion


def vector_umbrales(
    umbrales: dict[str, float] | None,
    etiquetas: Sequence[str],
    predeterminado: float = 0.5,
) -> np.ndarray:
    """Convierte el diccionario versionado al orden fijo de las etiquetas."""
    valores = [
        float(umbrales.get(etiqueta, predeterminado)) if umbrales else predeterminado
        for etiqueta in etiquetas
    ]
    if any(valor <= 0 or valor >= 1 for valor in valores):
        raise ValueError("Todos los umbrales deben estar estrictamente entre 0 y 1.")
    return np.asarray(valores, dtype=np.float32)
