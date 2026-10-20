"""
Entrenamiento del identificador multi-etiqueta de sabores.

Flujo:
  1) Construir los splits train/val/test (dataset.construir_splits)
  2) Preparar los DataLoaders con augmentation en train
  3) Crear el modelo (transfer learning, modelo.crear_modelo)
  4) Entrenar con BCEWithLogitsLoss balanceada (no CrossEntropy: cada sabor
     es un sí/no independiente) y early stopping sobre F1 "por ejemplo" de val
  5) Calibrar un umbral por sabor exclusivamente sobre validación
  6) Evaluar en test: Accuracy exacta, F1 macro/por-ejemplo, métricas por
     sabor, y una matriz de confusión por cada sabor
  7) Guardar modelo, balance, umbrales, histórico y métricas
"""

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    multilabel_confusion_matrix,
)
from torch.utils.data import DataLoader

import config
import balanceo
import dataset
import modelo as modelo_mod


def _crear_dataloaders():
    splits = dataset.construir_splits()

    for nombre in ("train", "val", "test"):
        print(f"  {nombre}: {len(splits[nombre])} imágenes")

    ds_train = dataset.ImagenesComida(splits["train"], dataset.transform_entrenamiento())
    ds_val = dataset.ImagenesComida(splits["val"], dataset.transform_evaluacion())
    ds_test = dataset.ImagenesComida(splits["test"], dataset.transform_evaluacion())

    dl_train = DataLoader(
        ds_train, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, persistent_workers=config.NUM_WORKERS > 0,
    )
    dl_val = DataLoader(
        ds_val, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, persistent_workers=config.NUM_WORKERS > 0,
    )
    dl_test = DataLoader(
        ds_test, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, persistent_workers=config.NUM_WORKERS > 0,
    )
    pos_weight, reporte_balance = balanceo.estadisticas_y_pos_weight(
        splits["train"], config.ETIQUETAS
    )
    return dl_train, dl_val, dl_test, pos_weight, reporte_balance


def _correr_epoca(modelo, dataloader, criterio, optimizador=None, umbrales=None):
    """Una pasada completa. Si optimizador es None, es modo evaluación (sin backprop)."""
    entrenando = optimizador is not None
    if entrenando:
        modelo_mod.modo_entrenamiento(modelo)
    else:
        modelo.eval()

    perdida_total = 0.0
    y_true, y_prob = [], []
    vector_umbrales = balanceo.vector_umbrales(
        umbrales, config.ETIQUETAS, config.UMBRAL_DECISION
    )

    contexto = torch.enable_grad() if entrenando else torch.no_grad()
    with contexto:
        for imagenes, etiquetas in dataloader:
            imagenes = imagenes.to(config.DISPOSITIVO)
            etiquetas = etiquetas.to(config.DISPOSITIVO)

            if entrenando:
                optimizador.zero_grad()

            salidas = modelo(imagenes)
            perdida = criterio(salidas, etiquetas)

            if entrenando:
                perdida.backward()
                optimizador.step()

            perdida_total += perdida.item() * imagenes.size(0)
            y_true.extend(etiquetas.int().cpu().tolist())
            y_prob.extend(torch.sigmoid(salidas).cpu().tolist())

    n = len(y_true)
    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)
    y_pred_arr = balanceo.aplicar_umbrales(y_prob_arr, vector_umbrales)
    metricas = {
        "perdida": perdida_total / n,
        # Accuracy "exacta": las 5 etiquetas deben coincidir perfectamente.
        # Es una vara muy alta a propósito -- F1 abajo es más indulgente.
        "accuracy_exacta": accuracy_score(y_true_arr, y_pred_arr),
        # F1 "por ejemplo": promedio del F1 de cada foto individual (cuántos
        # sabores le atinó de los que tiene / de los que predijo). Es la
        # métrica que mejor refleja "qué tan útil es la predicción para
        # alguien viendo una sola foto", así que es la que usa early stopping.
        "f1_muestras": f1_score(y_true_arr, y_pred_arr, average="samples", zero_division=0),
        # F1 "macro": promedio del F1 de cada sabor por separado, sin pesar
        # por frecuencia -- si el modelo ignora "amarga" (poco común) se nota.
        "f1_macro": f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0),
    }
    return metricas, y_true_arr, y_pred_arr, y_prob_arr


def entrenar():
    config.RUTA_SALIDAS.mkdir(parents=True, exist_ok=True)

    print("Construyendo splits (Food-101 + fotos propias + FoodSense curado)...")
    dl_train, dl_val, dl_test, pos_weight, reporte_balance = _crear_dataloaders()
    config.RUTA_BALANCE_JSON.write_text(
        json.dumps(reporte_balance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\nBalance calculado solamente con train:")
    for etiqueta in config.ETIQUETAS:
        positivos = reporte_balance["positivos"][etiqueta]
        peso = reporte_balance["pos_weight"][etiqueta]
        print(f"  {etiqueta}: {positivos} positivos, pos_weight={peso:.3f}")

    print(f"\nDispositivo: {config.DISPOSITIVO}")
    modelo = modelo_mod.crear_modelo(afinar_completo=config.AFINAR_COMPLETO)
    entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {entrenables:,} "
          f"(afinar_completo={config.AFINAR_COMPLETO})")
    criterio = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(config.DISPOSITIVO))
    optimizador = torch.optim.Adam(modelo_mod.grupos_de_parametros(modelo))

    mejor_f1_val = -1.0
    epocas_sin_mejorar = 0
    historico = []

    print("\nEntrenando...")
    for epoca in range(1, config.NUM_EPOCAS + 1):
        inicio = time.time()

        metricas_train, _, _, _ = _correr_epoca(modelo, dl_train, criterio, optimizador)
        metricas_val, y_val, _, y_prob_val = _correr_epoca(modelo, dl_val, criterio)
        umbrales_epoca = balanceo.optimizar_umbrales(
            y_val, y_prob_val, config.ETIQUETAS
        )
        pred_val_calibrada = balanceo.aplicar_umbrales(
            y_prob_val,
            balanceo.vector_umbrales(umbrales_epoca, config.ETIQUETAS, config.UMBRAL_DECISION),
        )
        metricas_val["accuracy_exacta"] = accuracy_score(y_val, pred_val_calibrada)
        metricas_val["f1_muestras"] = f1_score(
            y_val, pred_val_calibrada, average="samples", zero_division=0
        )
        metricas_val["f1_macro"] = f1_score(
            y_val, pred_val_calibrada, average="macro", zero_division=0
        )

        duracion = time.time() - inicio
        print(
            f"Época {epoca:02d}/{config.NUM_EPOCAS} "
            f"- train: perdida={metricas_train['perdida']:.4f} acc_exacta={metricas_train['accuracy_exacta']:.4f} "
            f"- val: perdida={metricas_val['perdida']:.4f} acc_exacta={metricas_val['accuracy_exacta']:.4f} "
            f"f1_muestras={metricas_val['f1_muestras']:.4f} f1_macro={metricas_val['f1_macro']:.4f} "
            f"({duracion:.1f}s)"
        )

        historico.append({
            "epoca": epoca,
            "perdida_train": metricas_train["perdida"],
            "accuracy_exacta_train": metricas_train["accuracy_exacta"],
            "perdida_val": metricas_val["perdida"],
            "accuracy_exacta_val": metricas_val["accuracy_exacta"],
            "f1_muestras_val": metricas_val["f1_muestras"],
            "f1_macro_val": metricas_val["f1_macro"],
        })

        if metricas_val["f1_muestras"] > mejor_f1_val:
            mejor_f1_val = metricas_val["f1_muestras"]
            epocas_sin_mejorar = 0
            torch.save(modelo.state_dict(), config.RUTA_MODELO)
            print(f"  -> Nuevo mejor modelo guardado (F1 muestras val = {mejor_f1_val:.4f})")
        else:
            epocas_sin_mejorar += 1
            if epocas_sin_mejorar >= config.PACIENCIA_EARLY_STOPPING:
                print(f"\nEarly stopping: {config.PACIENCIA_EARLY_STOPPING} épocas sin mejorar F1 de validación.")
                break

    pd.DataFrame(historico).to_csv(config.RUTA_HISTORICO_CSV, index=False)
    print(f"\nHistórico de entrenamiento guardado en '{config.RUTA_HISTORICO_CSV}'")

    # --- Evaluación final en test, con el MEJOR modelo (no el último) ---
    print("\nCargando mejor checkpoint para evaluar en test...")
    modelo.load_state_dict(torch.load(config.RUTA_MODELO, map_location=config.DISPOSITIVO))

    # ``pos_weight`` corrige el aprendizaje, pero desplaza las probabilidades.
    # Por eso cada umbral se calibra con validación y solo después se toca test.
    _, y_val, _, y_prob_val = _correr_epoca(modelo, dl_val, criterio)
    umbrales = balanceo.optimizar_umbrales(y_val, y_prob_val, config.ETIQUETAS)
    reporte_umbrales = {
        "metodo": "umbral por sabor que maximiza F1 en validation",
        "split_calibracion": "validation",
        "umbrales": umbrales,
    }
    config.RUTA_UMBRALES_JSON.write_text(
        json.dumps(reporte_umbrales, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Umbrales calibrados: " + ", ".join(
        f"{etiqueta}={umbrales[etiqueta]:.3f}" for etiqueta in config.ETIQUETAS
    ))

    metricas_test, y_true, y_pred, _ = _correr_epoca(
        modelo, dl_test, criterio, umbrales=umbrales
    )

    resultados = {
        "Accuracy_exacta": metricas_test["accuracy_exacta"],
        "F1_muestras": metricas_test["f1_muestras"],
        "F1_macro": metricas_test["f1_macro"],
    }

    print("\n=== Resultados en test ===")
    for nombre, valor in resultados.items():
        print(f"{nombre}: {valor:.4f}")

    pd.DataFrame([resultados]).to_csv(config.RUTA_METRICAS_CSV, index=False)
    print(f"\nMétricas guardadas en '{config.RUTA_METRICAS_CSV}'")

    # --- Métricas por sabor (precision/recall/f1/support) ---
    reporte = classification_report(
        y_true, y_pred, target_names=config.ETIQUETAS, output_dict=True, zero_division=0,
    )
    pd.DataFrame(reporte).transpose().to_csv(config.RUTA_METRICAS_POR_ETIQUETA_CSV)
    print(f"Métricas por sabor guardadas en '{config.RUTA_METRICAS_POR_ETIQUETA_CSV}'")

    # --- Una matriz de confusión 2x2 por cada sabor ---
    matrices = multilabel_confusion_matrix(y_true, y_pred)
    fig, ejes = plt.subplots(1, len(config.ETIQUETAS), figsize=(4 * len(config.ETIQUETAS), 4))
    for eje, etiqueta, matriz in zip(ejes, config.ETIQUETAS, matrices):
        eje.imshow(matriz, cmap="Blues")
        eje.set_title(etiqueta)
        eje.set_xticks([0, 1])
        eje.set_yticks([0, 1])
        eje.set_xticklabels(["no", "sí"])
        eje.set_yticklabels(["no", "sí"])
        eje.set_xlabel("Predicho")
        eje.set_ylabel("Real")
        for i in range(2):
            for j in range(2):
                eje.text(j, i, str(matriz[i, j]), ha="center", va="center")
    plt.suptitle("Matrices de confusión por sabor (test)")
    plt.tight_layout()
    plt.savefig(config.RUTA_MATRICES_CONFUSION, dpi=150)
    plt.close()
    print(f"Matrices de confusión guardadas en '{config.RUTA_MATRICES_CONFUSION}'")

    print(f"\nModelo final en '{config.RUTA_MODELO}'")


if __name__ == "__main__":
    entrenar()
