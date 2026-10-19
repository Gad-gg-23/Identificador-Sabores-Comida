"""
Predicción sobre una imagen (o una carpeta de imágenes) con el modelo ya
entrenado. Uso:

    python predecir.py --imagen ruta/foto.jpg
    python predecir.py --carpeta ruta/con/varias/fotos/
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps

import balanceo
import config
import dataset
import modelo as modelo_mod

EXTENSIONES_VALIDAS = dataset.EXTENSIONES_VALIDAS


def cargar_modelo() -> torch.nn.Module:
    if not config.RUTA_MODELO.exists():
        raise FileNotFoundError(
            f"No se encontró '{config.RUTA_MODELO}'. Corre primero 'python Comida.py entrenar'."
        )
    # El checkpoint contiene también el backbone; no hace falta volver a
    # descargar los pesos ImageNet para realizar inferencia.
    modelo = modelo_mod.crear_modelo(preentrenado=False)
    modelo.load_state_dict(torch.load(config.RUTA_MODELO, map_location=config.DISPOSITIVO))
    modelo.eval()
    return modelo


def cargar_umbrales() -> dict[str, float]:
    """Carga la calibración del entrenamiento o conserva compatibilidad anterior."""
    if not config.RUTA_UMBRALES_JSON.exists():
        return {etiqueta: config.UMBRAL_DECISION for etiqueta in config.ETIQUETAS}
    contenido = json.loads(config.RUTA_UMBRALES_JSON.read_text(encoding="utf-8"))
    umbrales = contenido.get("umbrales", contenido)
    vector = balanceo.vector_umbrales(umbrales, config.ETIQUETAS, config.UMBRAL_DECISION)
    return {etiqueta: float(vector[i]) for i, etiqueta in enumerate(config.ETIQUETAS)}


def predecir_imagen(
    modelo: torch.nn.Module,
    ruta_imagen: Path,
    umbrales: dict[str, float] | None = None,
) -> dict:
    ruta_imagen = Path(ruta_imagen)
    if not ruta_imagen.is_file():
        raise FileNotFoundError(f"No se encontró la imagen '{ruta_imagen}'.")
    if ruta_imagen.suffix.lower() not in EXTENSIONES_VALIDAS:
        raise ValueError(
            f"Formato no compatible: '{ruta_imagen.suffix}'. "
            "Usa JPG, JPEG, PNG, BMP o WEBP."
        )
    transform = dataset.transform_evaluacion()
    with Image.open(ruta_imagen) as archivo_imagen:
        # Respeta la orientación EXIF de fotografías tomadas con celular.
        imagen = ImageOps.exif_transpose(archivo_imagen).convert("RGB")
    tensor = transform(imagen).unsqueeze(0).to(config.DISPOSITIVO)

    with torch.no_grad():
        salida = modelo(tensor)
        probabilidades = torch.sigmoid(salida).squeeze(0)

    probs_por_etiqueta = {
        etiqueta: float(probabilidades[i]) for i, etiqueta in enumerate(config.ETIQUETAS)
    }
    umbrales = umbrales or cargar_umbrales()
    etiquetas_predichas = [
        etiqueta for etiqueta, prob in probs_por_etiqueta.items() if prob >= umbrales[etiqueta]
    ]
    if not etiquetas_predichas:
        # Todo plato tiene algún sabor dominante: si ninguno cruza el umbral,
        # se regresa al menos el más probable en vez de dejar la lista vacía.
        etiquetas_predichas = [max(probs_por_etiqueta, key=probs_por_etiqueta.get)]

    return {
        "archivo": ruta_imagen.name,
        "etiquetas": etiquetas_predichas,
        "probabilidades": probs_por_etiqueta,
        "umbrales": umbrales,
    }


def main():
    parser = argparse.ArgumentParser(description="Predice los sabores de una imagen o carpeta de comida.")
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--imagen", type=str, help="Ruta a una sola imagen.")
    grupo.add_argument("--carpeta", type=str, help="Ruta a una carpeta con varias imágenes.")
    args = parser.parse_args()

    modelo = cargar_modelo()
    umbrales = cargar_umbrales()

    if args.imagen:
        rutas = [Path(args.imagen)]
    else:
        carpeta = Path(args.carpeta)
        rutas = sorted(p for p in carpeta.iterdir() if p.suffix.lower() in EXTENSIONES_VALIDAS)
        if not rutas:
            print(f"No se encontraron imágenes en '{carpeta}'.")
            return

    for ruta in rutas:
        resultado = predecir_imagen(modelo, ruta, umbrales)
        probs_texto = ", ".join(
            f"{etiqueta}={prob:.3f}" for etiqueta, prob in resultado["probabilidades"].items()
        )
        etiquetas_texto = ", ".join(e.upper() for e in resultado["etiquetas"])
        print(f"{resultado['archivo']}: {etiquetas_texto} ({probs_texto})")


if __name__ == "__main__":
    main()
