"""Auditoría reproducible de los datasets de imágenes y etiquetas."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


RAIZ = Path(__file__).resolve().parent
ETIQUETAS = ("acida", "amarga", "dulce", "picosa", "salada")
EXTENSIONES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _sha256(ruta: Path) -> str:
    digest = hashlib.sha256()
    with open(ruta, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def auditar(carpeta_imagenes: Path, ruta_csv: Path) -> dict:
    problemas: list[str] = []
    if not ruta_csv.exists():
        return {"csv": str(ruta_csv), "error": "CSV inexistente", "problemas": ["CSV inexistente"]}
    if not carpeta_imagenes.exists():
        return {
            "csv": str(ruta_csv), "imagenes": str(carpeta_imagenes),
            "error": "Carpeta de imágenes inexistente", "problemas": ["Carpeta inexistente"],
        }

    with open(ruta_csv, newline="", encoding="utf-8-sig") as archivo:
        lector = csv.DictReader(archivo)
        columnas = lector.fieldnames or []
        faltantes = [columna for columna in ("archivo", *ETIQUETAS) if columna not in columnas]
        if faltantes:
            problemas.append(f"Columnas obligatorias ausentes: {faltantes}")
        filas = list(lector)

    nombres_csv = [fila.get("archivo", "").strip() for fila in filas]
    repetidos_csv = sorted(nombre for nombre, n in Counter(nombres_csv).items() if nombre and n > 1)
    if repetidos_csv:
        problemas.append(f"{len(repetidos_csv)} nombres repetidos en CSV")

    archivos_imagen = sorted(
        ruta for ruta in carpeta_imagenes.iterdir()
        if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_VALIDAS
    )
    nombres_imagen = {ruta.name for ruta in archivos_imagen}
    nombres_csv_set = {nombre for nombre in nombres_csv if nombre}
    sin_etiqueta = sorted(nombres_imagen - nombres_csv_set)
    sin_imagen = sorted(nombres_csv_set - nombres_imagen)
    if sin_etiqueta:
        problemas.append(f"{len(sin_etiqueta)} imágenes sin fila en CSV")
    if sin_imagen:
        problemas.append(f"{len(sin_imagen)} filas sin imagen")

    conteos = Counter()
    vectores_vacios = 0
    valores_invalidos: list[str] = []
    splits = Counter()
    for numero, fila in enumerate(filas, start=2):
        vector = []
        for etiqueta in ETIQUETAS:
            valor = fila.get(etiqueta, "")
            if valor not in ("0", "1"):
                valores_invalidos.append(f"fila {numero}, {etiqueta}={valor!r}")
                continue
            entero = int(valor)
            vector.append(entero)
            conteos[etiqueta] += entero
        if len(vector) == len(ETIQUETAS) and not any(vector):
            vectores_vacios += 1
        if fila.get("split"):
            splits[fila["split"]] += 1
    if valores_invalidos:
        problemas.append(f"{len(valores_invalidos)} valores de etiqueta inválidos")
    if vectores_vacios:
        problemas.append(f"{vectores_vacios} filas sin ningún sabor positivo")

    corruptas: list[str] = []
    hashes: dict[str, list[str]] = defaultdict(list)
    for ruta in archivos_imagen:
        try:
            with Image.open(ruta) as imagen:
                imagen.verify()
            hashes[_sha256(ruta)].append(ruta.name)
        except Exception as error:
            corruptas.append(f"{ruta.name}: {error}")
    duplicados = [nombres for nombres in hashes.values() if len(nombres) > 1]
    if corruptas:
        problemas.append(f"{len(corruptas)} imágenes corruptas")
    if duplicados:
        problemas.append(f"{len(duplicados)} grupos de duplicados exactos")

    return {
        "csv": str(ruta_csv),
        "imagenes": str(carpeta_imagenes),
        "filas_csv": len(filas),
        "archivos_imagen": len(archivos_imagen),
        "conteos_etiquetas": dict(conteos),
        "conteos_splits": dict(splits),
        "imagenes_sin_etiqueta": sin_etiqueta,
        "filas_sin_imagen": sin_imagen,
        "nombres_repetidos_csv": repetidos_csv,
        "valores_invalidos": valores_invalidos,
        "vectores_vacios": vectores_vacios,
        "imagenes_corruptas": corruptas,
        "duplicados_exactos": duplicados,
        "problemas": problemas,
        "estado": "OK" if not problemas else "REVISAR",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audita integridad, balance y duplicados del dataset.")
    parser.add_argument(
        "--dataset", choices=("propio", "foodsense", "todos"), default="todos",
        help="Dataset que se revisará.",
    )
    parser.add_argument("--salida-json", type=Path, help="Guarda también el reporte en JSON.")
    args = parser.parse_args()

    objetivos = {}
    if args.dataset in ("propio", "todos"):
        objetivos["fotos_propias"] = (
            RAIZ / "data/fotos_propias/imagenes", RAIZ / "data/fotos_propias/etiquetas.csv"
        )
    if args.dataset in ("foodsense", "todos"):
        objetivos["foodsense_curado"] = (
            RAIZ / "data/foodsense_curado/imagenes", RAIZ / "data/foodsense_curado/etiquetas.csv"
        )

    reporte = {nombre: auditar(*rutas) for nombre, rutas in objetivos.items()}
    texto = json.dumps(reporte, ensure_ascii=False, indent=2)
    print(texto)
    if args.salida_json:
        args.salida_json.parent.mkdir(parents=True, exist_ok=True)
        args.salida_json.write_text(texto + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
