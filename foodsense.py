"""Preparación reproducible del complemento humano FoodSense.

FoodSense contiene varias opiniones por imagen. Este módulo las agrega por
consenso y genera un CSV compatible con el resto del proyecto. Por defecto se
conservan todos los ejemplos con consenso; el desbalance restante se corrige en
la pérdida de entrenamiento. Las imágenes se descargan desde una revisión fija
de Hugging Face y no se guardan en Git.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
import unicodedata
import urllib.parse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
import requests


RAIZ = Path(__file__).resolve().parent
RUTA_DESTINO = RAIZ / "data" / "foodsense_curado"
RUTA_DATASET_PROPIO_CSV = RAIZ / "data" / "fotos_propias" / "etiquetas.csv"
RUTA_IMAGENES_PROPIAS = RAIZ / "data" / "fotos_propias" / "imagenes"

REPOSITORIO_HF = "sababishraq/foodsense-dataset"
# Revisión fija para que futuras modificaciones upstream no cambien el experimento.
REVISION_HF = "843b1ae42b18deddfb4452362ae1a06202ade1cb"
LICENCIA = "CC BY 4.0"
ETIQUETAS = ("acida", "amarga", "dulce", "picosa", "salada")

# Vocabulario conservador. No se equipara "savory" con "salada".
LEXICO = {
    "acida": {"sour", "tangy", "tart", "acidic", "citrus", "citrusy"},
    "amarga": {"bitter"},
    "dulce": {"sweet", "sugary"},
    "picosa": {"spicy", "spice", "peppery", "chili", "chilli"},
    "salada": {"salty", "salt", "briny"},
}

NEGACIONES = {"no", "not", "without", "isnt", "wasnt", "non"}
RESPUESTAS_INCIERTAS = {
    "unknown", "unsure", "not sure", "cant tell", "cannot tell", "no idea", "na"
}
EXTENSIONES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _normalizar_texto(texto: object) -> list[str]:
    texto_ascii = (
        unicodedata.normalize("NFKD", str(texto).lower())
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    return re.findall(r"[a-z]+", texto_ascii)


def sabores_en_descriptor(texto: object) -> set[str]:
    """Extrae sabores explícitos sin convertir `savory` en `salada`.

    Una palabra se ignora si aparece cerca de una negación, por ejemplo
    ``not too sweet`` o ``no spice``.
    """
    tokens = _normalizar_texto(texto)
    if not tokens or " ".join(tokens) in RESPUESTAS_INCIERTAS:
        return set()

    encontrados: set[str] = set()
    for etiqueta, terminos in LEXICO.items():
        for indice, token in enumerate(tokens):
            if token not in terminos:
                continue
            contexto_previo = tokens[max(0, indice - 3):indice]
            if not any(palabra in NEGACIONES for palabra in contexto_previo):
                encontrados.add(etiqueta)
    return encontrados


def _url_huggingface(archivo: str) -> str:
    ruta = urllib.parse.quote(archivo, safe="/")
    return (
        f"https://huggingface.co/datasets/{REPOSITORIO_HF}/resolve/"
        f"{REVISION_HF}/{ruta}?download=true"
    )


def _descargar(url: str, destino: Path, reintentos: int = 3) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists() and destino.stat().st_size > 0:
        return

    ultimo_error: Exception | None = None
    for intento in range(1, reintentos + 1):
        temporal = destino.with_suffix(destino.suffix + ".tmp")
        try:
            with requests.get(
                url,
                headers={"User-Agent": "Identificador-Sabores-Comida/1.0"},
                timeout=90,
                stream=True,
            ) as respuesta:
                respuesta.raise_for_status()
                with open(temporal, "wb") as salida:
                    for bloque in respuesta.iter_content(chunk_size=1024 * 1024):
                        if bloque:
                            salida.write(bloque)
            os.replace(temporal, destino)
            return
        except Exception as error:  # pragma: no cover - depende de la red
            ultimo_error = error
            temporal.unlink(missing_ok=True)
            if intento == reintentos:
                break
    raise RuntimeError(f"No se pudo descargar {url}: {ultimo_error}")


def _leer_conteos_existentes(ruta_csv: Path) -> Counter[str]:
    conteos: Counter[str] = Counter()
    if not ruta_csv.exists():
        return conteos
    with open(ruta_csv, newline="", encoding="utf-8-sig") as archivo:
        for fila in csv.DictReader(archivo):
            for etiqueta in ETIQUETAS:
                conteos[etiqueta] += int(fila.get(etiqueta, 0) or 0)
    return conteos


def construir_candidatos(
    ruta_metadata: Path,
    min_votos: int = 3,
    min_fraccion: float = 0.12,
) -> list[dict]:
    """Agrega las opiniones de FoodSense y devuelve una fila por imagen."""
    grupos: dict[str, list[dict[str, str]]] = defaultdict(list)
    with open(ruta_metadata, newline="", encoding="utf-8-sig") as archivo:
        for fila in csv.DictReader(archivo):
            if fila.get("split") == "excluded":
                continue
            grupos[fila["file_name"]].append(fila)

    candidatos: list[dict] = []
    for nombre_archivo, filas in sorted(grupos.items()):
        splits = {fila["split"] for fila in filas}
        if len(splits) != 1:
            raise ValueError(f"La imagen {nombre_archivo!r} aparece en varios splits: {splits}")

        votos: Counter[str] = Counter()
        validas = 0
        for fila in filas:
            if fila.get("CanInfer_taste") != "1" or not fila.get("taste_desc", "").strip():
                continue
            validas += 1
            votos.update(sabores_en_descriptor(fila["taste_desc"]))

        if validas == 0:
            continue
        sabores = {
            etiqueta
            for etiqueta in ETIQUETAS
            if votos[etiqueta] >= min_votos
            and votos[etiqueta] / validas >= min_fraccion
        }
        if not sabores:
            continue

        candidato = {
            "archivo": nombre_archivo,
            "split": next(iter(splits)),
            "n_anotaciones_validas": validas,
            "sabores": sabores,
        }
        for etiqueta in ETIQUETAS:
            candidato[f"votos_{etiqueta}"] = votos[etiqueta]
            candidato[f"fraccion_{etiqueta}"] = votos[etiqueta] / validas
        candidatos.append(candidato)
    return candidatos


def seleccionar_para_balance(
    candidatos: list[dict],
    conteos_existentes: Counter[str],
) -> tuple[list[dict], dict[str, int]]:
    """Selecciona ejemplos que cubren carencias sin inflar clases abundantes.

    El objetivo automático es alcanzar la clase no-salada más numerosa del
    dataset propio. Si no existen candidatos suficientes (como ocurre con
    amarga), se conservan todos los disponibles para esa clase.
    """
    no_salada = [conteos_existentes[etiqueta] for etiqueta in ETIQUETAS if etiqueta != "salada"]
    objetivo = max(no_salada, default=0)
    deficit = {
        etiqueta: max(0, objetivo - conteos_existentes[etiqueta])
        for etiqueta in ETIQUETAS
    }
    deficit["salada"] = 0

    pendientes = list(candidatos)
    seleccionados: list[dict] = []
    agregados: Counter[str] = Counter()

    while pendientes:
        def puntuacion(candidato: dict) -> tuple[float, float, str]:
            ganancia = 0.0
            confianza = 0.0
            # Orden fijo: evita que la aleatorización interna de los sets
            # pueda alterar desempates entre ejecuciones de Python.
            for etiqueta in ETIQUETAS:
                if etiqueta not in candidato["sabores"]:
                    continue
                restante = max(0, deficit[etiqueta] - agregados[etiqueta])
                if restante:
                    ganancia += restante / max(1, objetivo)
                    confianza += candidato[f"fraccion_{etiqueta}"]
            return ganancia, confianza, candidato["archivo"]

        mejor = max(pendientes, key=puntuacion)
        if puntuacion(mejor)[0] <= 0:
            break
        pendientes.remove(mejor)
        seleccionados.append(mejor)
        agregados.update(mejor["sabores"])

    return sorted(seleccionados, key=lambda x: x["archivo"]), deficit


def _sha256(ruta: Path) -> str:
    digest = hashlib.sha256()
    with open(ruta, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def _hashes_imagenes(carpeta: Path) -> dict[str, Path]:
    hashes: dict[str, Path] = {}
    if not carpeta.exists():
        return hashes
    for ruta in carpeta.iterdir():
        if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_VALIDAS:
            hashes.setdefault(_sha256(ruta), ruta)
    return hashes


def descargar_imagenes(
    filas: list[dict],
    carpeta_imagenes: Path,
    workers: int = 8,
) -> None:
    carpeta_imagenes.mkdir(parents=True, exist_ok=True)
    faltantes = [fila["archivo"] for fila in filas if not (carpeta_imagenes / fila["archivo"]).exists()]
    if not faltantes:
        return
    print(f"Descargando {len(faltantes)} imágenes curadas de FoodSense...")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ejecutor:
        futuros = {
            ejecutor.submit(_descargar, _url_huggingface(nombre), carpeta_imagenes / nombre): nombre
            for nombre in faltantes
        }
        completadas = 0
        for futuro in as_completed(futuros):
            futuro.result()
            completadas += 1
            if completadas % 100 == 0 or completadas == len(faltantes):
                print(f"  {completadas}/{len(faltantes)}")


def _validar_y_deduplicar(filas: list[dict], carpeta_imagenes: Path) -> tuple[list[dict], list[dict]]:
    hashes_propios = _hashes_imagenes(RUTA_IMAGENES_PROPIAS)
    hashes_foodsense: dict[str, Path] = {}
    validas: list[dict] = []
    descartadas: list[dict] = []

    for fila in filas:
        ruta = carpeta_imagenes / fila["archivo"]
        try:
            with Image.open(ruta) as imagen:
                imagen.verify()
        except Exception as error:
            descartadas.append({"archivo": fila["archivo"], "motivo": f"imagen_invalida: {error}"})
            continue

        digest = _sha256(ruta)
        if digest in hashes_propios:
            descartadas.append({
                "archivo": fila["archivo"],
                "motivo": f"duplicada_dataset_propio:{hashes_propios[digest].name}",
            })
            continue
        if digest in hashes_foodsense:
            descartadas.append({
                "archivo": fila["archivo"],
                "motivo": f"duplicada_foodsense:{hashes_foodsense[digest].name}",
            })
            continue
        hashes_foodsense[digest] = ruta
        fila["sha256"] = digest
        validas.append(fila)
    return validas, descartadas


def _escribir_csv_atomico(ruta: Path, filas: list[dict], campos: list[str]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=ruta.parent, delete=False) as tmp:
        escritor = csv.DictWriter(tmp, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(filas)
        temporal = Path(tmp.name)
    os.replace(temporal, ruta)
    ruta.chmod(0o644)


def preparar(
    destino: Path = RUTA_DESTINO,
    min_votos: int = 3,
    min_fraccion: float = 0.12,
    modo: str = "completo",
    workers: int = 8,
) -> dict:
    destino = Path(destino)
    carpeta_cache = destino / ".cache"
    ruta_metadata = carpeta_cache / "metadata.csv"
    _descargar(_url_huggingface("metadata.csv"), ruta_metadata)

    candidatos = construir_candidatos(ruta_metadata, min_votos, min_fraccion)
    conteos_existentes = _leer_conteos_existentes(RUTA_DATASET_PROPIO_CSV)
    if modo == "balance":
        seleccionados, deficit = seleccionar_para_balance(candidatos, conteos_existentes)
    elif modo == "completo":
        seleccionados = candidatos
        deficit = {etiqueta: 0 for etiqueta in ETIQUETAS}
    else:
        raise ValueError(f"Modo desconocido: {modo}")

    carpeta_imagenes = destino / "imagenes"
    descargar_imagenes(seleccionados, carpeta_imagenes, workers)
    seleccionados, descartadas = _validar_y_deduplicar(seleccionados, carpeta_imagenes)
    nombres_seleccionados = {fila["archivo"] for fila in seleccionados}
    # La carpeta es una caché regenerable: se retiran descargas antiguas que
    # ya no pertenecen al manifiesto para que la auditoría no reporte huérfanos.
    for ruta in carpeta_imagenes.iterdir():
        if (
            ruta.is_file()
            and ruta.suffix.lower() in EXTENSIONES_VALIDAS
            and re.fullmatch(r"\d{4}_.+\.(?:jpg|jpeg|png|bmp|webp)", ruta.name, re.IGNORECASE)
            and ruta.name not in nombres_seleccionados
        ):
            ruta.unlink()

    filas_csv: list[dict] = []
    for candidato in seleccionados:
        fila = {
            "archivo": candidato["archivo"],
            **{etiqueta: int(etiqueta in candidato["sabores"]) for etiqueta in ETIQUETAS},
            "split": candidato["split"],
            "fuente": f"https://huggingface.co/datasets/{REPOSITORIO_HF}",
            "revision_fuente": REVISION_HF,
            "licencia": LICENCIA,
            "n_anotaciones_validas": candidato["n_anotaciones_validas"],
            "sha256": candidato["sha256"],
        }
        for etiqueta in ETIQUETAS:
            fila[f"votos_{etiqueta}"] = candidato[f"votos_{etiqueta}"]
            fila[f"fraccion_{etiqueta}"] = f"{candidato[f'fraccion_{etiqueta}']:.6f}"
        filas_csv.append(fila)

    campos = [
        "archivo", *ETIQUETAS, "split", "fuente", "revision_fuente", "licencia",
        "n_anotaciones_validas", "sha256",
        *(f"votos_{etiqueta}" for etiqueta in ETIQUETAS),
        *(f"fraccion_{etiqueta}" for etiqueta in ETIQUETAS),
    ]
    _escribir_csv_atomico(destino / "etiquetas.csv", filas_csv, list(campos))
    _escribir_csv_atomico(destino / "descartadas.csv", descartadas, ["archivo", "motivo"])

    conteos_nuevos = {
        etiqueta: sum(int(fila[etiqueta]) for fila in filas_csv)
        for etiqueta in ETIQUETAS
    }
    resumen = {
        "fuente": REPOSITORIO_HF,
        "revision": REVISION_HF,
        "licencia": LICENCIA,
        "modo": modo,
        "min_votos": min_votos,
        "min_fraccion": min_fraccion,
        "candidatos_con_consenso": len(candidatos),
        "imagenes_seleccionadas": len(filas_csv),
        "imagenes_descartadas": len(descartadas),
        "conteos_existentes": dict(conteos_existentes),
        "deficit_inicial": deficit,
        "conteos_agregados": conteos_nuevos,
        "conteos_por_split": dict(Counter(fila["split"] for fila in filas_csv)),
    }
    (destino / "resumen.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(resumen, ensure_ascii=False, indent=2))
    return resumen


def asegurar_preparado(destino: Path = RUTA_DESTINO, workers: int = 8) -> None:
    """Garantiza que el manifiesto y todas sus imágenes estén disponibles."""
    destino = Path(destino)
    ruta_csv = destino / "etiquetas.csv"
    if not ruta_csv.exists():
        preparar(destino=destino, workers=workers)
        return

    with open(ruta_csv, newline="", encoding="utf-8-sig") as archivo:
        filas = list(csv.DictReader(archivo))

    # Repara automáticamente descargas truncadas o alteradas.
    for fila in filas:
        ruta = destino / "imagenes" / fila["archivo"]
        digest_esperado = fila.get("sha256", "")
        if ruta.exists() and digest_esperado and _sha256(ruta) != digest_esperado:
            ruta.unlink()
    descargar_imagenes(filas, destino / "imagenes", workers)

    faltantes = [
        fila["archivo"] for fila in filas
        if not (destino / "imagenes" / fila["archivo"]).exists()
    ]
    if faltantes:
        raise RuntimeError(f"Faltan {len(faltantes)} imágenes FoodSense después de descargar.")
    alteradas = [
        fila["archivo"] for fila in filas
        if fila.get("sha256")
        and _sha256(destino / "imagenes" / fila["archivo"]) != fila["sha256"]
    ]
    if alteradas:
        raise RuntimeError(f"Falló la verificación SHA-256 de {len(alteradas)} imágenes FoodSense.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara el complemento humano FoodSense.")
    subparsers = parser.add_subparsers(dest="comando", required=True)

    preparar_parser = subparsers.add_parser("preparar", help="Cura y descarga FoodSense.")
    preparar_parser.add_argument("--destino", type=Path, default=RUTA_DESTINO)
    preparar_parser.add_argument("--min-votos", type=int, default=3)
    preparar_parser.add_argument("--min-fraccion", type=float, default=0.12)
    preparar_parser.add_argument("--modo", choices=("balance", "completo"), default="completo")
    preparar_parser.add_argument("--workers", type=int, default=8)

    asegurar_parser = subparsers.add_parser("asegurar", help="Descarga imágenes del manifiesto existente.")
    asegurar_parser.add_argument("--destino", type=Path, default=RUTA_DESTINO)
    asegurar_parser.add_argument("--workers", type=int, default=8)

    args = parser.parse_args()
    if args.comando == "preparar":
        preparar(args.destino, args.min_votos, args.min_fraccion, args.modo, args.workers)
    else:
        asegurar_preparado(args.destino, args.workers)


if __name__ == "__main__":
    main()
