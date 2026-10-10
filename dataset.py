"""
Construcción del dataset multi-etiqueta de sabores a partir de dos fuentes:

1. Food-101 (101 categorías de platillos, descargado por torchvision,
   SIN necesitar cuenta de Kaggle) -> cada imagen se re-etiqueta con el
   vector de sabores de config.etiquetas_multihot() según el nombre del
   platillo (no según su sabor real fotografiado, es un criterio por
   categoría).
2. data/fotos_propias/ -> fotos tomadas por el equipo con el "ruido" que
   pidió el profesor (platillos reales, fondos distintos, luz distinta),
   etiquetadas A MANO por sabor en etiquetas.csv (aquí sí hay alguien
   diciendo "esto sabe a X", no se infiere del nombre del platillo).

El resultado son 3 listas de (ruta_imagen, vector_multihot): train / val /
test, listas para envolver en ImagenesComida (Dataset de PyTorch).
"""

import csv
import random
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

import config

EXTENSIONES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================================================
# FOOD-101 (vía torchvision, descarga oficial sin Kaggle)
# =========================================================

def _cargar_split_food101(split: str) -> list[tuple[Path, str]]:
    """
    split: 'train' o 'test' (los únicos que trae Food-101 de fábrica).
    Descarga (o reutiliza el caché de) Food-101 y regresa una lista de
    (ruta_imagen, nombre_clase) usando los atributos internos que expone
    torchvision.datasets.Food101 (_image_files, _labels, classes).
    """
    from torchvision.datasets import Food101

    config.RUTA_CACHE_FOOD101.mkdir(parents=True, exist_ok=True)
    ds = Food101(root=str(config.RUTA_CACHE_FOOD101), split=split, download=True)

    rutas = ds._image_files
    indices_clase = ds._labels
    nombres_clase = ds.classes

    return [(Path(ruta), nombres_clase[idx]) for ruta, idx in zip(rutas, indices_clase)]


def _etiquetar(indice_nombres: list[tuple[Path, str]]) -> list[tuple[Path, list[int]]]:
    return [(ruta, config.etiquetas_multihot(nombre)) for ruta, nombre in indice_nombres]


# =========================================================
# FOTOS PROPIAS (etiquetadas a mano por el equipo en etiquetas.csv)
# =========================================================

def indexar_fotos_propias() -> list[tuple[Path, list[int]]]:
    """
    Lee data/fotos_propias/etiquetas.csv (columnas: archivo,acida,amarga,
    dulce,picosa,salada) y empareja cada fila con su imagen en
    data/fotos_propias/imagenes/<archivo>.

    Filas con un archivo que no existe, o cuyo archivo no está en el CSV,
    se avisan por consola y se ignoran (mejor avisar que fallar en medio
    del entrenamiento por un typo en el nombre del archivo).
    """
    if not config.RUTA_ETIQUETAS_PROPIAS_CSV.exists():
        return []

    indice = []
    with open(config.RUTA_ETIQUETAS_PROPIAS_CSV, newline="", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            archivo = fila.get("archivo", "").strip()
            if not archivo:
                continue
            ruta = config.RUTA_IMAGENES_PROPIAS / archivo
            if not ruta.exists():
                print(f"Aviso: '{archivo}' está en etiquetas.csv pero no existe en "
                      f"data/fotos_propias/imagenes/ — se ignora.")
                continue
            vector = [int(fila.get(etiqueta, 0) or 0) for etiqueta in config.ETIQUETAS]
            indice.append((ruta, vector))

    archivos_en_csv = {fila[0].name for fila in indice}
    for ruta_img in _listar_imagenes(config.RUTA_IMAGENES_PROPIAS):
        if ruta_img.name not in archivos_en_csv:
            print(f"Aviso: '{ruta_img.name}' está en data/fotos_propias/imagenes/ pero no "
                  f"tiene fila en etiquetas.csv — se ignora (agrégala con sus sabores).")

    return indice


def _listar_imagenes(carpeta: Path) -> list[Path]:
    if not carpeta.exists():
        return []
    return [p for p in carpeta.iterdir() if p.suffix.lower() in EXTENSIONES_VALIDAS]


# =========================================================
# CONSTRUCCIÓN DE LOS 3 SPLITS (train / val / test)
# =========================================================

def construir_splits() -> dict[str, list[tuple[Path, list[int]]]]:
    """
    - Food-101 solo trae train/test oficiales: el test se deja intacto
      (nunca se mezcla con train, para no inflar las métricas), y el
      train se reparte en train/val con train_test_split estratificado
      por NOMBRE DE PLATILLO (no por sabor: con multi-etiqueta no hay una
      sola clase para estratificar).
    - Las fotos propias (ruido real del equipo, etiquetadas a mano) se
      reparten aparte, sin estratificar, para medir específicamente qué
      tan bien generaliza al ruido real.
    """
    print("Descargando/verificando Food-101 (train)...")
    train_food101 = _cargar_split_food101("train")
    print("Descargando/verificando Food-101 (test)...")
    test_food101 = _cargar_split_food101("test")

    nombres_clase_train = [nombre for _, nombre in train_food101]
    train_idx, val_idx = train_test_split(
        range(len(train_food101)),
        test_size=config.FRACCION_VALIDACION,
        random_state=config.RANDOM_STATE,
        stratify=nombres_clase_train,
    )

    splits: dict[str, list[tuple[Path, list[int]]]] = {
        "train": _etiquetar([train_food101[i] for i in train_idx]),
        "val": _etiquetar([train_food101[i] for i in val_idx]),
        "test": _etiquetar(test_food101),
    }

    fotos_propias = indexar_fotos_propias()
    if fotos_propias:
        train_fp, test_fp = train_test_split(
            fotos_propias,
            test_size=config.FRACCION_PRUEBA_FOTOS_PROPIAS,
            random_state=config.RANDOM_STATE,
        )
        splits["train"].extend(train_fp)
        splits["test"].extend(test_fp)
        print(f"Fotos propias encontradas: {len(fotos_propias)} "
              f"({len(train_fp)} a train, {len(test_fp)} a test).")
    else:
        print("Aviso: no hay fotos propias etiquetadas en data/fotos_propias/ todavía. "
              "Se entrena solo con Food-101 (menos robusto al ruido real y a los sabores "
              "propios de la comida mexicana, poco representada en Food-101).")

    for clave, lista in splits.items():
        random.Random(config.RANDOM_STATE).shuffle(lista)
    return splits


# =========================================================
# TRANSFORMACIONES (aquí vive el "aplastar el ruido con datos")
# =========================================================
# ImageNet mean/std porque el modelo base (MobileNetV2) se preentrenó con eso.
_MEDIA_IMAGENET = [0.485, 0.456, 0.406]
_STD_IMAGENET = [0.229, 0.224, 0.225]


def transform_entrenamiento() -> transforms.Compose:
    """
    Data augmentation agresiva: simula los mismos tipos de 'ruido' que
    describió el profesor (ángulos distintos, luz distinta, platillo
    parcialmente fuera de cuadro, fotos borrosas de celular) para que el
    modelo generalice el sabor típico del platillo y no memorice el fondo
    o el plato exacto de una foto.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(config.TAMANO_IMAGEN, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(_MEDIA_IMAGENET, _STD_IMAGENET),
        # RandomErasing simula pedazos tapados/cortados; va después de ToTensor.
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def transform_evaluacion() -> transforms.Compose:
    """Sin augmentation: solo redimensionar y normalizar (val/test/predicción)."""
    return transforms.Compose([
        transforms.Resize((config.TAMANO_IMAGEN, config.TAMANO_IMAGEN)),
        transforms.ToTensor(),
        transforms.Normalize(_MEDIA_IMAGENET, _STD_IMAGENET),
    ])


# =========================================================
# DATASET DE PYTORCH
# =========================================================

class ImagenesComida(Dataset):
    """Envuelve una lista de (ruta_imagen, vector_multihot) para PyTorch."""

    def __init__(self, items: list[tuple[Path, list[int]]], transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        import torch

        ruta, vector = self.items[idx]
        imagen = Image.open(ruta).convert("RGB")
        imagen = self.transform(imagen)
        etiquetas = torch.tensor(vector, dtype=torch.float32)  # BCEWithLogitsLoss pide float
        return imagen, etiquetas
