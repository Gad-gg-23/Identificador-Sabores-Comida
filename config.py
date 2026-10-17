"""
Configuración central del identificador de sabores de comida.

Un solo lugar con rutas, hiperparámetros y la tabla de mapeo
clase-de-platillo -> sabores, para que entrenar.py, dataset.py y
predecir.py no se desincronicen.

Esto es multi-etiqueta, no clasificación normal: cada platillo puede tener
VARIOS sabores a la vez (el mole puede ser dulce + picoso + amargo al
mismo tiempo). El modelo no elige una sola clase, decide sí/no para cada
uno de los 5 sabores por separado.
"""

import os
from pathlib import Path
import torch

# =========================================================
# RUTAS
# =========================================================

RAIZ = Path(__file__).resolve().parent

RUTA_DATOS = RAIZ / "data"
RUTA_FOTOS_PROPIAS = RUTA_DATOS / "fotos_propias"  # fotos tomadas por el equipo (ruido real)
RUTA_IMAGENES_PROPIAS = RUTA_FOTOS_PROPIAS / "imagenes"
RUTA_ETIQUETAS_PROPIAS_CSV = RUTA_FOTOS_PROPIAS / "etiquetas.csv"

# Complemento externo curado con consenso humano. El manifiesto se guarda en
# Git; las imágenes se descargan de FoodSense cuando hacen falta.
RUTA_FOODSENSE = RUTA_DATOS / "foodsense_curado"
RUTA_IMAGENES_FOODSENSE = RUTA_FOODSENSE / "imagenes"
RUTA_ETIQUETAS_FOODSENSE_CSV = RUTA_FOODSENSE / "etiquetas.csv"

# Food-101 NO se guarda dentro del proyecto: torchvision lo descarga a su
# propia carpeta de caché (fuera del repo). Pesa ~5 GB comprimido / ~10 GB
# ya extraído.
RUTA_CACHE_FOOD101 = Path.home() / ".cache" / "comida_food101"

RUTA_SALIDAS = RAIZ / "salidas"
RUTA_MODELO = RUTA_SALIDAS / "modelo_comida.pt"
RUTA_HISTORICO_CSV = RUTA_SALIDAS / "historico_entrenamiento.csv"
RUTA_METRICAS_CSV = RUTA_SALIDAS / "resultados_comida.csv"
RUTA_METRICAS_POR_ETIQUETA_CSV = RUTA_SALIDAS / "metricas_por_etiqueta.csv"
RUTA_MATRICES_CONFUSION = RUTA_SALIDAS / "matrices_confusion.png"
RUTA_UMBRALES_JSON = RUTA_SALIDAS / "umbrales_por_sabor.json"
RUTA_BALANCE_JSON = RUTA_SALIDAS / "balance_entrenamiento.json"

# =========================================================
# ETIQUETAS: los 5 sabores (multi-etiqueta, no son mutuamente excluyentes)
# =========================================================

ETIQUETAS = ["acida", "amarga", "dulce", "picosa", "salada"]  # orden alfabético, fijo

# =========================================================
# MAPA_SABORES: clase-de-platillo (nombre de Food-101) -> set de sabores
# =========================================================
# Food-101 (101 categorías) NO trae etiquetas de sabor, solo el nombre del
# platillo. Esta tabla es el criterio con el que se re-etiquetan sus fotos
# por sabor en vez de por nombre de platillo. Es un PRIMER BORRADOR hecho
# con criterio culinario general (salsas, ingredientes típicos, método de
# preparación) — revísenla en equipo y ajústenla antes de entrenar en
# serio; es la decisión más importante de todo el proyecto.
#
# Si Food-101 trae alguna clase que no está aquí, etiquetas_multihot()
# lanza un error explícito (mejor fallar ruidosamente que etiquetar mal
# en silencio) indicando exactamente qué clase falta agregar.

MAPA_SABORES: dict[str, set[str]] = {
    "apple_pie": {"dulce", "acida"},
    "baby_back_ribs": {"salada", "dulce"},
    "baklava": {"dulce"},
    "beef_carpaccio": {"salada", "acida"},
    "beef_tartare": {"salada", "acida"},
    "beet_salad": {"dulce", "acida"},
    "beignets": {"dulce"},
    "bibimbap": {"salada", "picosa"},
    "bread_pudding": {"dulce"},
    "breakfast_burrito": {"salada", "picosa"},
    "bruschetta": {"salada", "acida"},
    "caesar_salad": {"salada", "acida"},
    "cannoli": {"dulce"},
    "caprese_salad": {"salada", "acida"},
    "carrot_cake": {"dulce"},
    "ceviche": {"acida", "salada", "picosa"},
    "cheesecake": {"dulce"},
    "cheese_plate": {"salada"},
    "chicken_curry": {"picosa", "salada"},
    "chicken_quesadilla": {"salada", "picosa"},
    "chicken_wings": {"picosa", "salada"},
    "chocolate_cake": {"dulce", "amarga"},
    "chocolate_mousse": {"dulce", "amarga"},
    "churros": {"dulce"},
    "clam_chowder": {"salada"},
    "club_sandwich": {"salada"},
    "crab_cakes": {"salada"},
    "creme_brulee": {"dulce", "amarga"},
    "croque_madame": {"salada"},
    "cup_cakes": {"dulce"},
    "deviled_eggs": {"salada", "acida"},
    "donuts": {"dulce"},
    "dumplings": {"salada", "picosa"},
    "edamame": {"salada"},
    "eggs_benedict": {"salada", "acida"},
    "escargots": {"salada"},
    "falafel": {"salada", "picosa"},
    "filet_mignon": {"salada"},
    "fish_and_chips": {"salada", "acida"},
    "foie_gras": {"salada", "dulce"},
    "french_fries": {"salada"},
    "french_onion_soup": {"salada", "dulce"},
    "french_toast": {"dulce"},
    "fried_calamari": {"salada", "acida"},
    "fried_rice": {"salada", "picosa"},
    "frozen_yogurt": {"dulce", "acida"},
    "garlic_bread": {"salada"},
    "gnocchi": {"salada"},
    "greek_salad": {"salada", "acida", "amarga"},
    "grilled_cheese_sandwich": {"salada"},
    "grilled_salmon": {"salada"},
    "guacamole": {"salada", "acida", "picosa"},
    "gyoza": {"salada", "picosa"},
    "hamburger": {"salada"},
    "hot_and_sour_soup": {"picosa", "acida"},
    "hot_dog": {"salada", "picosa"},
    "huevos_rancheros": {"salada", "picosa", "acida"},
    "hummus": {"salada", "acida"},
    "ice_cream": {"dulce"},
    "lasagna": {"salada", "acida"},
    "lobster_bisque": {"salada", "dulce"},
    "lobster_roll_sandwich": {"salada", "acida"},
    "macaroni_and_cheese": {"salada"},
    "macarons": {"dulce"},
    "miso_soup": {"salada"},
    "mussels": {"salada", "acida"},
    "nachos": {"salada", "picosa"},
    "omelette": {"salada"},
    "onion_rings": {"salada"},
    "oysters": {"salada", "acida"},
    "pad_thai": {"salada", "dulce", "picosa", "acida"},
    "paella": {"salada"},
    "pancakes": {"dulce"},
    "panna_cotta": {"dulce"},
    "peking_duck": {"salada", "dulce"},
    "pho": {"salada", "picosa", "acida"},
    "pizza": {"salada", "acida"},
    "pork_chop": {"salada"},
    "poutine": {"salada"},
    "prime_rib": {"salada"},
    "pulled_pork_sandwich": {"salada", "dulce"},
    "ramen": {"salada", "picosa"},
    "ravioli": {"salada", "acida"},
    "red_velvet_cake": {"dulce"},
    "risotto": {"salada"},
    "samosa": {"salada", "picosa"},
    "sashimi": {"salada", "acida"},
    "scallops": {"salada", "dulce"},
    "seaweed_salad": {"salada", "dulce", "acida"},
    "shrimp_and_grits": {"salada", "picosa"},
    "spaghetti_bolognese": {"salada", "acida"},
    "spaghetti_carbonara": {"salada"},
    "spring_rolls": {"salada", "dulce", "acida"},
    "steak": {"salada"},
    "strawberry_shortcake": {"dulce", "acida"},
    "sushi": {"salada", "acida"},
    "takoyaki": {"salada", "dulce"},
    "tacos": {"salada", "picosa", "acida"},
    "tiramisu": {"dulce", "amarga"},
    "tuna_tartare": {"salada", "acida"},
    "waffles": {"dulce"},
}


def normalizar_nombre_clase(nombre: str) -> str:
    """'Apple Pie', 'apple-pie' -> 'apple_pie' (formato de Food-101)."""
    return nombre.strip().lower().replace(" ", "_").replace("-", "_")


def etiquetas_multihot(nombre_clase: str) -> list[int]:
    """
    Traduce el nombre de una clase de Food-101 (o de una fila de
    data/fotos_propias/etiquetas.csv) a un vector multi-etiqueta de 0/1,
    uno por cada sabor en ETIQUETAS (orden fijo).

    Lanza un error explícito si la clase no está en MAPA_SABORES.
    """
    clave = normalizar_nombre_clase(nombre_clase)
    sabores = MAPA_SABORES.get(clave)
    if sabores is None:
        raise ValueError(
            f"Clase sin mapear a sabores: '{nombre_clase}' (normalizada: '{clave}'). "
            "Agrégala a MAPA_SABORES en config.py."
        )
    return [1 if etiqueta in sabores else 0 for etiqueta in ETIQUETAS]


# =========================================================
# HIPERPARÁMETROS
# =========================================================

TAMANO_IMAGEN = 224          # tamaño de entrada esperado por MobileNetV2
BATCH_SIZE = 32
NUM_EPOCAS = 15
TASA_APRENDIZAJE = 1e-3
PACIENCIA_EARLY_STOPPING = 4  # épocas sin mejorar F1 de validación antes de parar
RANDOM_STATE = 42
UMBRAL_DECISION = 0.5         # probabilidad mínima para marcar un sabor como presente

# Food-101 solo trae train/test oficiales; el val se separa del train.
FRACCION_VALIDACION = 0.1

# Fracción de las fotos propias que se reserva como prueba "de ruido real"
# (el resto se mezcla al entrenamiento). NO se estratifica (cada foto
# puede tener varios sabores a la vez, estratificar multi-etiqueta
# necesitaría una librería aparte) — se reparte aleatoriamente nomás.
FRACCION_PRUEBA_FOTOS_PROPIAS = 0.2

DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carga de datos en paralelo: en CPU local (Windows) la decodificación de
# imágenes y el augmentation corrían en un solo hilo y eran el cuello de
# botella real, no el modelo. En Colab (Linux) el fork de procesos es más
# barato, así que ahí se usan más workers.
NUM_WORKERS = min(4, os.cpu_count() or 2)
PIN_MEMORY = DISPOSITIVO.type == "cuda"
