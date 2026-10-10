# Identificador de Sabores de Comida

Clasificador de imágenes **multi-etiqueta**: dado una foto de un platillo, el
modelo regresa cuáles de estos 5 sabores tiene (puede ser más de uno a la vez):

- Ácida
- Amarga
- Dulce
- Picosa
- Salada

Por ejemplo, una foto de mole debería regresar algo como `DULCE, PICOSA,
AMARGA`. Esto es justo lo que lo distingue de un clasificador normal: no
elige una sola categoría, decide sí/no para cada sabor por separado.

Para la guía paso a paso de cómo entrenar esto en equipo (Colab, subir
fotos propias, reentrenar), ver [GUIA_EQUIPO.md](GUIA_EQUIPO.md).

## Por qué multi-etiqueta y no un clasificador normal

Un clasificador "normal" elige **una** clase de varias posibles (softmax:
las probabilidades suman 1). Aquí cada sabor es una decisión independiente (sigmoid por salida,
entrenado con `BCEWithLogitsLoss`): el modelo puede decir "sí" a varios
sabores a la vez, porque así es la comida real.

## De dónde salen las etiquetas de sabor

Ningún dataset de fotos de comida viene etiquetado por sabor. Se usa
[Food-101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)
(101 categorías de platillos, ~100,000 fotos, descargado automáticamente
por `torchvision` — **no necesita cuenta de Kaggle**), y cada categoría
(ej. `mole`... bueno, Food-101 no tiene mole, pero sí `chicken_curry`,
`tacos`, `pad_thai`, etc.) se re-etiqueta por sabor según la tabla
`MAPA_SABORES` en [config.py](config.py).

**Esa tabla es un primer borrador** hecho con criterio culinario general
(no la validó un experto en sabores ni el profesor) — revísenla en equipo
antes de entrenar en serio, es la decisión que más va a afectar qué tan
bien funciona el modelo al final.

## Cómo se "aplasta el ruido con datos"

Dos fuentes de ruido, dos estrategias:

1. **Fotos de Food-101**: el dataset es ruidoso *a propósito* desde su
   creación original (fotos reales de usuarios, mal encuadradas, colores
   intensos, algunas mal etiquetadas) — con suficientes fotos por clase,
   ese ruido se promedia y no domina el aprendizaje.
2. **Fotos propias del equipo** (`data/fotos_propias/`): además se les
   aplica *data augmentation* agresivo en el entrenamiento (recortes,
   rotación, cambios de luz/color, perspectiva, blur, partes tapadas) para
   que el modelo no memorice una foto exacta sino el patrón visual real
   del platillo.

## Uso local

```bash
pip install -r requirements.txt
python Comida.py datos       # descarga/verifica Food-101 (~5 GB, tarda)
python Comida.py entrenar    # entrena y guarda el modelo
python Comida.py predecir --imagen ruta/foto.jpg
```

Entrenar localmente sin GPU es lento — lo normal es entrenar en Colab (ver
[GUIA_EQUIPO.md](GUIA_EQUIPO.md)) y opcionalmente descargar o traer de
vuelta el modelo ya entrenado.

## Estructura del proyecto

```
Comida.py           Punto de entrada (datos / entrenar / predecir)
config.py           Rutas, hiperparámetros, tabla platillo->sabores
dataset.py          Descarga Food-101, indexa fotos propias, augmentation
modelo.py           Transfer learning MobileNetV2, cabeza de 5 salidas
entrenar.py         Entrenamiento + métricas + matrices de confusión
predecir.py         Predicción sobre una imagen o carpeta
data/fotos_propias/ Fotos del equipo + etiquetas.csv (sabores a mano)
salidas/            Modelo entrenado, métricas, gráficas (se genera solo)
colab/              Notebook para entrenar con GPU gratuita de Colab
```
