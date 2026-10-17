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

La mayoría de los datasets de comida solo identifica el platillo. Como base se usa
[Food-101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)
(101 categorías de platillos, ~100,000 fotos, descargado automáticamente
por `torchvision` — **no necesita cuenta de Kaggle**), y cada categoría
(ej. `mole`... bueno, Food-101 no tiene mole, pero sí `chicken_curry`,
`tacos`, `pad_thai`, etc.) se re-etiqueta por sabor según la tabla
`MAPA_SABORES` en [config.py](config.py). Además, el complemento FoodSense
aporta descriptores de sabor que varios participantes infirieron visualmente;
son expectativas humanas basadas en la imagen, no una medición química ni una
degustación del platillo.

**Esa tabla es un primer borrador** hecho con criterio culinario general
(no la validó un experto en sabores ni el profesor) — revísenla en equipo
antes de entrenar en serio, es la decisión que más va a afectar qué tan
bien funciona el modelo al final.

Para las **fotos propias del equipo** (`data/fotos_propias/`), en cambio,
el sabor no se infiere de ningún nombre de categoría: se etiqueta a mano
siguiendo [`comidas_platillos.csv`](comidas_platillos.csv), la tabla
oficial del equipo (~300 platillos) — ver la Parte 2 de
[GUIA_EQUIPO.md](GUIA_EQUIPO.md).

## Cómo se "aplasta el ruido con datos"

Tres fuentes de datos, tres estrategias:

1. **Fotos de Food-101**: el dataset es ruidoso *a propósito* desde su
   creación original (fotos reales de usuarios, mal encuadradas, colores
   intensos, algunas mal etiquetadas) — con suficientes fotos por clase,
   ese ruido se promedia y no domina el aprendizaje.
2. **Fotos propias del equipo** (`data/fotos_propias/`): además se les
   aplica *data augmentation* agresivo en el entrenamiento (recortes,
   rotación, cambios de luz/color, perspectiva, blur, partes tapadas) para
   que el modelo no memorice una foto exacta sino el patrón visual real
   del platillo.
3. **FoodSense curado** (`data/foodsense_curado/`): complemento permanente
   con 1,711 imágenes y anotaciones humanas agregadas por consenso. Se conservan
   sus splits oficiales y se verifican licencia, integridad y duplicados. La
   comparación de otras fuentes está en
   [`DATASETS_EVALUADOS.md`](DATASETS_EVALUADOS.md).

El entrenamiento calcula automáticamente un peso independiente para cada sabor
con la fórmula `negativos / positivos`, usando únicamente `train`. Después
calibra un umbral por sabor en validación. Así se mejora el aprendizaje de
`amarga` sin duplicar archivos ni usar información del conjunto de prueba.

## Uso local

```bash
pip install -r requirements.txt
python Comida.py datos       # prepara los tres datasets; Food-101 pesa ~5 GB
python Comida.py foodsense   # regenera el complemento humano (normalmente no hace falta)
python Comida.py entrenar    # entrena y guarda el modelo
python Comida.py predecir --imagen ruta/foto.jpg
python auditar_datos.py --dataset todos
```

Entrenar localmente sin GPU es lento — lo normal es entrenar en Colab (ver
[GUIA_EQUIPO.md](GUIA_EQUIPO.md)) y opcionalmente descargar o traer de
vuelta el modelo ya entrenado.

## Estructura del proyecto

```
Comida.py           Punto de entrada (datos / foodsense / entrenar / predecir)
config.py           Rutas, hiperparámetros, tabla platillo->sabores
dataset.py          Combina Food-101, fotos propias y FoodSense; augmentation
modelo.py           Transfer learning MobileNetV2, cabeza de 5 salidas
entrenar.py         Entrenamiento + métricas + matrices de confusión
predecir.py         Predicción sobre una imagen o carpeta
data/fotos_propias/ Fotos del equipo + etiquetas.csv (sabores a mano)
data/foodsense_curado/ Manifiesto humano compatible; imágenes bajo demanda
foodsense.py         Curación, consenso, balance y descarga reproducible
balanceo.py           Pesos de clase y calibración de umbrales por sabor
auditar_datos.py     Integridad, balance, duplicados y archivos huérfanos
salidas/            Modelo entrenado, métricas, gráficas (se genera solo)
colab/              Notebook para entrenar con GPU gratuita de Colab
```
