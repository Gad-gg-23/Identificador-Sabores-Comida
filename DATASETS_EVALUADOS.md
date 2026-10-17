# Datasets evaluados para complementar sabores

La prioridad fue encontrar imágenes de comida con etiquetas explícitas de
`ácida`, `amarga`, `dulce`, `picosa` y `salada`, descarga reproducible y una
licencia compatible con un proyecto educativo. No se incorporaron fuentes solo
porque fueran grandes: inferir sabor desde una categoría o ingrediente agrega
ruido y puede empeorar la señal humana que se intenta mejorar.

| Fuente | Señal disponible | Decisión |
|---|---|---|
| [FoodSense](https://huggingface.co/datasets/sababishraq/foodsense-dataset) | 2,987 imágenes y 66,842 pares participante-imagen con descriptores sensoriales; CC BY 4.0 | **Integrado.** Se conservan 1,711 imágenes con consenso explícito y sus splits oficiales. |
| [Matsunaga et al.](https://www.vislab.is.i.nagoya-u.ac.jp/~murase/pdf/1276-pdf.pdf) / Rakuten | 1,827 recetas etiquetadas manualmente; incluye 213 ejemplos amargos | No integrado: el artículo es público, pero los datos proceden del acceso institucional de Rakuten/NII y no tienen descarga redistribuible directa. |
| [Yummly Food-100](https://vision.cornell.edu/se3/projects/cost-effective-hits/) | 100 imágenes y comparaciones humanas de similitud de sabor | No integrado: demasiado pequeño, no ofrece etiquetas binarias equivalentes y las imágenes se distribuyen bajo una declaración de *fair use*. |
| [Yummly Recipe Data](https://doi.org/10.7910/DVN/W0OB30) | Registro CC0 de recetas obtenidas mediante la antigua API | No integrado: la versión publicada actualmente lista cero archivos y su descripción limita el uso a investigación/educación. |
| [Yummly48K](https://arxiv.org/abs/1805.12081) y Yummly-66K | Imágenes, recetas y atributos de sabor derivados de Yummly | No integrado: no se encontró una descarga oficial vigente con imágenes, metadatos y licencia verificable; la API original fue retirada. |
| [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) | Imágenes, ingredientes, masas y nutrición; CC BY 4.0 | No integrado: no contiene etiquetas sensoriales. Convertir nutrientes/ingredientes a sabor sería una heurística, no verdad de terreno. |
| [VireoFood-172](https://github.com/PublicRelease/FoodRecog) | 110,241 imágenes y 353 ingredientes | No integrado por la misma razón: reconoce platillos e ingredientes, no sabor percibido. |
| [Fruits-360](https://github.com/fruits-360) y conjuntos agrícolas de café | Clases de frutas, vegetales o granos | No integrado: fondos de estudio y objetos aislados producirían un fuerte cambio de dominio respecto de platos reales; “café” o “toronja” tampoco garantizan una preparación amarga. |

## Estrategia adoptada

1. Usar todos los ejemplos FoodSense que superan tres votos y 12% de consenso.
2. Mantener intactos sus splits para evitar que una imagen aparezca en entrenamiento
   y evaluación.
3. Calcular [`pos_weight = negativos / positivos`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
   exclusivamente sobre `train`, siguiendo la formulación oficial de PyTorch
   para clasificación multi-etiqueta.
4. Calibrar un umbral diferente para cada sabor exclusivamente sobre validación.
5. Evaluar una sola vez sobre test con esos umbrales ya fijados.

Esta combinación amplía datos humanos sin introducir material de licencia dudosa
y aborda el desbalance que ningún dataset público accesible resuelve por sí solo.
