# Complemento FoodSense curado

Este directorio contiene un subconjunto de
[FoodSense](https://huggingface.co/datasets/sababishraq/foodsense-dataset),
publicado bajo licencia **CC BY 4.0**.

El manifiesto actual contiene **1,711 imágenes** con consenso.

Referencia: Ishraq, Sabab; Aarushi, Aarushi; Jiang, Juncai; Chen, Chen.
*FoodSense: A Multisensory Food Dataset and Benchmark for Predicting Taste,
Smell, Texture, and Sound from Images*. CVPR Workshops, 2026.

Las anotaciones originales son opiniones humanas en texto libre. El script
`foodsense.py` conserva todas las imágenes con consenso explícito en alguno de
los cinco sabores del proyecto. El desbalance que aún existe se corrige con
`pos_weight`, calculado exclusivamente con el split de entrenamiento. `savory`
no se interpreta como `salada`.

Estas anotaciones representan el sabor que las personas esperan al mirar la
foto; no son mediciones químicas ni resultados de una degustación directa.

- `etiquetas.csv`: manifiesto versionado y compatible con el entrenamiento.
- `resumen.json`: parámetros, procedencia y distribución del subconjunto.
- `descartadas.csv`: duplicados o imágenes inválidas detectadas.
- `imagenes/`: archivos descargados; no se guardan en Git.

Para regenerarlo de forma reproducible:

```bash
python Comida.py foodsense
python auditar_datos.py --dataset todos
```

El manifiesto fija la revisión upstream
`843b1ae42b18deddfb4452362ae1a06202ade1cb`. El entrenamiento incorpora este
complemento automáticamente y respeta los splits oficiales por imagen.
