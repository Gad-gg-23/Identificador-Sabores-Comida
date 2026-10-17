# Guía para el equipo — Identificador de Sabores de Comida

Repositorio: **[github.com/Gad-gg-23/Identificador-Sabores-Comida](https://github.com/Gad-gg-23/Identificador-Sabores-Comida)**

Notebook de Colab (ábrelo directo con este link, no hace falta buscarlo):
**[colab.research.google.com/.../Comida_Colab.ipynb](https://colab.research.google.com/github/Gad-gg-23/Identificador-Sabores-Comida/blob/main/colab/Comida_Colab.ipynb)**

---

## Parte 0 — Lo que hace el admin del repo, una sola vez

1. Entra a la página del repo en GitHub → **Settings** → **Collaborators** → **Add people** → escribe el usuario o correo de GitHub de cada compañero → **Add**.
2. Cada compañero recibe una invitación — **tiene que aceptarla** antes de poder subir cambios.
3. Les pasas esta guía y el link del notebook de arriba.

Nadie más necesita hacer este paso. El resto de la guía es para **cada compañero**.

---

## Parte 1 — La primera vez que abres el notebook (cada persona, una sola vez)

**No se necesita cuenta de Kaggle ni token de ningún tipo**: el dataset (Food-101) se descarga directo desde su fuente oficial con una sola línea, `torchvision` ya sabe hacerlo solo.

### 1.1 Abrir el notebook

1. Abre el link del notebook de arriba.
2. Inicia sesión con tu cuenta de Google si te lo pide.

### 1.2 Activar GPU

`Entorno de ejecución` (menú de arriba) → `Cambiar tipo de entorno de ejecución` → selecciona **GPU** (la T4 gratuita) → **Guardar**.

### 1.3 Correr todo el notebook

`Entorno de ejecución → Ejecutar todas` (o celda por celda con ▶️, de arriba hacia abajo).

Lo que vas a ver pasar, en orden:

1. **Clona el repo** — rápido, segundos.
2. **Verifica la GPU** — debe imprimir `GPU disponible: True`. Si dice `False`, te faltó el paso 1.2.
3. **Fotos desde Drive** — déjala tal cual (`USAR_DRIVE = False`), no la necesitas.
4. **Descarga Food-101 y entrena** — tiene DOS celdas:
   - La primera descarga Food-101 (**pesa ~5 GB — la primera vez tarda varios minutos**, vas a ver progreso). Las siguientes veces es más rápido porque ya queda en caché mientras la sesión siga abierta.
   - La segunda es la que **entrena de verdad**: va a imprimir el progreso época por época. Tarda varios minutos más (son ~100,000 imágenes). **Espera a que termine por completo** — al final debe decir `Modelo final en '.../salidas/modelo_comida.pt'`. Si avanzas a la siguiente celda antes de que termine, te va a salir un error de "archivo no encontrado".
5. **Ver resultados** — te muestra Accuracy exacta, F1 por ejemplo, F1 por sabor, la tabla de métricas por sabor, y las 5 matrices de confusión (una por sabor).
6. **Descargar o subir el modelo** (opcional) — ver Parte 4.
7. **Probar con tu propia foto** (opcional) — ver Parte 5.

---

## Parte 2 — Cómo subir fotos con su sabor (en cualquier momento, SIN necesidad de Colab)

Esto se hace desde el navegador, directo en GitHub — no toques Colab para este paso. **Cada foto necesita que alguien diga explícitamente qué sabores tiene** (no basta con solo subirla a una carpeta), porque el sabor no se puede adivinar solo del nombre del platillo.

**Antes de inventar los sabores a ojo, revisa [`comidas_platillos.csv`](comidas_platillos.csv)** (en la raíz del repo) — es la tabla oficial del equipo con los sabores ya decididos para ~280 platillos. Si tu platillo ya está ahí, copia esos mismos valores para que todos usemos el mismo criterio (así el modelo no recibe etiquetas contradictorias para el mismo platillo). Solo si tu platillo NO aparece en esa tabla, decide el sabor tú mismo con buen criterio (o pregúntale a quien lo preparó).

1. Entra al repo en GitHub, navega a `data/fotos_propias/imagenes/`.
2. Botón **Add file → Upload files** → arrastra tus fotos (jpg o png) → **Commit changes**.
3. Ahora ve a `data/fotos_propias/etiquetas.csv` → ícono de **lápiz (Edit)** → agrega una línea por cada foto que subiste, con este formato:

   ```
   archivo,acida,amarga,dulce,picosa,salada
   mole_poblano_1.jpg,0,1,1,0,0
   ceviche_camaron.jpg,1,0,0,1,1
   ```

   Es decir: el nombre EXACTO del archivo que subiste, y luego un `1` o `0` por cada sabor, en ese orden (ácida, amarga, dulce, picosa, salada). El `mole_poblano_1.jpg` de arriba dice "amarga=1, dulce=1, el resto 0" — ajústalo a lo que realmente pruebes.

4. Abajo, en "Commit changes" → **Commit changes directly to the `main` branch** → **Commit changes**.

Consejos:
- Entre más variadas las fotos (ángulos, luz, fondos, platillo cortado/entero), mejor generaliza el modelo.
- Si subes una foto y se te olvida agregarla al CSV (o al revés), el entrenamiento la va a ignorar y te va a avisar por consola cuál falta — no rompe nada, pero tampoco se usa.
- Si no estás seguro de un sabor, mejor pregúntale a alguien que sí probó el platillo que a adivinar — una etiqueta mal puesta enseña mal al modelo.

---

## Parte 3 — Cómo reentrenar después de que se suban fotos nuevas

**La regla simple y segura: vuelve a correr TODO el notebook desde el principio** (`Entorno de ejecución → Ejecutar todas`), cada vez que quieras entrenar con las fotos más recientes.

¿Por qué todo y no solo un par de celdas? Porque Colab reinicia la sesión seguido (por inactividad, o simplemente al volver otro día), y cuando eso pasa se borra todo lo que se había descargado — el repo clonado, el dataset, todo. Correr todo de nuevo siempre funciona, sin importar si la sesión era nueva o no.

Pasos:
1. Abre de nuevo el link del notebook (o sigue en la pestaña que ya tenías abierta).
2. `Entorno de ejecución → Ejecutar todas`.
3. Food-101 se vuelve a descargar (si la sesión es nueva) — espera esos minutos otra vez.
4. El `git pull` de la celda 1 trae automáticamente las fotos y el `etiquetas.csv` más reciente.
5. Al entrenar, el modelo ya las va a incluir.
6. Revisa la celda de resultados para ver si las métricas cambiaron.

---

## Parte 4 — Compartir el modelo entrenado con el equipo (opcional)

Al final del notebook hay dos formas de quedarte con el modelo que acabas de entrenar:

**Opción A — descargarlo a tu computadora:** corre esa celda y Colab te lo baja directo (`modelo_comida.pt`).

**Opción B — subirlo de vuelta al repo** (para que todo el equipo tenga el último modelo entrenado): esa celda te va a pedir:
- Tu **usuario de GitHub**.
- Un **Personal Access Token de GitHub** (NO tu contraseña). Para sacarlo:
  1. Ve a https://github.com/settings/tokens
  2. **Generate new token → Generate new token (classic)**.
  3. Ponle un nombre (ej. "Colab Comida") y una expiración (ej. 30 días).
  4. Marca la casilla **`repo`**.
  5. **Generate token** y cópialo — solo se muestra una vez.
  6. Pégalo en el campo de Colab (no se ve mientras escribes, es normal).

⚠️ Ese token es como una contraseña: no lo compartas ni lo pegues en chats. Si se expone, revócalo en esa misma página (botón "Delete") y genera uno nuevo.

---

## Parte 5 — Probar el modelo con una foto tuya

Al correr la última celda te aparece un botón **"Elegir archivos"** — selecciona una foto de un platillo desde tu computadora. Te va a imprimir algo como:

```
mi_mole.jpg: DULCE, PICOSA, AMARGA (acida=0.120, amarga=0.710, dulce=0.880, picosa=0.650, salada=0.340)
```

Puedes subir varias fotos a la vez y repetir la celda las veces que quieras.

---

## Problemas comunes

| Error | Causa | Solución |
|---|---|---|
| `GPU disponible: False` | No activaste GPU | `Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU` |
| `ValueError: Clase sin mapear a sabores` | Alguna categoría de Food-101 no está en `MAPA_SABORES` (config.py) | Agrégala a la tabla en `config.py` con los sabores que le correspondan |
| `[Errno 2] No such file or directory: '/content/repo'` | Colab reinició la sesión (VM nueva), se perdió todo lo descargado | `Entorno de ejecución → Ejecutar todas` desde el principio |
| El `git push` de la Parte 4 pide usuario/token y falla | Token mal copiado, expirado, o sin el permiso `repo` | Genera un token nuevo (Parte 4) y vuelve a intentar |
| Una foto propia no aparece en el entrenamiento | Está en `imagenes/` pero no tiene fila en `etiquetas.csv` (o viceversa) | Revisa el aviso que imprime la consola con el nombre exacto que falta |
