"""Interfaz gráfica local para identificar sabores en una fotografía.

La ventana reutiliza la misma carga de modelo, transformaciones y umbrales de
``predecir.py``. El modelo se carga una sola vez en segundo plano para que la
interfaz permanezca responsiva incluso cuando PyTorch tarda en iniciar.

Uso directo::

    python interfaz.py

O mediante el punto de entrada del proyecto::

    python Comida.py interfaz
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk, UnidentifiedImageError


EXTENSIONES_INTERFAZ = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SABORES_UI = {
    "acida": ("Ácida", "#D79A2B"),
    "amarga": ("Amarga", "#7763A8"),
    "dulce": ("Dulce", "#D86C82"),
    "picosa": ("Picosa", "#D9583B"),
    "salada": ("Salada", "#3F88A6"),
}

FONDO = "#F3F0E8"
SUPERFICIE = "#FFFEFA"
TEXTO = "#20231F"
TEXTO_SUAVE = "#696D65"
BORDE = "#DAD7CE"
VERDE = "#386C55"
VERDE_CLARO = "#E5EFE9"
ROJO = "#A4473D"


def validar_archivo_imagen(ruta: str | Path) -> Path:
    """Valida existencia, extensión y contenido antes de inferir."""
    ruta = Path(ruta).expanduser()
    if not ruta.is_file():
        raise ValueError("El archivo seleccionado no existe o no es un archivo regular.")
    if ruta.suffix.lower() not in EXTENSIONES_INTERFAZ:
        raise ValueError("Formato no compatible. Usa JPG, JPEG, PNG, BMP o WEBP.")
    try:
        with Image.open(ruta) as imagen:
            imagen.verify()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("El archivo no contiene una imagen válida o está dañado.") from error
    return ruta.resolve()


def nombre_sabor(etiqueta: str) -> str:
    """Nombre presentable sin cambiar las claves que usa el modelo."""
    return SABORES_UI.get(etiqueta, (etiqueta.capitalize(), "#777777"))[0]


def resumen_sabores(etiquetas: list[str]) -> str:
    return " · ".join(nombre_sabor(etiqueta) for etiqueta in etiquetas)


class AplicacionSabores:
    """Ventana principal y coordinación segura entre Tk y PyTorch."""

    def __init__(self, raiz: tk.Tk):
        self.raiz = raiz
        self.raiz.title("Identificador de sabores")
        self.raiz.configure(bg=FONDO)
        self.raiz.minsize(900, 620)
        self._centrar_ventana(1040, 700)

        self.ruta_imagen: Path | None = None
        self.imagen_tk: ImageTk.PhotoImage | None = None
        self.modelo = None
        self.umbrales: dict[str, float] | None = None
        self.modulo_prediccion = None
        self.modelo_listo = False
        self.ocupado = False
        self.cerrada = False
        self.eventos: queue.Queue[tuple[str, object]] = queue.Queue()
        self.filas_resultado: dict[str, dict[str, object]] = {}

        self._configurar_estilos()
        self._construir_interfaz()
        self.raiz.protocol("WM_DELETE_WINDOW", self._cerrar)
        self.raiz.after(100, self._procesar_eventos)
        self._iniciar_carga_modelo()

    def _centrar_ventana(self, ancho: int, alto: int) -> None:
        pantalla_x = self.raiz.winfo_screenwidth()
        pantalla_y = self.raiz.winfo_screenheight()
        x = max(0, (pantalla_x - ancho) // 2)
        y = max(0, (pantalla_y - alto) // 2)
        self.raiz.geometry(f"{ancho}x{alto}+{x}+{y}")

    def _configurar_estilos(self) -> None:
        estilo = ttk.Style(self.raiz)
        estilo.theme_use("clam")
        estilo.configure(
            "Primario.TButton",
            background=VERDE,
            foreground="white",
            borderwidth=0,
            padding=(20, 12),
            font=("Helvetica", 11, "bold"),
        )
        estilo.map(
            "Primario.TButton",
            background=[("disabled", "#AEB9B2"), ("active", "#2C5A45")],
            foreground=[("disabled", "#F3F3F0")],
        )
        estilo.configure(
            "Secundario.TButton",
            background=SUPERFICIE,
            foreground=TEXTO,
            bordercolor=BORDE,
            borderwidth=1,
            padding=(18, 11),
            font=("Helvetica", 11, "bold"),
        )
        estilo.map("Secundario.TButton", background=[("active", "#ECE9E0")])

    def _construir_interfaz(self) -> None:
        contenedor = tk.Frame(self.raiz, bg=FONDO, padx=38, pady=30)
        contenedor.pack(fill="both", expand=True)
        contenedor.grid_columnconfigure(0, weight=3, uniform="panel")
        contenedor.grid_columnconfigure(1, weight=2, uniform="panel")
        contenedor.grid_rowconfigure(1, weight=1)

        encabezado = tk.Frame(contenedor, bg=FONDO)
        encabezado.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 22))
        tk.Label(
            encabezado,
            text="IDENTIFICADOR MULTIETIQUETA",
            bg=FONDO,
            fg=VERDE,
            font=("Helvetica", 10, "bold"),
        ).pack(anchor="w")
        tk.Label(
            encabezado,
            text="¿Qué sabores tiene este platillo?",
            bg=FONDO,
            fg=TEXTO,
            font=("Helvetica", 28, "bold"),
        ).pack(anchor="w", pady=(4, 3))
        tk.Label(
            encabezado,
            text="Selecciona una imagen del platillo",
            bg=FONDO,
            fg=TEXTO_SUAVE,
            font=("Helvetica", 12),
        ).pack(anchor="w")

        self._construir_panel_imagen(contenedor)
        self._construir_panel_resultados(contenedor)
        self._construir_barra_estado(contenedor)

    def _construir_barra_estado(self, padre: tk.Frame) -> None:
        barra = tk.Frame(padre, bg=FONDO)
        barra.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(16, 0))
        self.indicador_estado = tk.Label(
            barra,
            text="●",
            bg=FONDO,
            fg=TEXTO_SUAVE,
            font=("Helvetica", 10),
        )
        self.indicador_estado.pack(side="left")
        self.etiqueta_estado = tk.Label(
            barra,
            text="Cargando el modelo…",
            bg=FONDO,
            fg=TEXTO_SUAVE,
            font=("Helvetica", 10),
            anchor="w",
        )
        self.etiqueta_estado.pack(side="left", padx=(6, 0))

    def _crear_tarjeta(self, padre: tk.Widget) -> tk.Frame:
        return tk.Frame(
            padre,
            bg=SUPERFICIE,
            highlightbackground=BORDE,
            highlightthickness=1,
            bd=0,
        )

    def _construir_panel_imagen(self, padre: tk.Frame) -> None:
        tarjeta = self._crear_tarjeta(padre)
        tarjeta.grid(row=1, column=0, sticky="nsew", padx=(0, 11))
        tarjeta.grid_rowconfigure(0, weight=1)
        tarjeta.grid_columnconfigure(0, weight=1)

        self.marco_vista = tk.Frame(tarjeta, bg="#E8E5DC", padx=18, pady=18)
        self.marco_vista.grid(row=0, column=0, sticky="nsew", padx=20, pady=(20, 14))
        self.marco_vista.grid_rowconfigure(0, weight=1)
        self.marco_vista.grid_columnconfigure(0, weight=1)

        self.etiqueta_vista = tk.Label(
            self.marco_vista,
            text="＋\n\nSelecciona una imagen del platillo",
            bg="#E8E5DC",
            fg=TEXTO_SUAVE,
            font=("Helvetica", 14),
            justify="center",
        )
        self.etiqueta_vista.grid(row=0, column=0, sticky="nsew")

        self.etiqueta_archivo = tk.Label(
            tarjeta,
            text="Ningún archivo seleccionado",
            bg=SUPERFICIE,
            fg=TEXTO_SUAVE,
            font=("Helvetica", 10),
            anchor="w",
        )
        self.etiqueta_archivo.grid(row=1, column=0, sticky="ew", padx=22)

        acciones = tk.Frame(tarjeta, bg=SUPERFICIE)
        acciones.grid(row=2, column=0, sticky="ew", padx=20, pady=(13, 20))
        acciones.grid_columnconfigure(0, weight=1)
        acciones.grid_columnconfigure(1, weight=1)

        self.boton_seleccionar = ttk.Button(
            acciones,
            text="Elegir imagen",
            style="Secundario.TButton",
            command=self._seleccionar_imagen,
        )
        self.boton_seleccionar.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.boton_analizar = ttk.Button(
            acciones,
            text="Analizar sabores",
            style="Primario.TButton",
            command=self._analizar,
            state="disabled",
        )
        self.boton_analizar.grid(row=0, column=1, sticky="ew", padx=(6, 0))

    def _construir_panel_resultados(self, padre: tk.Frame) -> None:
        tarjeta = self._crear_tarjeta(padre)
        tarjeta.grid(row=1, column=1, sticky="nsew", padx=(11, 0))
        tarjeta.grid_columnconfigure(0, weight=1)

        tk.Label(
            tarjeta,
            text="RESULTADO",
            bg=SUPERFICIE,
            fg=TEXTO_SUAVE,
            font=("Helvetica", 9, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=24, pady=(23, 5))
        self.etiqueta_resumen = tk.Label(
            tarjeta,
            text="Esperando una imagen",
            bg=SUPERFICIE,
            fg=TEXTO,
            font=("Helvetica", 18, "bold"),
            anchor="w",
            justify="left",
            wraplength=300,
        )
        self.etiqueta_resumen.grid(row=1, column=0, sticky="ew", padx=24, pady=(0, 7))
        self.etiqueta_explicacion = tk.Label(
            tarjeta,
            text="Las barras mostrarán la probabilidad estimada por sabor.",
            bg=SUPERFICIE,
            fg=TEXTO_SUAVE,
            font=("Helvetica", 10),
            anchor="w",
            justify="left",
            wraplength=300,
        )
        self.etiqueta_explicacion.grid(row=2, column=0, sticky="ew", padx=24, pady=(0, 16))

        separador = tk.Frame(tarjeta, height=1, bg=BORDE)
        separador.grid(row=3, column=0, sticky="ew", padx=24)

        filas = tk.Frame(tarjeta, bg=SUPERFICIE)
        filas.grid(row=4, column=0, sticky="nsew", padx=24, pady=(12, 16))
        filas.grid_columnconfigure(0, weight=1)
        tarjeta.grid_rowconfigure(4, weight=1)

        for indice, (clave, (titulo, color)) in enumerate(SABORES_UI.items()):
            fila = tk.Frame(filas, bg=SUPERFICIE)
            fila.grid(row=indice, column=0, sticky="ew", pady=6)
            fila.grid_columnconfigure(0, weight=1)

            cabecera = tk.Frame(fila, bg=SUPERFICIE)
            cabecera.grid(row=0, column=0, sticky="ew")
            tk.Label(
                cabecera,
                text=titulo,
                bg=SUPERFICIE,
                fg=TEXTO,
                font=("Helvetica", 11, "bold"),
            ).pack(side="left")
            valor = tk.Label(
                cabecera,
                text="—",
                bg=SUPERFICIE,
                fg=TEXTO_SUAVE,
                font=("Helvetica", 10, "bold"),
            )
            valor.pack(side="right")

            barra = tk.Canvas(
                fila,
                height=9,
                bg="#E8E6DF",
                highlightthickness=0,
                bd=0,
            )
            barra.grid(row=1, column=0, sticky="ew", pady=(6, 3))
            estado = tk.Label(
                fila,
                text="Sin analizar",
                bg=SUPERFICIE,
                fg=TEXTO_SUAVE,
                font=("Helvetica", 8),
                anchor="w",
            )
            estado.grid(row=2, column=0, sticky="w")
            self.filas_resultado[clave] = {
                "barra": barra,
                "valor": valor,
                "estado": estado,
                "color": color,
            }



    def _iniciar_carga_modelo(self) -> None:
        self.modelo_listo = False
        self._actualizar_estado("Cargando el modelo…", "#C49135")
        threading.Thread(target=self._cargar_modelo, daemon=True).start()

    def _cargar_modelo(self) -> None:
        try:
            import predecir

            modelo = predecir.cargar_modelo()
            umbrales = predecir.cargar_umbrales()
            self.eventos.put(("modelo_listo", (predecir, modelo, umbrales)))
        except Exception as error:  # se presenta de forma legible en la ventana
            self.eventos.put(("error_modelo", error))

    def _seleccionar_imagen(self) -> None:
        seleccion = filedialog.askopenfilename(
            parent=self.raiz,
            title="Selecciona una fotografía de comida",
            filetypes=[
                ("Imágenes compatibles", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if not seleccion:
            return
        try:
            ruta = validar_archivo_imagen(seleccion)
            self._mostrar_imagen(ruta)
        except ValueError as error:
            messagebox.showerror("Imagen no válida", str(error), parent=self.raiz)

    def _mostrar_imagen(self, ruta: Path) -> None:
        with Image.open(ruta) as original:
            imagen = ImageOps.exif_transpose(original).convert("RGB")
            imagen.thumbnail((520, 390), Image.Resampling.LANCZOS)
            self.imagen_tk = ImageTk.PhotoImage(imagen)

        self.ruta_imagen = ruta
        self.etiqueta_vista.configure(image=self.imagen_tk, text="", bg="#E8E5DC")
        self.etiqueta_archivo.configure(text=f"{ruta.name}  ·  {self._tamano_legible(ruta.stat().st_size)}")
        self.etiqueta_resumen.configure(text="Imagen lista para analizar")
        self.etiqueta_explicacion.configure(
            text="Pulsa “Analizar sabores” para ejecutar el modelo sobre esta fotografía."
        )
        self._limpiar_barras()
        self._actualizar_boton_analizar()

    def _analizar(self) -> None:
        if not self.ruta_imagen or not self.modelo_listo or self.ocupado:
            return
        self.ocupado = True
        self._actualizar_boton_analizar()
        self.boton_seleccionar.configure(state="disabled")
        self.etiqueta_resumen.configure(text="Analizando imagen…")
        self.etiqueta_explicacion.configure(text="El modelo está calculando cada sabor por separado.")
        self._actualizar_estado("Ejecutando la predicción…", "#C49135")
        threading.Thread(target=self._predecir, args=(self.ruta_imagen,), daemon=True).start()

    def _predecir(self, ruta: Path) -> None:
        try:
            resultado = self.modulo_prediccion.predecir_imagen(
                self.modelo, ruta, self.umbrales
            )
            self.eventos.put(("prediccion_lista", resultado))
        except Exception as error:
            self.eventos.put(("error_prediccion", error))

    def _procesar_eventos(self) -> None:
        try:
            while True:
                tipo, datos = self.eventos.get_nowait()
                if tipo == "modelo_listo":
                    self.modulo_prediccion, self.modelo, self.umbrales = datos
                    self.modelo_listo = True
                    self._actualizar_boton_analizar()
                    self._actualizar_estado("Modelo listo", VERDE)
                elif tipo == "error_modelo":
                    self.modelo_listo = False
                    self._actualizar_estado("No fue posible cargar el modelo", ROJO)
                    messagebox.showerror(
                        "No se pudo cargar el modelo",
                        self._mensaje_error_modelo(datos),
                        parent=self.raiz,
                    )
                elif tipo == "prediccion_lista":
                    self.ocupado = False
                    self.boton_seleccionar.configure(state="normal")
                    self._mostrar_resultado(datos)
                    self._actualizar_boton_analizar()
                    self._actualizar_estado("Análisis terminado", VERDE)
                elif tipo == "error_prediccion":
                    self.ocupado = False
                    self.boton_seleccionar.configure(state="normal")
                    self._actualizar_boton_analizar()
                    self.etiqueta_resumen.configure(text="No se pudo analizar la imagen")
                    self._actualizar_estado("Error durante la predicción", ROJO)
                    messagebox.showerror(
                        "Error de predicción",
                        f"No fue posible analizar la imagen.\n\nDetalle: {datos}",
                        parent=self.raiz,
                    )
        except queue.Empty:
            pass
        if not self.cerrada:
            self.raiz.after(100, self._procesar_eventos)

    def _mostrar_resultado(self, resultado: dict) -> None:
        etiquetas = resultado["etiquetas"]
        probabilidades = resultado["probabilidades"]
        umbrales = resultado["umbrales"]
        self.etiqueta_resumen.configure(text=resumen_sabores(etiquetas))
        self.etiqueta_explicacion.configure(
            text="Un sabor se marca cuando su probabilidad alcanza el umbral de decisión."
        )

        for clave, widgets in self.filas_resultado.items():
            probabilidad = float(probabilidades.get(clave, 0.0))
            umbral = float(umbrales.get(clave, 0.5))
            detectado = clave in etiquetas
            barra: tk.Canvas = widgets["barra"]
            barra.update_idletasks()
            ancho = max(1, barra.winfo_width())
            barra.delete("all")
            barra.create_rectangle(0, 0, ancho, 9, fill="#E8E6DF", outline="")
            barra.create_rectangle(
                0,
                0,
                ancho * max(0.0, min(1.0, probabilidad)),
                9,
                fill=widgets["color"],
                outline="",
            )
            posicion_umbral = ancho * max(0.0, min(1.0, umbral))
            barra.create_line(posicion_umbral, 0, posicion_umbral, 9, fill=TEXTO, width=2)
            widgets["valor"].configure(text=f"{probabilidad:.0%}", fg=TEXTO)
            widgets["estado"].configure(
                text=("DETECTADO" if detectado else f"Umbral {umbral:.0%}"),
                fg=(VERDE if detectado else TEXTO_SUAVE),
            )

    def _limpiar_barras(self) -> None:
        for widgets in self.filas_resultado.values():
            widgets["barra"].delete("all")
            widgets["valor"].configure(text="—", fg=TEXTO_SUAVE)
            widgets["estado"].configure(text="Sin analizar", fg=TEXTO_SUAVE)

    def _actualizar_boton_analizar(self) -> None:
        estado = "normal" if self.ruta_imagen and self.modelo_listo and not self.ocupado else "disabled"
        self.boton_analizar.configure(state=estado)

    def _actualizar_estado(self, texto: str, color: str) -> None:
        self.indicador_estado.configure(fg=color)
        self.etiqueta_estado.configure(text=texto)

    @staticmethod
    def _tamano_legible(bytes_archivo: int) -> str:
        if bytes_archivo < 1024 * 1024:
            return f"{bytes_archivo / 1024:.0f} KB"
        return f"{bytes_archivo / (1024 * 1024):.1f} MB"

    @staticmethod
    def _mensaje_error_modelo(error: object) -> str:
        detalle = str(error)
        if isinstance(error, ModuleNotFoundError):
            return (
                "Faltan dependencias de Python. Instálalas con:\n\n"
                "pip install -r requirements.txt\n\n"
                f"Detalle: {detalle}"
            )
        if isinstance(error, FileNotFoundError):
            return f"{detalle}\n\nEntrena el modelo o copia el checkpoint a la carpeta salidas/."
        return f"El modelo existe, pero no pudo cargarse.\n\nDetalle: {detalle}"

    def _cerrar(self) -> None:
        self.cerrada = True
        self.raiz.destroy()


def main() -> None:
    try:
        raiz = tk.Tk()
    except tk.TclError as error:
        raise SystemExit(
            "No se pudo abrir una ventana gráfica. Ejecuta este comando en una "
            "sesión de escritorio con Tk disponible."
        ) from error
    AplicacionSabores(raiz)
    raiz.mainloop()


if __name__ == "__main__":
    main()
