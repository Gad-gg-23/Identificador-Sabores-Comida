import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import balanceo
import config
import dataset
import predecir


class ReglaAlMenosUnSaborTest(unittest.TestCase):
    def test_marca_el_mas_probable_si_ninguno_cruza_el_umbral(self):
        probabilidades = np.asarray([
            [0.10, 0.20, 0.30, 0.05, 0.40],  # ninguno cruza 0.5 -> salada
            [0.90, 0.10, 0.10, 0.10, 0.80],  # dos cruzan -> se quedan los dos
        ])
        prediccion = balanceo.aplicar_umbrales(probabilidades, np.full(5, 0.5))
        np.testing.assert_array_equal(prediccion, [[0, 0, 0, 0, 1], [1, 0, 0, 0, 1]])

    def test_prediccion_y_evaluacion_usan_la_misma_regla(self):
        class ModeloBajo(torch.nn.Module):
            def forward(self, x):  # todas las probabilidades < 0.5, la más alta es dulce
                return torch.tensor([[-3.0, -3.0, -1.0, -3.0, -2.0]], device=x.device)

        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "foto.jpg"
            Image.new("RGB", (300, 200), "orange").save(ruta)
            umbrales = {etiqueta: 0.5 for etiqueta in config.ETIQUETAS}
            resultado = predecir.predecir_imagen(ModeloBajo(), ruta, umbrales)

        self.assertEqual(resultado["etiquetas"], ["dulce"])


class PreprocesadoTest(unittest.TestCase):
    def test_evaluacion_recorta_sin_estirar(self):
        # Mitad izquierda roja, mitad derecha azul, foto 2:1. Con Resize((224,224))
        # se estiraba; con recorte central el borde queda en la columna 112.
        imagen = Image.new("RGB", (400, 200), "red")
        imagen.paste(Image.new("RGB", (200, 200), "blue"), (200, 0))
        tensor = dataset.transform_evaluacion()(imagen)

        self.assertEqual(tuple(tensor.shape), (3, config.TAMANO_IMAGEN, config.TAMANO_IMAGEN))
        rojo = tensor[0]
        self.assertGreater(rojo[:, 100].mean(), rojo[:, 124].mean())

    def test_cargar_imagen_respeta_orientacion_exif(self):
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "celular.jpg"
            exif = Image.Exif()
            exif[0x0112] = 6  # "rotar 90°", como guardan muchas fotos de celular
            Image.new("RGB", (300, 100), "green").save(ruta, exif=exif)

            self.assertEqual(dataset.cargar_imagen(ruta).size, (100, 300))


if __name__ == "__main__":
    unittest.main()
