import tempfile
import unittest
from pathlib import Path

from PIL import Image

import interfaz


class InterfazTest(unittest.TestCase):
    def test_valida_una_imagen_real(self):
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "platillo.jpg"
            Image.new("RGB", (20, 20), "orange").save(ruta)

            resultado = interfaz.validar_archivo_imagen(ruta)

        self.assertEqual(resultado.name, "platillo.jpg")
        self.assertTrue(resultado.is_absolute())

    def test_rechaza_archivo_danado_aunque_tenga_extension_de_imagen(self):
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "platillo.jpg"
            ruta.write_text("esto no es una imagen", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "dañado"):
                interfaz.validar_archivo_imagen(ruta)

    def test_presenta_las_etiquetas_sin_cambiar_las_claves_del_modelo(self):
        self.assertEqual(
            interfaz.resumen_sabores(["dulce", "picosa", "amarga"]),
            "Dulce · Picosa · Amarga",
        )


if __name__ == "__main__":
    unittest.main()
