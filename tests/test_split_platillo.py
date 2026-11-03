import unittest
from pathlib import Path

import config
import dataset
import entrenar


class SplitPorPlatilloTest(unittest.TestCase):
    def test_platillo_de_archivo_quita_numero_y_mayusculas(self):
        self.assertEqual(dataset.platillo_de_archivo(Path("Mole_Negro12.jpg")), "mole_negro")
        self.assertEqual(dataset.platillo_de_archivo(Path("mole_negro.jpg")), "mole_negro")
        self.assertEqual(dataset.platillo_de_archivo(Path("tacos_al_pastor_3.png")), "tacos_al_pastor")

    def test_ningun_platillo_queda_en_dos_splits(self):
        fotos = [
            # "platillo_7x3.jpg" -> platillo "platillo_7x", foto 3
            (Path(f"platillo_{p}x{i}.jpg"), [0, 0, 1, 0, 0])
            for p in range(40)
            for i in range(1, 6)
        ]
        train, val, test = dataset.repartir_por_platillo(fotos, 0.15, 0.2)

        def platillos(split):
            return {dataset.platillo_de_archivo(ruta) for ruta, _ in split}

        self.assertEqual(len(train) + len(val) + len(test), len(fotos))
        self.assertTrue(train and val and test)
        self.assertFalse(platillos(train) & platillos(val))
        self.assertFalse(platillos(train) & platillos(test))
        self.assertFalse(platillos(val) & platillos(test))

    def test_fuente_segun_carpeta(self):
        self.assertEqual(entrenar._fuente(config.RUTA_IMAGENES_PROPIAS / "pozole3.jpg"), "Fotos propias")
        self.assertEqual(entrenar._fuente(config.RUTA_IMAGENES_FOODSENSE / "x.jpg"), "FoodSense")
        self.assertEqual(entrenar._fuente(Path("/cache/food-101/images/tacos/1.jpg")), "Food-101")


if __name__ == "__main__":
    unittest.main()
