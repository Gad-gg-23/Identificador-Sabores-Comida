import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import foodsense


class FoodSenseTest(unittest.TestCase):
    def test_extrae_sabores_y_respeta_negaciones(self):
        self.assertEqual(
            foodsense.sabores_en_descriptor("sweet, tangy and spicy"),
            {"dulce", "acida", "picosa"},
        )
        self.assertEqual(foodsense.sabores_en_descriptor("not too sweet, no spice"), set())
        self.assertEqual(foodsense.sabores_en_descriptor("savory"), set())
        self.assertEqual(foodsense.sabores_en_descriptor("salty but not bitter"), {"salada"})

    def test_construye_una_fila_por_imagen_con_consenso(self):
        campos = ["file_name", "split", "CanInfer_taste", "taste_desc"]
        filas = [
            {"file_name": "a.jpg", "split": "train", "CanInfer_taste": "1", "taste_desc": "sweet"},
            {"file_name": "a.jpg", "split": "train", "CanInfer_taste": "1", "taste_desc": "sweet sugary"},
            {"file_name": "a.jpg", "split": "train", "CanInfer_taste": "1", "taste_desc": "sweet and salty"},
            {"file_name": "b.jpg", "split": "excluded", "CanInfer_taste": "1", "taste_desc": "bitter"},
        ]
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "metadata.csv"
            with open(ruta, "w", newline="", encoding="utf-8") as archivo:
                escritor = csv.DictWriter(archivo, fieldnames=campos)
                escritor.writeheader()
                escritor.writerows(filas)
            candidatos = foodsense.construir_candidatos(ruta, min_votos=2, min_fraccion=0.5)

        self.assertEqual(len(candidatos), 1)
        self.assertEqual(candidatos[0]["archivo"], "a.jpg")
        self.assertEqual(candidatos[0]["sabores"], {"dulce"})
        self.assertEqual(candidatos[0]["votos_dulce"], 3)

    def test_balance_prioriza_clases_con_deficit(self):
        def candidato(nombre, sabores):
            fila = {
                "archivo": nombre,
                "sabores": set(sabores),
                "n_anotaciones_validas": 10,
                "split": "train",
            }
            for etiqueta in foodsense.ETIQUETAS:
                fila[f"fraccion_{etiqueta}"] = 0.5 if etiqueta in sabores else 0.0
                fila[f"votos_{etiqueta}"] = 5 if etiqueta in sabores else 0
            return fila

        existentes = Counter({"acida": 8, "amarga": 1, "dulce": 5, "picosa": 10, "salada": 20})
        candidatos = [
            candidato("salada.jpg", {"salada"}),
            candidato("amarga.jpg", {"amarga"}),
            candidato("dulce.jpg", {"dulce"}),
        ]
        seleccionados, _ = foodsense.seleccionar_para_balance(candidatos, existentes)
        self.assertEqual({fila["archivo"] for fila in seleccionados}, {"amarga.jpg", "dulce.jpg"})


if __name__ == "__main__":
    unittest.main()
