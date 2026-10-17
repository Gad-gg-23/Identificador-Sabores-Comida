import unittest
from pathlib import Path
from unittest.mock import patch

import dataset


class IntegracionDatasetTest(unittest.TestCase):
    def test_foodsense_se_incorpora_siempre_y_respeta_sus_splits(self):
        food101_train = [
            (Path(f"{clase}_{indice}.jpg"), clase)
            for clase in ("apple_pie", "tacos")
            for indice in range(10)
        ]
        food101_test = [(Path("food101_test.jpg"), "apple_pie")]
        foodsense = {
            "train": [(Path("foodsense_train.jpg"), [1, 0, 0, 0, 0])],
            "validation": [(Path("foodsense_val.jpg"), [0, 0, 1, 0, 0])],
            "test": [(Path("foodsense_test.jpg"), [0, 0, 0, 1, 0])],
        }

        def cargar_food101(split):
            return food101_train if split == "train" else food101_test

        with (
            patch.object(dataset, "_cargar_split_food101", side_effect=cargar_food101),
            patch.object(dataset, "indexar_fotos_propias", return_value=[]),
            patch.object(dataset, "indexar_foodsense_curado", return_value=foodsense) as indexar,
        ):
            resultado = dataset.construir_splits()

        indexar.assert_called_once_with()
        self.assertIn(foodsense["train"][0], resultado["train"])
        self.assertIn(foodsense["validation"][0], resultado["val"])
        self.assertIn(foodsense["test"][0], resultado["test"])


if __name__ == "__main__":
    unittest.main()
