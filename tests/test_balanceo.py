import unittest
from pathlib import Path

import numpy as np

import balanceo


class BalanceoTest(unittest.TestCase):
    def test_pos_weight_usa_negativos_entre_positivos(self):
        items = [
            (Path("a.jpg"), [1, 0]),
            (Path("b.jpg"), [0, 1]),
            (Path("c.jpg"), [0, 1]),
            (Path("d.jpg"), [0, 0]),
        ]
        pesos, reporte = balanceo.estadisticas_y_pos_weight(items, ["rara", "comun"])

        np.testing.assert_allclose(pesos.numpy(), [3.0, 1.0])
        self.assertEqual(reporte["positivos"], {"rara": 1, "comun": 2})
        self.assertEqual(reporte["imagenes_train"], 4)

    def test_umbrales_se_calibran_por_etiqueta(self):
        reales = np.asarray([
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
        ])
        probabilidades = np.asarray([
            [0.40, 0.20],
            [0.45, 0.10],
            [0.35, 0.80],
            [0.30, 0.70],
        ])

        umbrales = balanceo.optimizar_umbrales(
            reales, probabilidades, ["rara", "comun"]
        )

        self.assertGreater(umbrales["rara"], 0.35)
        self.assertLessEqual(umbrales["rara"], 0.40)
        self.assertGreater(umbrales["comun"], 0.20)
        self.assertLessEqual(umbrales["comun"], 0.70)


if __name__ == "__main__":
    unittest.main()
