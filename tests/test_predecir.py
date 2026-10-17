import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config
import predecir


class UmbralesPrediccionTest(unittest.TestCase):
    def test_modelos_anteriores_conservan_umbral_predeterminado(self):
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "inexistente.json"
            with patch.object(config, "RUTA_UMBRALES_JSON", ruta):
                umbrales = predecir.cargar_umbrales()

        self.assertEqual(
            umbrales,
            {etiqueta: config.UMBRAL_DECISION for etiqueta in config.ETIQUETAS},
        )

    def test_carga_umbrales_calibrados_en_el_orden_de_configuracion(self):
        esperados = {
            "acida": 0.25,
            "amarga": 0.15,
            "dulce": 0.60,
            "picosa": 0.40,
            "salada": 0.70,
        }
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "umbrales.json"
            ruta.write_text(
                json.dumps({"umbrales": esperados}), encoding="utf-8"
            )
            with patch.object(config, "RUTA_UMBRALES_JSON", ruta):
                obtenidos = predecir.cargar_umbrales()

        for etiqueta, esperado in esperados.items():
            self.assertAlmostEqual(obtenidos[etiqueta], esperado, places=6)


if __name__ == "__main__":
    unittest.main()
