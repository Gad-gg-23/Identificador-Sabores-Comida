import unittest

import torch
import torch.nn as nn

import config
import modelo as modelo_mod


def _batchnorms(modelo, entrenables):
    return [
        capa for capa in modelo.modules()
        if isinstance(capa, nn.BatchNorm2d)
        and any(p.requires_grad for p in capa.parameters()) == entrenables
    ]


class ModoEntrenamientoTest(unittest.TestCase):
    def _pasar_lote(self, modelo):
        modelo_mod.modo_entrenamiento(modelo)
        with torch.no_grad():
            modelo(torch.randn(4, 3, 224, 224, device=next(modelo.parameters()).device))

    def test_backbone_congelado_conserva_estadisticas(self):
        modelo = modelo_mod.crear_modelo(preentrenado=False)
        congeladas = _batchnorms(modelo, entrenables=False)
        antes = [capa.running_mean.clone() for capa in congeladas]

        self._pasar_lote(modelo)

        self.assertTrue(modelo.training)
        self.assertTrue(modelo.classifier.training)
        for capa, media in zip(congeladas, antes):
            self.assertFalse(capa.training)
            self.assertTrue(torch.equal(capa.running_mean, media))

    def test_capas_descongeladas_si_se_actualizan(self):
        modelo = modelo_mod.crear_modelo(afinar_completo=True, preentrenado=False)
        entrenables = _batchnorms(modelo, entrenables=True)
        self.assertTrue(entrenables)
        antes = [capa.running_mean.clone() for capa in entrenables]

        self._pasar_lote(modelo)

        self.assertTrue(all(capa.training for capa in entrenables))
        self.assertTrue(any(
            not torch.equal(capa.running_mean, media) for capa, media in zip(entrenables, antes)
        ))


class GruposDeParametrosTest(unittest.TestCase):
    def test_backbone_descongelado_aprende_mas_lento_que_la_cabeza(self):
        modelo = modelo_mod.crear_modelo(afinar_completo=True, preentrenado=False)
        cabeza, backbone = modelo_mod.grupos_de_parametros(modelo)

        self.assertEqual(cabeza["lr"], config.TASA_APRENDIZAJE)
        self.assertEqual(backbone["lr"], config.TASA_APRENDIZAJE_BACKBONE)
        self.assertLess(backbone["lr"], cabeza["lr"])
        # Todos los parámetros entrenables quedan en exactamente un grupo.
        en_grupos = [id(p) for grupo in (cabeza, backbone) for p in grupo["params"]]
        entrenables = [id(p) for p in modelo.parameters() if p.requires_grad]
        self.assertCountEqual(en_grupos, entrenables)

    def test_sin_afinar_solo_entrena_la_cabeza(self):
        modelo = modelo_mod.crear_modelo(afinar_completo=False, preentrenado=False)
        grupos = modelo_mod.grupos_de_parametros(modelo)
        self.assertEqual(len(grupos), 1)

    def test_un_paso_de_optimizacion_mueve_el_backbone_descongelado(self):
        modelo = modelo_mod.crear_modelo(afinar_completo=True, preentrenado=False)
        optimizador = torch.optim.Adam(modelo_mod.grupos_de_parametros(modelo))
        ultima = modelo.features[-1][0].weight
        congelada = modelo.features[0][0].weight
        antes_ultima, antes_congelada = ultima.detach().clone(), congelada.detach().clone()

        modelo_mod.modo_entrenamiento(modelo)
        dispositivo = next(modelo.parameters()).device
        perdida = nn.BCEWithLogitsLoss()(
            modelo(torch.randn(4, 3, 224, 224, device=dispositivo)),
            torch.ones(4, len(config.ETIQUETAS), device=dispositivo),
        )
        perdida.backward()
        optimizador.step()

        self.assertFalse(torch.equal(ultima.detach(), antes_ultima))
        self.assertTrue(torch.equal(congelada.detach(), antes_congelada))


if __name__ == "__main__":
    unittest.main()
