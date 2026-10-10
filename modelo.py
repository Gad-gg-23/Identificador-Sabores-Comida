"""
Modelo: transfer learning con MobileNetV2 preentrenada en ImageNet.

Mismo criterio que Frutas_Verduras/modelo.py: se congela el backbone y
solo se entrena una cabeza nueva. La diferencia es que aquí la cabeza
tiene 5 salidas (una por sabor) SIN softmax -- cada salida es independiente
(sigmoid implícito en BCEWithLogitsLoss durante el entrenamiento, y
explícito en predecir.py), porque un platillo puede ser dulce Y picoso Y
amargo a la vez; softmax forzaría a que las probabilidades sumen 1 entre
sabores, que es justo lo que NO queremos.
"""

import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

import config


def crear_modelo(afinar_completo: bool = False) -> nn.Module:
    """
    afinar_completo=False (por defecto): solo se entrena la cabeza nueva,
        el resto de la red queda congelada (rápido, pocos datos necesarios).
    afinar_completo=True: además se descongelan las últimas capas
        convolucionales para un ajuste más fino (más lento, usar solo si
        ya hay bastantes fotos propias acumuladas).
    """
    modelo = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)

    for parametro in modelo.parameters():
        parametro.requires_grad = False

    # Cabeza original: Sequential(Dropout, Linear(1280, 1000)) -> la cambiamos
    # por 5 salidas independientes (una por sabor), sin activación final.
    num_caracteristicas = modelo.classifier[1].in_features
    modelo.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_caracteristicas, len(config.ETIQUETAS)),
    )

    if afinar_completo:
        # Descongela el último bloque de 'features' para ajuste fino.
        for parametro in modelo.features[-3:].parameters():
            parametro.requires_grad = True

    return modelo.to(config.DISPOSITIVO)
