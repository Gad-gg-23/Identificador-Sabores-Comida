"""
Modelo: transfer learning con MobileNetV2 preentrenada en ImageNet.

Se congela casi todo el backbone (las capas convolucionales que ya saben
extraer bordes/texturas/formas) y se entrena una cabeza nueva; con
config.AFINAR_COMPLETO también se ajustan las últimas capas. La cabeza
tiene 5 salidas (una por sabor) SIN softmax -- cada salida es independiente
(sigmoid implícito en BCEWithLogitsLoss durante el entrenamiento, y
explícito en predecir.py), porque un platillo puede ser dulce Y picoso Y
amargo a la vez; softmax forzaría a que las probabilidades sumen 1 entre
sabores, que es justo lo que NO queremos.
"""

import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

import config


def crear_modelo(afinar_completo: bool = False, preentrenado: bool = True) -> nn.Module:
    """
    afinar_completo=False (por defecto): solo se entrena la cabeza nueva,
        el resto de la red queda congelada (rápido, pocos datos necesarios).
    afinar_completo=True: además se descongelan las últimas capas
        convolucionales para un ajuste más fino (más lento, usar solo si
        ya hay bastantes fotos propias acumuladas).
    preentrenado=False: construye la arquitectura sin descargar ImageNet;
        se usa al cargar un checkpoint local que ya contiene todos los pesos.
    """
    pesos = MobileNet_V2_Weights.IMAGENET1K_V2 if preentrenado else None
    modelo = mobilenet_v2(weights=pesos)

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


def grupos_de_parametros(modelo: nn.Module) -> list[dict]:
    """
    Grupos para el optimizador: la cabeza nueva aprende con TASA_APRENDIZAJE y
    las capas descongeladas del backbone con TASA_APRENDIZAJE_BACKBONE (menor),
    para ajustarlas sin "olvidar" lo aprendido en ImageNet.
    """
    cabeza = [p for p in modelo.classifier.parameters() if p.requires_grad]
    backbone = [p for p in modelo.features.parameters() if p.requires_grad]
    grupos = [{"params": cabeza, "lr": config.TASA_APRENDIZAJE}]
    if backbone:
        grupos.append({"params": backbone, "lr": config.TASA_APRENDIZAJE_BACKBONE})
    return grupos


def modo_entrenamiento(modelo: nn.Module) -> None:
    """
    modelo.train() pero dejando en modo evaluación las BatchNorm congeladas.

    requires_grad=False evita que el optimizador toque los pesos, pero en
    modo train las BatchNorm siguen actualizando sus estadísticas
    (running_mean / running_var) con cada lote. Con el augmentation agresivo
    eso desvía las estadísticas de ImageNet que el backbone espera, y en
    evaluación/predicción se usarían las desviadas. Las capas descongeladas
    (afinar_completo=True) sí siguen actualizándose.
    """
    modelo.train()
    for capa in modelo.modules():
        if isinstance(capa, nn.modules.batchnorm._BatchNorm) and not any(
            p.requires_grad for p in capa.parameters()
        ):
            capa.eval()
