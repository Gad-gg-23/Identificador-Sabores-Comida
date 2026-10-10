"""
Punto de entrada único del identificador de sabores de comida.

    python Comida.py datos        -> descarga/verifica Food-101
    python Comida.py entrenar     -> entrena y guarda el modelo
    python Comida.py predecir --imagen ruta.jpg
    python Comida.py predecir --carpeta ruta/
"""

import argparse
import sys


def comando_datos():
    import dataset

    splits = dataset.construir_splits()
    for nombre, lista in splits.items():
        print(f"  {nombre}: {len(lista)} imágenes")


def comando_entrenar():
    import entrenar

    entrenar.entrenar()


def comando_predecir(args_restantes):
    import predecir

    sys.argv = [sys.argv[0]] + args_restantes
    predecir.main()


def main():
    parser = argparse.ArgumentParser(description="Identificador multi-etiqueta de sabores de comida.")
    subparsers = parser.add_subparsers(dest="comando", required=True)
    subparsers.add_parser("datos", help="Descarga y verifica Food-101.")
    subparsers.add_parser("entrenar", help="Entrena el modelo.")
    subparsers.add_parser("predecir", help="Predice sobre una imagen o carpeta.")

    args, restantes = parser.parse_known_args()

    if args.comando == "datos":
        comando_datos()
    elif args.comando == "entrenar":
        comando_entrenar()
    elif args.comando == "predecir":
        comando_predecir(restantes)


if __name__ == "__main__":
    main()
