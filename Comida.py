"""
Punto de entrada único del identificador de sabores de comida.

    python Comida.py datos        -> prepara y verifica los tres datasets
    python Comida.py foodsense    -> regenera el complemento humano FoodSense
    python Comida.py entrenar     -> entrena y guarda el modelo
    python Comida.py interfaz     -> abre la aplicación gráfica
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


def comando_foodsense():
    import foodsense

    foodsense.preparar()


def comando_predecir(args_restantes):
    import predecir

    sys.argv = [sys.argv[0]] + args_restantes
    predecir.main()


def comando_interfaz():
    import interfaz

    interfaz.main()


def main():
    parser = argparse.ArgumentParser(description="Identificador multi-etiqueta de sabores de comida.")
    subparsers = parser.add_subparsers(dest="comando", required=True)
    subparsers.add_parser("datos", help="Prepara y verifica los tres datasets.")
    subparsers.add_parser("entrenar", help="Entrena el modelo.")
    subparsers.add_parser("foodsense", help="Regenera y descarga el complemento humano FoodSense.")
    subparsers.add_parser("interfaz", help="Abre la interfaz gráfica de predicción.")
    subparsers.add_parser("predecir", help="Predice sobre una imagen o carpeta.")

    args, restantes = parser.parse_known_args()

    if args.comando == "datos":
        comando_datos()
    elif args.comando == "entrenar":
        comando_entrenar()
    elif args.comando == "foodsense":
        comando_foodsense()
    elif args.comando == "interfaz":
        comando_interfaz()
    elif args.comando == "predecir":
        comando_predecir(restantes)


if __name__ == "__main__":
    main()
