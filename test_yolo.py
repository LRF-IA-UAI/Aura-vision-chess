"""
test_yolo.py
Prueba de YOLO directamente sobre la imagen original (sin warp de tablero).
Proyecto: Robot Ajedrecista - CAETI (UAI)

Uso:
    python test_yolo.py foto_tablero.jpg

Controles:
    Q / Esc  ->  salir
"""

import cv2
import os
import sys
from collections import Counter

from piece_classifier import PieceClassifier


WIN = "YOLO detecciones | Q para salir"


def color_para_clase(cls_name: str) -> tuple:
    """Asigna un color BGR segun si la pieza es blanca, negra u otra."""
    nombre = cls_name.lower()
    if "white" in nombre:
        return (200, 200, 255)   # azul claro
    if "black" in nombre:
        return (0, 140, 255)     # naranja
    return (0, 220, 0)           # verde para clases inesperadas


def dibujar_detecciones(imagen, detections: list[dict]):
    """Dibuja todos los bboxes con etiqueta y confianza sobre la imagen."""
    anotado = imagen.copy()
    font    = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color = color_para_clase(det["class"])
        label = f"{det['class']}  {det['confidence']:.2f}"

        cv2.rectangle(anotado, (x1, y1), (x2, y2), color, 2)

        # Fondo opaco para el texto
        (tw, th), _ = cv2.getTextSize(label, font, 0.42, 1)
        cv2.rectangle(anotado, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(anotado, label, (x1 + 2, y1 - 3),
                    font, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    return anotado


def main():
    if len(sys.argv) < 2:
        print("Uso: python test_yolo.py <imagen>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"[ERROR] No se encontro: {path}")
        sys.exit(1)

    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERROR] No se pudo leer: {path}")
        sys.exit(1)

    print(f"[INFO] Imagen: {path}  ({frame.shape[1]}x{frame.shape[0]})")

    # Crear clasificador (imprime las clases del modelo al iniciar)
    classifier = PieceClassifier()

    # Correr inferencia sobre la imagen original (sin warp)
    detections = classifier.detect_pieces(frame)

    # Imprimir resultado en consola
    print(f"\n{'='*55}")
    print(f"  DETECCIONES: {len(detections)} total")
    print(f"{'='*55}")
    for i, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        print(f"  {i:2d}. {det['class']:<22}  conf={det['confidence']:.2f}"
              f"  bbox=({x1},{y1})-({x2},{y2})")

    print(f"\n  CONTEO POR CLASE:")
    conteo = Counter(d["class"] for d in detections)
    for cls, n in sorted(conteo.items()):
        print(f"    {cls:<22} {n}")
    print(f"{'='*55}\n")

    # Mostrar imagen anotada
    anotado = dibujar_detecciones(frame, detections)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1000, 700)
    cv2.imshow(WIN, anotado)

    while True:
        if cv2.waitKey(50) & 0xFF in [ord('q'), ord('Q'), 27]:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
