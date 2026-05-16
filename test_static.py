"""
test_static.py
Pipeline completo: detección de tablero + YOLO en piezas + FEN.
Proyecto: Robot Ajedrecista - CAETI (UAI)

YOLO corre sobre la foto original (mayor resolución) y los bbox se
proyectan al tablero aplanado 800x800 con la matriz de perspectiva.

Uso:
    python test_static.py <path_imagen>

Controles:
    Q / Esc  ->  salir
"""

import cv2
import numpy as np
import os
import sys

from board_detector import BoardDetector
from piece_classifier import PieceClassifier, CLASS_TO_FEN

WIN = "YOLO | izq: bboxes  der: casillas  — Q para salir"

# Colores (BGR)
COLOR_WHITE_PIECE = (50,  200,  50)   # verde
COLOR_BLACK_PIECE = (30,  140, 255)   # naranja
OVERLAY_WHITE     = (180, 255, 180)   # verde claro
OVERLAY_BLACK     = (100, 180, 255)   # naranja claro
GRID_COLOR        = (220, 220, 220)


def _draw_bboxes(board: np.ndarray,
                 detections_board: list[dict]) -> np.ndarray:
    """Panel izquierdo: bboxes proyectados al tablero, con etiqueta FEN."""
    img = board.copy()
    for det in detections_board:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        is_white = det["class"].startswith("white_")
        color    = COLOR_WHITE_PIECE if is_white else COLOR_BLACK_PIECE
        label    = CLASS_TO_FEN.get(det["class"], "?")

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        ly = max(y1 - 4, th + 4)
        cv2.rectangle(img, (x1, ly - th - 4), (x1 + tw + 6, ly + 2), color, -1)
        cv2.putText(img, label, (x1 + 2, ly - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.putText(img, "YOLO detections", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, "YOLO detections", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _draw_piece_map(board: np.ndarray,
                    pieces: dict[str, str]) -> np.ndarray:
    """Panel derecho: casillas coloreadas + letra FEN centrada."""
    img     = board.copy()
    overlay = img.copy()

    for square, letter in pieces.items():
        ci = ord(square[0]) - ord('a')
        ri = 8 - int(square[1])
        x1, y1 = ci * 100, ri * 100
        x2, y2 = x1 + 100, y1 + 100
        color = OVERLAY_WHITE if letter.isupper() else OVERLAY_BLACK
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

    # Grid
    for i in range(9):
        cv2.line(img, (i * 100, 0),  (i * 100, 800), GRID_COLOR, 1)
        cv2.line(img, (0, i * 100),  (800, i * 100), GRID_COLOR, 1)

    # Letra FEN centrada en cada casilla ocupada
    for square, letter in pieces.items():
        ci = ord(square[0]) - ord('a')
        ri = 8 - int(square[1])
        cx = ci * 100 + 50
        cy = ri * 100 + 58
        text_color = (0, 80, 0) if letter.isupper() else (0, 40, 160)
        cv2.putText(img, letter, (cx - 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, letter, (cx - 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, text_color, 2, cv2.LINE_AA)

    # Notación algebraica pequeña en cada esquina
    cols = "abcdefgh"
    for ri in range(8):
        for ci in range(8):
            cv2.putText(img, f"{cols[ci]}{8 - ri}",
                        (ci * 100 + 3, ri * 100 + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                        (180, 180, 180), 1, cv2.LINE_AA)

    cv2.putText(img, "Piece map", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, "Piece map", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    if len(sys.argv) < 2:
        print("Uso: python test_static.py <path_imagen>")
        sys.exit(1)

    path_imagen = sys.argv[1]
    if not os.path.isfile(path_imagen):
        print(f"[ERROR] No se encontro: {path_imagen}")
        sys.exit(1)

    frame = cv2.imread(path_imagen)
    if frame is None:
        print(f"[ERROR] No se pudo leer: {path_imagen}")
        sys.exit(1)

    # --- Paso 1: detectar y aplanar el tablero ---
    detector    = BoardDetector()
    corners_raw, strategy = detector._detect_board_corners(frame)

    if corners_raw is None:
        print("[ERROR] Tablero NO detectado.")
        sys.exit(1)

    M     = detector._compute_perspective_matrix(corners_raw)
    board = cv2.warpPerspective(frame, M,
                                (detector.board_size, detector.board_size))

    # --- Paso 2: YOLO en foto original → mapear al tablero ---
    classifier  = PieceClassifier()
    pieces_dict = classifier.detect_on_original(frame, M, conf=0.25)

    # --- Paso 3: FEN ---
    fen = classifier.build_fen(pieces_dict)
    print(f"FEN: {fen}")

    # --- Paso 4: visualización 2 paneles ---
    left  = _draw_bboxes(board, classifier.last_detections_board)
    right = _draw_piece_map(board, pieces_dict)

    sep   = np.full((800, 4, 3), 60, dtype=np.uint8)
    panel = np.hstack([left, sep, right])

    cv2.imwrite("result_test_static.jpg", panel)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1604, 800)
    cv2.imshow(WIN, panel)

    while True:
        if cv2.waitKey(50) & 0xFF in [ord('q'), ord('Q'), 27]:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
