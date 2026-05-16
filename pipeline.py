"""
pipeline.py — Integración completa: visión por computadora + Stockfish
Robot Ajedrecista CAETI (UAI)

Uso:
    python pipeline.py foto_tablero.jpg
"""

import sys
import os
import atexit
import cv2
import numpy as np
import chess

# ---------------------------------------------------------------------------
# Importar ChessEngine desde el proyecto hermano
# ---------------------------------------------------------------------------
_ENGINE_DIR = r"C:\Users\Botmaker\Desktop\robot-ajedrecista-integracion-stockfish-y-python"
sys.path.insert(0, _ENGINE_DIR)
from chess_engine import ChessEngine

from board_detector import BoardDetector
from piece_classifier import PieceClassifier, CLASS_TO_FEN

# ---------------------------------------------------------------------------
# Rutas de configuración
# ---------------------------------------------------------------------------
_STOCKFISH_PATH = (
    r"C:\Users\Botmaker\Downloads\stockfish-windows-x86-64-avx2"
    r"\stockfish\stockfish-windows-x86-64-avx2.exe"
)
_MODEL_PATH = "chess_model.pt"

# ---------------------------------------------------------------------------
# Inicialización global (ocurre al importar o ejecutar el módulo)
# ---------------------------------------------------------------------------
print("[1/3] Inicializando BoardDetector...")
detector = BoardDetector()

print("[2/3] Inicializando PieceClassifier...")
classifier = PieceClassifier(_MODEL_PATH)

print("[3/3] Inicializando ChessEngine (Stockfish)...")
engine = ChessEngine(_STOCKFISH_PATH)

# Liberar Stockfish al salir del proceso
atexit.register(engine.close)


# ---------------------------------------------------------------------------
# Modo básico: detección de ocupación por comparación con referencia vacía
# ---------------------------------------------------------------------------

DIFF_THRESHOLD = 20   # diferencia media mínima para considerar una celda ocupada


def _compute_grid_divisions(warped: np.ndarray) -> tuple:
    """
    Detecta las líneas de la grilla del tablero con HoughLinesP y devuelve
    las posiciones de los 9 bordes horizontales y verticales.
    Si no se encuentran 7+ líneas en cada dirección, usa división uniforme.

    Returns:
        (row_divs, col_divs) — listas de 9 enteros cada una
    """
    h, w = warped.shape[:2]

    gray  = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=50,
        minLineLength=min(w, h) * 0.40,
        maxLineGap=20
    )

    h_pos, v_pos = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
            if angle < 15:
                h_pos.append((y1 + y2) / 2.0)
            elif angle > 75:
                v_pos.append((x1 + x2) / 2.0)

    def cluster_positions(positions, size):
        if not positions:
            return []
        gap = size * 0.04
        sorted_p = sorted(positions)
        clusters, group = [], [sorted_p[0]]
        for p in sorted_p[1:]:
            if p - group[-1] < gap:
                group.append(p)
            else:
                clusters.append(float(np.median(group)))
                group = [p]
        clusters.append(float(np.median(group)))
        return sorted(clusters)

    h_clusters = cluster_positions(h_pos, h)
    v_clusters = cluster_positions(v_pos, w)

    if len(h_clusters) >= 7 and len(v_clusters) >= 7:
        print(f"[BasicPipeline] Hough alignment: "
              f"{len(h_clusters)} H-lines, {len(v_clusters)} V-lines detectadas")

        def make_9_divs(clusters, size):
            divs = [0.0] + [c for c in clusters if 0 < c < size] + [float(size)]
            # Reducir si hay demasiadas: eliminar la que crea el gap más pequeño
            while len(divs) > 9:
                gaps = [divs[i + 1] - divs[i] for i in range(len(divs) - 1)]
                idx  = gaps.index(min(gaps))
                divs.pop(idx + 1 if idx < len(gaps) - 1 else idx)
            # Si faltan, caer a división uniforme
            if len(divs) != 9:
                return [int(size * i / 8) for i in range(9)]
            return [int(d) for d in divs]

        row_divs = make_9_divs(h_clusters, h)
        col_divs = make_9_divs(v_clusters, w)

        if len(row_divs) == 9 and len(col_divs) == 9:
            return row_divs, col_divs

    # División uniforme de respaldo
    return ([int(h * i / 8) for i in range(9)],
            [int(w * i / 8) for i in range(9)])


def detect_occupied_with_reference(warped_current: np.ndarray,
                                    warped_ref: np.ndarray) -> tuple:
    """
    Detecta casillas ocupadas comparando el frame actual contra el tablero vacío
    de referencia.

    Para cada celda (región central 60%):
        diff      = abs(celda_actual − celda_referencia)
        mean_diff = np.mean(diff)
        ocupada   = mean_diff > DIFF_THRESHOLD

    Este enfoque es robusto ante el color y la iluminación base del tablero:
    solo detecta lo que cambió respecto al estado vacío conocido.

    Returns:
        (grid, metrics)
        grid    — lista 8×8 de bool (True = ocupada)
        metrics — lista 8×8 de dict con {'mean_diff': float}
    """
    h, w = warped_current.shape[:2]

    # Asegurar que la referencia tenga el mismo tamaño
    if warped_ref.shape[:2] != (h, w):
        warped_ref = cv2.resize(warped_ref, (w, h))

    row_divs, col_divs = _compute_grid_divisions(warped_current)

    grid    = []
    metrics = []

    for row in range(8):
        row_data    = []
        row_metrics = []

        for col in range(8):
            y0 = row_divs[row]
            y1 = row_divs[row + 1]
            x0 = col_divs[col]
            x1 = col_divs[col + 1]

            # Región central 60%
            cy0 = y0 + max(1, int((y1 - y0) * 0.20))
            cy1 = y1 - max(1, int((y1 - y0) * 0.20))
            cx0 = x0 + max(1, int((x1 - x0) * 0.20))
            cx1 = x1 - max(1, int((x1 - x0) * 0.20))

            cell_cur = warped_current[cy0:cy1, cx0:cx1].astype(np.float32)
            cell_ref = warped_ref[cy0:cy1, cx0:cx1].astype(np.float32)

            mean_diff = float(np.mean(np.abs(cell_cur - cell_ref)))
            occupied  = mean_diff > DIFF_THRESHOLD

            row_data.append(occupied)
            row_metrics.append({"mean_diff": mean_diff})

        grid.append(row_data)
        metrics.append(row_metrics)

    return grid, metrics


def run_basic_pipeline(image_path: str, reference_path: str | None = None) -> dict:
    """
    Pipeline básico: solo OpenCV, sin YOLO ni Stockfish.
    Detecta el tablero, lo aplana y determina qué casillas están ocupadas
    comparando contra una imagen de referencia del tablero vacío.

    Args:
        image_path:     Ruta a la imagen actual (con piezas).
        reference_path: Ruta a la imagen de referencia (tablero vacío).
                        Si es None o no existe, retorna error explicativo.

    Returns:
        {
            "success":        bool,
            "error":          str | None,
            "_warped_board":  np.ndarray,
            "occupied_grid":  list[list[bool]],   # 8×8
            "occupied_count": int,
            "cell_metrics":   list[list[dict]],   # 8×8 — {'mean_diff': float}
        }
    """
    result: dict = {
        "success":        False,
        "error":          None,
        "_warped_board":  None,
        "occupied_grid":  None,
        "occupied_count": 0,
        "cell_metrics":   None,
    }

    try:
        # Verificar que existe la referencia
        if not reference_path or not os.path.exists(reference_path):
            raise RuntimeError(
                "Primero presiona R con el tablero vacio para guardar la referencia"
            )

        print(f"[BasicPipeline] Cargando imagen: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

        print(f"[BasicPipeline] Cargando referencia: {reference_path}")
        ref_image = cv2.imread(reference_path)
        if ref_image is None:
            raise FileNotFoundError(f"No se pudo cargar la referencia: {reference_path}")

        print("[BasicPipeline] Detectando tablero (imagen actual)...")
        warped = detector.detect_board(image)
        if warped is None:
            raise RuntimeError(
                "No se detectó el tablero. "
                "Asegurate de que el tablero sea visible y bien iluminado."
            )
        print(f"[BasicPipeline] Tablero aplanado: {warped.shape[1]}x{warped.shape[0]} px")

        print("[BasicPipeline] Detectando tablero (referencia vacía)...")
        warped_ref = detector.detect_board(ref_image)
        if warped_ref is None:
            raise RuntimeError(
                "No se detectó el tablero en la imagen de referencia. "
                "Volvé a presionar R con el tablero vacío bien visible."
            )

        grid, metrics = detect_occupied_with_reference(warped, warped_ref)
        count = sum(v for row in grid for v in row)
        print(f"[BasicPipeline] Casillas ocupadas: {count}/64  (umbral diff={DIFF_THRESHOLD})")

        # Imprimir tabla de diferencias por casilla
        cols = list("abcdefgh")
        print(f"\n{'Casilla':>8} {'mean_diff':>10}  {'OCC':>4}")
        print("  " + "-" * 30)
        for row_i, (row_bool, row_met) in enumerate(zip(grid, metrics)):
            for col_i, (occ, m) in enumerate(zip(row_bool, row_met)):
                sq = f"{cols[col_i]}{8 - row_i}"
                print(f"  {sq:>6}  {m['mean_diff']:>9.1f}  {'OCC' if occ else '---':>4}")

        result.update({
            "success":        True,
            "_warped_board":  warped,
            "occupied_grid":  grid,
            "occupied_count": count,
            "cell_metrics":   metrics,
        })

    except Exception as exc:
        result["error"] = str(exc)
        print(f"[BasicPipeline] Error: {exc}")

    return result


# ---------------------------------------------------------------------------
# Validación de FEN con python-chess
# ---------------------------------------------------------------------------

def validate_fen(fen: str) -> tuple[bool, str]:
    try:
        board = chess.Board(fen)
        if not board.king(chess.WHITE) or not board.king(chess.BLACK):
            return False, "Falta un rey en la posición"
        return True, ""
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_pipeline(image_path: str) -> dict:
    """
    Ejecuta el pipeline completo sobre una imagen de tablero.

    Args:
        image_path: Ruta a la imagen JPG/PNG del tablero físico.

    Returns:
        {
            "fen":           str,        # FEN detectado por visión
            "best_move_uci": str,        # ej. "e2e4"
            "best_move_san": str,        # ej. "e4"
            "board_visual":  str,        # tablero en consola (unicode)
            "success":       bool,
            "error":         str | None,
            "_warped_board": np.ndarray  # tablero aplanado 800x800 (uso interno)
        }
    """
    result: dict = {
        "fen":             None,
        "best_move_uci":   None,
        "best_move_san":   None,
        "board_visual":    None,
        "success":         False,
        "error":           None,
        "_warped_board":   None,
        "_raw_detections": [],
    }

    try:
        # 1. Cargar imagen
        print(f"[Pipeline] Cargando imagen: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

        # 2. Detectar y aplanar el tablero
        print("[Pipeline] Detectando tablero...")
        warped = detector.detect_board(image)
        if warped is None:
            raise RuntimeError(
                "No se detectó el tablero en la imagen. "
                "Asegúrate de que el tablero sea visible y bien iluminado."
            )
        print(f"[Pipeline] Tablero aplanado: {warped.shape[1]}x{warped.shape[0]} px")

        # 3. Detectar piezas con YOLO y construir FEN
        print("[Pipeline] Detectando piezas con YOLO...")
        pieces_dict, raw_detections = classifier.detect_on_board(warped)
        result["_raw_detections"] = raw_detections
        print(f"[Pipeline] Piezas detectadas: {len(pieces_dict)}")

        fen = classifier.build_fen(pieces_dict)

        # Imprimir FEN prominentemente para depurar detección YOLO
        print()
        print("=" * 60)
        print(f"  FEN detectado por YOLO: {fen}")
        print("=" * 60)
        print()

        # 4. Validar FEN antes de pasarlo a Stockfish
        valid, reason = validate_fen(fen)
        if not valid:
            return {
                "success":         False,
                "error":           f"FEN inválido: {reason}",
                "fen":             fen,
                "best_move_uci":   None,
                "best_move_san":   None,
                "board_visual":    None,
                "_warped_board":   warped,
                "_raw_detections": raw_detections,
            }

        # 5. Cargar FEN en el motor
        engine.set_position(fen)

        # 6. Calcular mejor jugada con Stockfish
        print("[Pipeline] Calculando mejor jugada...")
        uci, san = engine.get_best_move()

        # 7. Representación visual del tablero
        board_visual = engine.get_board_visual()

        result.update({
            "fen":           fen,
            "best_move_uci": uci,
            "best_move_san": san,
            "board_visual":  board_visual,
            "success":       True,
            "error":         None,
            "_warped_board": warped,
        })
        # _raw_detections ya fue asignado antes de la validación

    except Exception as exc:
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Bloque principal
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python pipeline.py <foto_tablero.jpg>")
        sys.exit(1)

    image_path = sys.argv[1]
    result = run_pipeline(image_path)

    if not result["success"]:
        print(f"\n[ERROR] {result['error']}")
        sys.exit(1)

    # --- Output de consola ---
    print(f"\nFEN detectado:  {result['fen']}")
    print(f"Mejor jugada:   {result['best_move_san']} ({result['best_move_uci']})")
    print()
    print(result["board_visual"])

    # --- Ventana OpenCV con tablero aplanado y jugada sugerida ---
    display = result["_warped_board"].copy()

    # Dibujar bboxes de piezas detectadas por YOLO
    _COLOR_WHITE = (0, 220, 0)    # verde  (BGR) — piezas blancas
    _COLOR_BLACK = (0, 140, 255)  # naranja (BGR) — piezas negras

    for det in result["_raw_detections"]:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cls   = det["class"]
        letter = CLASS_TO_FEN.get(cls, "?")
        conf_v = det["confidence"]
        label  = f"{letter} {conf_v:.2f}"
        color  = _COLOR_WHITE if cls.startswith("white_") else _COLOR_BLACK

        # Rectángulo alrededor de la pieza
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        # "Q 0.82" encima del rectángulo (sombra negra + color)
        text_y = max(y1 - 4, 14)
        cv2.putText(display, label, (x1 + 2, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, label, (x1 + 2, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # Texto principal: jugada sugerida
    move_text = f"{result['best_move_san']}  ({result['best_move_uci']})"

    # Sombra negra para legibilidad sobre cualquier fondo
    cv2.putText(
        display, move_text,
        (18, 46),
        cv2.FONT_HERSHEY_SIMPLEX, 1.3,
        (0, 0, 0), 5, cv2.LINE_AA,
    )
    # Texto verde
    cv2.putText(
        display, move_text,
        (18, 46),
        cv2.FONT_HERSHEY_SIMPLEX, 1.3,
        (0, 220, 0), 3, cv2.LINE_AA,
    )

    # Subtítulo con FEN (fuente pequeña)
    fen_short = result["fen"].split(" ")[0]   # solo la parte de posición
    cv2.putText(
        display, fen_short,
        (10, display.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (0, 0, 0), 3, cv2.LINE_AA,
    )
    cv2.putText(
        display, fen_short,
        (10, display.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (200, 255, 200), 1, cv2.LINE_AA,
    )

    cv2.imshow("Tablero detectado — Jugada sugerida", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
