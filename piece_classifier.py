"""
piece_classifier.py
Clasificacion de piezas de ajedrez usando YOLO.
Proyecto: Robot Ajedrecista - CAETI (UAI)

Carga de modelo (en orden de prioridad):
  1. chess_model.pt  en el directorio actual (si ya existe)
  2. Descarga automatica via Roboflow API
     (requiere: pip install roboflow  y  set ROBOFLOW_API_KEY=tu_clave)
  3. yolov8n.pt  fallback — confirma que YOLO funciona pero NO detecta piezas

Para obtener el modelo de ajedrez sin API:
  1. Ir a https://universe.roboflow.com/chess-pieces-detection/chess-pieces-detection-v2
  2. Crear cuenta gratis (o entrar con Google)
  3. "Download" → formato "YOLOv8" → descargar best.pt
  4. Guardarlo como chess_model.pt en esta carpeta
"""

import os
import cv2
import numpy as np
import torch
import ultralytics.nn.tasks as _ul_tasks
from ultralytics import YOLO

# -----------------------------------------------------------------------
# Fix de compatibilidad: PyTorch >= 2.6 cambia weights_only=True por
# defecto, lo que rompe la carga de checkpoints YOLOv8 existentes.
# Este patch fuerza weights_only=False solo si la carga segura falla.
# -----------------------------------------------------------------------
_orig_torch_safe_load = _ul_tasks.torch_safe_load

def _patched_torch_safe_load(weight):
    try:
        return _orig_torch_safe_load(weight)
    except Exception:
        return torch.load(weight, map_location="cpu", weights_only=False), weight

_ul_tasks.torch_safe_load = _patched_torch_safe_load


MODEL_PATH = "chess_model.pt"

# Mapeo clase YOLO → letra FEN (mayúscula=blanca, minúscula=negra)
CLASS_TO_FEN = {
    "white_king":   "K", "white_queen":  "Q", "white_rook":   "R",
    "white_bishop": "B", "white_knight": "N", "white_pawn":   "P",
    "black_king":   "k", "black_queen":  "q", "black_rook":   "r",
    "black_bishop": "b", "black_knight": "n", "black_pawn":   "p",
}


class PieceClassifier:
    """
    Detecta y clasifica piezas de ajedrez sobre el tablero aplanado (800x800)
    usando un modelo YOLO entrenado en piezas de ajedrez.

    Metodos principales:
      detect_pieces(board_image)  ->  list[dict]
      map_to_grid(detections)     ->  dict[str, str]   {"e4": "white_pawn", ...}
    """

    def __init__(self, model_path: str = MODEL_PATH, conf_threshold: float = 0.30):
        self.conf_threshold = conf_threshold
        self.model = self._load_model(model_path)
        names = list(self.model.names.values())
        print(f"[INFO] Modelo listo. Clases ({len(names)}): {names}")

    # ------------------------------------------------------------------
    # Carga del modelo
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str) -> YOLO:
        # 1. Modelo local especifico de ajedrez
        if os.path.isfile(model_path):
            print(f"[INFO] Cargando modelo de ajedrez: {model_path}")
            return YOLO(model_path)

        # 2. Fallback: yolov8n.pt (confirma que YOLO funciona, no detecta piezas)
        print()
        print("=" * 62)
        print("  chess_model.pt no encontrado.")
        print("  Usando yolov8n.pt como fallback.")
        print()
        print("  Para obtener un modelo entrenado en piezas de ajedrez:")
        print("  Opcion A — Roboflow Universe (navegador, sin cuenta):")
        print("    1. https://universe.roboflow.com/chess-pieces-detection")
        print("       /chess-pieces-detection-v2")
        print("    2. Download -> YOLOv8 -> best.pt")
        print("    3. Guardar como chess_model.pt en esta carpeta")
        print()
        print("  Opcion B — Entrenar con el dataset de Roboflow:")
        print("    Ver: docs/training.md  (proximo paso)")
        print("=" * 62)
        print()
        return YOLO("yolov8n.pt")

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------

    def detect_pieces(self, board_image: np.ndarray) -> list[dict]:
        """
        Corre inferencia YOLO sobre el tablero aplanado.

        Args:
            board_image: Imagen BGR del tablero (tipicamente 800x800)

        Returns:
            Lista de detecciones, cada una como dict:
            {
              "bbox":       [x1, y1, x2, y2],   # coordenadas en px
              "class":      str,                 # nombre de la clase
              "confidence": float                # confianza 0-1
            }
            Ordenada por confianza descendente.
        """
        results    = self.model(board_image, conf=self.conf_threshold, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                detections.append({
                    "bbox":       [x1, y1, x2, y2],
                    "class":      self.model.names[cls_id],
                    "confidence": conf,
                })

        detections.sort(key=lambda d: -d["confidence"])
        return detections

    # ------------------------------------------------------------------
    # Mapeo a grilla algebraica
    # ------------------------------------------------------------------

    def map_to_grid(self, detections: list[dict],
                    board_size: int = 800) -> dict[str, str]:
        """
        Convierte las detecciones de bbox a notacion algebraica del tablero.

        El centro del bbox determina la casilla:
          columna: cx // (board_size/8)  ->  0=a … 7=h
          fila:    cy // (board_size/8)  ->  0=fila8 … 7=fila1

        Si dos detecciones caen en la misma casilla, gana la de mayor
        confianza.

        Args:
            detections: Salida de detect_pieces()
            board_size: Lado del tablero en px (default 800)

        Returns:
            Dict: {"e4": "white_pawn", "d7": "black_queen", ...}
        """
        cell   = board_size / 8
        cols   = "abcdefgh"
        # casilla -> (clase, confianza)
        best: dict[str, tuple[str, float]] = {}

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            ci = int(cx / cell)
            ri = int(cy / cell)
            if not (0 <= ci <= 7 and 0 <= ri <= 7):
                continue
            square = f"{cols[ci]}{8 - ri}"
            conf   = det["confidence"]
            if square not in best or conf > best[square][1]:
                best[square] = (det["class"], conf)

        return {sq: cls for sq, (cls, _) in best.items()}

    # ------------------------------------------------------------------
    # Detección + mapeo a casilla con letras FEN
    # ------------------------------------------------------------------

    def detect_on_board(self, board_image: np.ndarray,
                        conf: float = 0.25) -> tuple[dict[str, str], list[dict]]:
        """
        Corre YOLO sobre el tablero aplanado (800x800) y devuelve un dict
        casilla → letra FEN y la lista de detecciones raw con sus bboxes.

        Args:
            board_image: Imagen BGR 800x800 del tablero aplanado.
            conf:        Umbral de confianza (default 0.4).

        Returns:
            Tuple (pieces_dict, raw_detections):
              pieces_dict:    {"e4": "P", "d7": "n", ...}
              raw_detections: lista de dicts con "bbox", "class", "confidence"
        """
        results = self.model(board_image, conf=conf, verbose=False)

        raw: list[dict] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                cls_id  = int(box.cls[0])
                conf_v  = float(box.conf[0])
                raw.append({
                    "bbox":       [x1, y1, x2, y2],
                    "class":      self.model.names[cls_id],
                    "confidence": conf_v,
                })

        # Guardar detecciones crudas para visualización externa
        self.last_detections: list[dict] = sorted(raw, key=lambda d: -d["confidence"])

        cell = 100  # 800 px / 8 casillas
        cols = "abcdefgh"
        best: dict[str, tuple[str, float]] = {}

        for det in self.last_detections:
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            ci = int(cx / cell)
            ri = int(cy / cell)
            if not (0 <= ci <= 7 and 0 <= ri <= 7):
                continue
            square   = f"{cols[ci]}{8 - ri}"
            conf_v   = det["confidence"]
            fen_letter = CLASS_TO_FEN.get(det["class"], "?")
            if square not in best or conf_v > best[square][1]:
                best[square] = (fen_letter, conf_v)

        return {sq: letter for sq, (letter, _) in best.items()}, self.last_detections

    # ------------------------------------------------------------------
    # Detección en imagen original + mapeo por perspectiva
    # ------------------------------------------------------------------

    def detect_on_original(self, original_frame: np.ndarray,
                           perspective_M: np.ndarray,
                           board_size: int = 800,
                           conf: float = 0.25) -> dict[str, str]:
        """
        Corre YOLO sobre la foto original (mayor calidad que el warp) y mapea
        cada detección al tablero usando la matriz de perspectiva M.

        El centro (cx, cy) de cada bbox en la foto original se transforma a
        coordenadas del tablero 800x800 con cv2.perspectiveTransform(M).
        Detecciones cuyo centro queda fuera del tablero se descartan.

        Almacena las detecciones raw en self.last_detections (con bbox en
        coordenadas de la foto original) y en self.last_detections_board
        (bbox proyectados al tablero aplanado).

        Args:
            original_frame:  Imagen BGR completa de la cámara.
            perspective_M:   Matriz 3x3 de perspectiva (de _compute_perspective_matrix).
            board_size:      Tamaño del tablero aplanado en px (default 800).
            conf:            Umbral de confianza (default 0.4).

        Returns:
            {"e4": "P", "d7": "n", ...}
        """
        results = self.model(original_frame, conf=conf, imgsz=640, verbose=False)

        raw: list[dict] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                cls_id = int(box.cls[0])
                conf_v = float(box.conf[0])
                raw.append({
                    "bbox":       [x1, y1, x2, y2],
                    "class":      self.model.names[cls_id],
                    "confidence": conf_v,
                })

        self.last_detections = sorted(raw, key=lambda d: -d["confidence"])

        cell = board_size / 8
        cols = "abcdefgh"
        best: dict[str, tuple[str, float, list]] = {}

        # Proyectar centros al espacio del tablero
        centers_orig = []
        for det in self.last_detections:
            x1, y1, x2, y2 = det["bbox"]
            centers_orig.append([(x1 + x2) / 2, (y1 + y2) / 2])

        if not centers_orig:
            self.last_detections_board = []
            return {}

        pts = np.array(centers_orig, dtype=np.float32).reshape(1, -1, 2)
        pts_board = cv2.perspectiveTransform(pts, perspective_M)[0]  # (N, 2)

        self.last_detections_board: list[dict] = []
        for det, (bx, by) in zip(self.last_detections, pts_board):
            ci = int(bx / cell)
            ri = int(by / cell)
            if not (0 <= ci <= 7 and 0 <= ri <= 7):
                continue                     # fuera del tablero
            square     = f"{cols[ci]}{8 - ri}"
            conf_v     = det["confidence"]
            fen_letter = CLASS_TO_FEN.get(det["class"], "?")

            # bbox proyectado al tablero (escalar bbox original → board)
            x1, y1, x2, y2 = det["bbox"]
            corners_o = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                                 dtype=np.float32).reshape(1, -1, 2)
            corners_b = cv2.perspectiveTransform(corners_o, perspective_M)[0]
            bx1 = float(corners_b[:, 0].min())
            by1 = float(corners_b[:, 1].min())
            bx2 = float(corners_b[:, 0].max())
            by2 = float(corners_b[:, 1].max())

            self.last_detections_board.append({
                "bbox":       [bx1, by1, bx2, by2],
                "class":      det["class"],
                "confidence": conf_v,
                "square":     square,
            })

            if square not in best or conf_v > best[square][1]:
                best[square] = (fen_letter, conf_v, [bx1, by1, bx2, by2])

        return {sq: letter for sq, (letter, _, _) in best.items()}

    # ------------------------------------------------------------------
    # Construcción del FEN
    # ------------------------------------------------------------------

    def build_fen(self, pieces_dict: dict[str, str]) -> str:
        """
        Construye la cadena FEN a partir del dict casilla → letra FEN.

        Args:
            pieces_dict: {"e4": "P", "d7": "n", ...}

        Returns:
            FEN completo: "<posición> w - - 0 1"
        """
        rows = []
        for rank in range(8, 0, -1):          # fila 8 → fila 1
            row   = ""
            empty = 0
            for col in "abcdefgh":
                sq = f"{col}{rank}"
                if sq in pieces_dict:
                    if empty:
                        row  += str(empty)
                        empty = 0
                    row += pieces_dict[sq]
                else:
                    empty += 1
            if empty:
                row += str(empty)
            rows.append(row)
        return "/".join(rows) + " w - - 0 1"
