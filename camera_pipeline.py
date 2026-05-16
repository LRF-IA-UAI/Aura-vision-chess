"""
camera_pipeline.py — Feed de cámara en vivo con análisis de tablero
Robot Ajedrecista CAETI (UAI)

Uso:
    python camera_pipeline.py
"""

# SETUP INICIAL (una vez):
# 1. Colocar el tablero de ajedrez con su BORDE ROJO en el campo de visión de la cámara.
# 2. Asegurarse de que la iluminación sea razonablemente uniforme — evitar brillos directos
#    o sombras fuertes sobre el borde rojo, ya que el tracking depende de detectar ese color.
# 3. Con el tablero VACÍO, presionar R para calibrar. La tecla R:
#    a) Detecta el borde rojo del tablero como referencia geométrica de tracking continuo.
#    b) Detecta las esquinas internas del ajedrez (findChessboardCorners) para sub-pixel.
#    c) Calcula la homografía borde-rojo → área de juego y la guarda para tracking futuro.
# 4. Una vez calibrado, la homografía se actualiza automáticamente cada frame mientras
#    el borde rojo sea visible. Si se pierde momentáneamente, se usa la última matriz
#    válida (modo STALE — el HUD lo indica en amarillo).

import json
import os
import queue
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2
import numpy as np

from pipeline import (run_pipeline, run_basic_pipeline, detector as _board_detector,
                      classifier as _classifier, engine as _engine,
                      validate_fen as _validate_fen)
from piece_classifier import CLASS_TO_FEN

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
CAPTURE_PATH     = "ultima_captura.jpg"
REFERENCE_PATH   = "tablero_vacio.jpg"
TARGET_W, TARGET_H = 1280, 720
PIPELINE_TIMEOUT = 10.0   # segundos
MSG_DURATION     = 5.0    # segundos que dura el mensaje de estado

_COLOR_WHITE  = (0, 220, 0)    # verde  BGR — piezas blancas
_COLOR_BLACK  = (0, 140, 255)  # naranja BGR — piezas negras
_COLOR_GREEN  = (0, 220, 0)
_COLOR_RED    = (0, 0, 220)
_COLOR_YELLOW = (0, 220, 220)
_BLACK        = (0, 0, 0)
_WHITE        = (255, 255, 255)

WINDOW_MAIN  = "Feed en vivo — Robot Ajedrecista"
WINDOW_BOARD = "Tablero detectado — Jugada sugerida"
WINDOW_BASIC = "Tablero basico — Ocupacion de casillas"   # legacy, no usado por el nuevo modo B
WINDOW_OCC   = "Ocupacion — Modo B"

# ---------------------------------------------------------------------------
# Estado compartido entre hilo principal y hilo del pipeline
# ---------------------------------------------------------------------------
_state = {
    "running":       False,   # pipeline completo corriendo ahora
    "result":        None,    # último resultado del pipeline completo
    "msg_text":      "",      # texto a mostrar en el feed
    "msg_color":     _COLOR_GREEN,
    "msg_expire":    0.0,     # time.time() hasta cuándo mostrarlo
    "pending_board": False,   # hay tablero nuevo por mostrar (modo completo)
    "lock":          threading.Lock(),
    # --- Modo básico ---
    "basic_running": False,   # pipeline básico corriendo ahora
    "basic_result":  None,    # último resultado del pipeline básico
    "basic_pending": False,   # hay resultado básico nuevo por mostrar
}

# ---------------------------------------------------------------------------
# Estado del modo debug visual (tecla D)
# ---------------------------------------------------------------------------
_debug_state = {
    "active":      False,   # overlay de debug visible
    "corners":     None,    # últimas esquinas detectadas (caché)
    "strategy":    None,    # estrategia usada en la última detección
    "last_update": 0.0,     # time.time() del último ciclo de detección
}
_DEBUG_REFRESH = 0.5        # segundos entre re-detecciones para no bloquear la UI

# ---------------------------------------------------------------------------
# Estado de calibración de homografía (tecla R)
# ---------------------------------------------------------------------------
_calibration = {
    "matrix":       None,   # np.ndarray 3x3 — None si no calibrado
    "corners":      None,   # np.ndarray (4,2) en orden TL,TR,BR,BL — None si no calibrado
    "empty_warped": None,   # np.ndarray 800×800 BGR — tablero vacío aplanado (para modo B)
}

# ---------------------------------------------------------------------------
# Estado del tracking por borde rojo continuo
# ---------------------------------------------------------------------------
_tracking_state = {
    "red_to_board_homography": None,   # H: corners del borde rojo → esquinas del área de juego
    "last_board_corners":      None,   # últimas esquinas estimadas (4,2) float32
    "tracking_status":         "OFF",  # "OFF" | "ON (red border)" | "STALE (red border lost)"
    "_red_border_corners":     None,   # (4,2) float32 corners del borde rojo (para debug)
}

# ---------------------------------------------------------------------------
# Estado del modo B: ocupación en vivo (frame-diff vs tablero vacío)
# ---------------------------------------------------------------------------
_mode_b = {
    "active":    False,   # True mientras la ventana de ocupación está visible
    "threshold": 15,      # umbral de mean_diff para considerar casilla ocupada
}

# ---------------------------------------------------------------------------
# Dashboard web — MQTT, MJPEG y filtro de estabilidad
# ---------------------------------------------------------------------------
try:
    import paho.mqtt.client as _paho_mqtt
    _MQTT_AVAILABLE = True
except ImportError:
    _MQTT_AVAILABLE = False

_MQTT_BROKER   = "localhost"
_MQTT_PORT     = 1883
_MJPEG_PORT    = 8765
_STABLE_FRAMES = 20     # frames consecutivos para considerar estado estable

_mqtt_client   = None           # asignado en _start_mqtt()
_command_queue = queue.Queue()  # comandos del dashboard → main loop

_mjpeg_lock = threading.Lock()
_mjpeg_buf  = [b""]            # [0] = bytes JPEG del frame más reciente

# Buffer de estabilidad: 8×8 de {"state": str | None, "count": int}
_stable_buf     = [[{"state": None, "count": 0} for _ in range(8)] for _ in range(8)]
_last_pub_board = None          # último tablero publicado (para detectar cambios)

# ---------------------------------------------------------------------------
# Monkey-patch de _detect_board_corners para usar corners calibrados.
# Cuando _calibration["corners"] es no-None, TODOS los flujos que llamen a
# detector._detect_board_corners() (incluyendo detect_board(), run_pipeline(),
# run_basic_pipeline() y el debug overlay) recibirán los corners calibrados
# sin re-detectar.
# ---------------------------------------------------------------------------
_original_detect_corners = _board_detector._detect_board_corners

def _detect_corners_with_calibration(image: np.ndarray):
    if _calibration["corners"] is not None:
        return _calibration["corners"].copy(), "chessboard_corners"
    return _original_detect_corners(image)

_board_detector._detect_board_corners = _detect_corners_with_calibration


# ---------------------------------------------------------------------------
# Detección de esquinas internas + extrapolación para calibración (tecla R)
# ---------------------------------------------------------------------------
_CAL_PATTERN = (7, 7)


def _find_inner_corners_for_calibration(frame: np.ndarray):
    """
    Detecta las 49 esquinas internas del tablero (patrón 7×7) usando
    exclusivamente findChessboardCornersSB / findChessboardCorners.
    Prueba múltiples escalas y devuelve un array (7, 7, 2) en coordenadas
    de la imagen original (con refinamiento subpixel), o None si falla.
    Solo se usa en el flujo de calibración (tecla R).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for scale in [1.0, 0.75, 0.5]:
        h, w = gray.shape
        scaled = (cv2.resize(gray, (int(w * scale), int(h * scale)))
                  if scale != 1.0 else gray.copy())
        scaled_eq = cv2.equalizeHist(scaled)

        ret, corners = False, None

        # Intentar findChessboardCornersSB primero (más robusto, OpenCV 4+)
        try:
            ret, corners = cv2.findChessboardCornersSB(scaled_eq, _CAL_PATTERN)
            if not ret:
                ret, corners = cv2.findChessboardCornersSB(scaled, _CAL_PATTERN)
        except cv2.error:
            ret = False

        # Fallback a findChessboardCorners estándar
        if not ret:
            flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
                     + cv2.CALIB_CB_NORMALIZE_IMAGE
                     + cv2.CALIB_CB_FAST_CHECK)
            ret, corners = cv2.findChessboardCorners(scaled_eq, _CAL_PATTERN, flags)
            if not ret:
                ret, corners = cv2.findChessboardCorners(scaled, _CAL_PATTERN, flags)

        if not ret or corners is None:
            continue

        # Volver a resolución original
        if scale != 1.0:
            corners = corners / scale

        # Refinar posición subpixel en la imagen a escala completa
        corners_f32 = corners.reshape(-1, 1, 2).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_f32 = cv2.cornerSubPix(gray, corners_f32, (11, 11), (-1, -1), criteria)

        return corners_f32.reshape(7, 7, 2)   # filas × cols × (x, y)

    return None


def extrapolate_outer_corners(inner_7x7: np.ndarray) -> np.ndarray:
    """
    Extrapola las 4 esquinas exteriores del tablero 8×8 a partir de las
    49 esquinas internas en grilla 7×7.

    inner_7x7[0,0] = top-left interno, [0,6] = top-right interno,
    [6,0] = bottom-left interno, [6,6] = bottom-right interno.

    Para cada esquina exterior se proyecta una casilla más afuera usando
    el vecino diagonal inmediato como referencia de paso:
        outer_TL = 2 * inner[0,0] - inner[1,1]
        outer_TR = 2 * inner[0,6] - inner[1,5]
        outer_BR = 2 * inner[6,6] - inner[5,5]
        outer_BL = 2 * inner[6,0] - inner[5,1]

    Returns:
        Array (4, 2) float32 en orden [TL, TR, BR, BL]
    """
    tl = 2.0 * inner_7x7[0, 0] - inner_7x7[1, 1]
    tr = 2.0 * inner_7x7[0, 6] - inner_7x7[1, 5]
    br = 2.0 * inner_7x7[6, 6] - inner_7x7[5, 5]
    bl = 2.0 * inner_7x7[6, 0] - inner_7x7[5, 1]
    return np.array([tl, tr, br, bl], dtype=np.float32).reshape(4, 2)


def _detect_red_border_corners(frame: np.ndarray):
    """
    Detecta los 4 corners del borde rojo del tablero en el frame.

    Pipeline:
      1. BGR → HSV.
      2. Máscara roja: mask_low (H 0-10) | mask_high (H 170-180), S>80, V>50.
      3. Morfología: dilate 5×5 × 2 (cerrar gaps) → erode 5×5 × 1 (no inflar de más).
      4. findContours con RETR_CCOMP (jerarquía 2 niveles: externo e interno).
      5. Filtrar: área > 5 % del frame Y tiene padre en la jerarquía (es el hueco interior).
      6. Elegir el candidato de mayor área.
      7. approxPolyDP iterativo (eps 1 %→5 % del perímetro) hasta obtener 4 vértices.
      8. Validar aspect ratio del bounding box ∈ [0.7, 1.4].
      9. Ordenar TL, TR, BR, BL con suma/diferencia de coordenadas.

    Returns: np.ndarray (4, 2) float32 o None si falla.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_low  = cv2.inRange(hsv, (0,   80, 50), (10,  255, 255))
    mask_high = cv2.inRange(hsv, (170, 80, 50), (180, 255, 255))
    mask = cv2.bitwise_or(mask_low, mask_high)

    kern = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kern, iterations=2)
    mask = cv2.erode(mask,  kern, iterations=1)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return None

    h_frame, w_frame = frame.shape[:2]
    min_area  = 0.05 * h_frame * w_frame
    hier      = hierarchy[0]   # shape (N, 4): next, prev, firstChild, parent

    # Candidatos: área > 5 % del frame Y tiene padre (es el hueco interior del marco rojo)
    candidates = []
    for i, cnt in enumerate(contours):
        area   = cv2.contourArea(cnt)
        parent = hier[i][3]
        if area > min_area and parent >= 0:
            candidates.append((area, cnt))

    if not candidates:
        return None

    _, best = max(candidates, key=lambda x: x[0])

    # approxPolyDP iterativo hasta 4 vértices
    peri   = cv2.arcLength(best, True)
    approx = None
    for pct in [0.01, 0.02, 0.03, 0.04, 0.05]:
        approx = cv2.approxPolyDP(best, pct * peri, True)
        if len(approx) == 4:
            break

    if approx is None or len(approx) != 4:
        return None

    # Validar aspect ratio
    _, _, bw, bh = cv2.boundingRect(approx)
    if bh == 0:
        return None
    ar = bw / bh
    if not (0.7 <= ar <= 1.4):
        return None

    # Ordenar TL, TR, BR, BL
    pts = approx.reshape(4, 2).astype(np.float32)
    s   = pts.sum(axis=1)
    d   = np.diff(pts, axis=1).flatten()
    tl  = pts[np.argmin(s)]
    br  = pts[np.argmax(s)]
    tr  = pts[np.argmin(d)]
    bl  = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def _update_red_border_tracking(frame: np.ndarray) -> None:
    """
    Detecta el borde rojo del tablero y, si la homografía está calibrada,
    actualiza _calibration["matrix"] y ["corners"] con la estimación actual.
    Debe llamarse al inicio de cada frame, ANTES del debug overlay y modo B.
    """
    red_corners = _detect_red_border_corners(frame)

    # Caché para el debug overlay (actualizar siempre)
    _tracking_state["_red_border_corners"] = red_corners

    if red_corners is not None and _tracking_state["red_to_board_homography"] is not None:
        H         = _tracking_state["red_to_board_homography"]
        board_pts = cv2.perspectiveTransform(
            red_corners.reshape(4, 1, 2), H
        ).reshape(4, 2)
        ordered   = _board_detector._ordenar_esquinas(board_pts.astype(np.float32))

        _calibration["corners"] = ordered
        _calibration["matrix"]  = _board_detector._compute_perspective_matrix(ordered)
        _tracking_state["last_board_corners"] = ordered
        _tracking_state["tracking_status"]    = "ON (red border)"

    elif _calibration["matrix"] is not None:
        _tracking_state["tracking_status"] = "STALE (red border lost)"
    else:
        _tracking_state["tracking_status"] = "OFF"


# ---------------------------------------------------------------------------
# Ayudas de dibujo
# ---------------------------------------------------------------------------

def _put_text_outlined(img, text, pos, scale=0.7, color=_WHITE, thickness=2):
    """Texto con sombra negra para legibilidad sobre cualquier fondo."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                _BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def _draw_hud(frame, cam_label, running, pipeline_start,
              msg_text, msg_color, msg_expire, tracking_status="OFF"):
    """Dibuja toda la HUD sobre el frame (modifica in-place)."""
    h, w = frame.shape[:2]

    # --- Línea superior: estado ---
    now = time.time()
    if running and pipeline_start is not None:
        elapsed = now - pipeline_start
        top_txt   = f"Analizando...  {elapsed:.1f}s"
        top_color = _COLOR_YELLOW
    elif msg_expire > now and msg_text:
        top_txt   = msg_text
        top_color = msg_color
    else:
        top_txt   = None
        top_color = _WHITE

    if top_txt:
        _put_text_outlined(frame, top_txt, (14, 38),
                           scale=0.75, color=top_color, thickness=2)

    # --- Indicador de calibración + tracking borde rojo (esquina superior derecha) ---
    if tracking_status.startswith("ON"):
        cal_txt   = "CAL: ON | TRACKING: red border OK"
        cal_color = (0, 220, 0)     # verde
    elif tracking_status.startswith("STALE"):
        cal_txt   = "CAL: ON | TRACKING: STALE (borde rojo perdido)"
        cal_color = (0, 220, 220)   # amarillo
    else:
        cal_txt   = "CAL: OFF (presiona R con tablero vacio)"
        cal_color = (0, 0, 220)     # rojo

    (cal_tw, _), _ = cv2.getTextSize(cal_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    _put_text_outlined(frame, cal_txt, (w - cal_tw - 16, 38),
                       scale=0.45, color=cal_color, thickness=1)

    # --- Barra inferior: instrucciones + etiqueta de cámara ---
    bar_h = 36
    cv2.rectangle(frame, (0, h - bar_h), (w, h), (30, 30, 30), -1)

    _put_text_outlined(frame,
                       "ESPACIO: analisis  |  R: calibrar  |  B: ocupacion  |  P: photo  |  D: debug  |  Q: salir",
                       (12, h - 10), scale=0.55, color=_WHITE)

    cam_txt = cam_label
    (tw, _), _ = cv2.getTextSize(cam_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    _put_text_outlined(frame, cam_txt, (w - tw - 16, h - 10),
                       scale=0.55, color=_COLOR_YELLOW)


def _build_board_display(result: dict) -> np.ndarray:
    """Construye la imagen del tablero aplanado con bboxes de piezas."""
    warped = result.get("_warped_board")
    if warped is None:
        return np.zeros((800, 800, 3), dtype=np.uint8)

    display = warped.copy()

    # Bboxes de piezas detectadas
    for det in result.get("_raw_detections", []):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cls    = det["class"]
        letter = CLASS_TO_FEN.get(cls, "?")
        conf_v = det["confidence"]
        label  = f"{letter} {conf_v:.2f}"
        color  = _COLOR_WHITE if cls.startswith("white_") else _COLOR_BLACK

        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        text_y = max(y1 - 4, 14)
        cv2.putText(display, label, (x1 + 2, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _BLACK, 3, cv2.LINE_AA)
        cv2.putText(display, label, (x1 + 2, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # Jugada sugerida (arriba)
    san = result.get("best_move_san", "")
    uci = result.get("best_move_uci", "")
    move_text = f"{san}  ({uci})"
    cv2.putText(display, move_text, (18, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, _BLACK, 5, cv2.LINE_AA)
    cv2.putText(display, move_text, (18, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, _COLOR_GREEN, 3, cv2.LINE_AA)

    # FEN (abajo)
    fen_short = (result.get("fen") or "").split(" ")[0]
    dh = display.shape[0]
    cv2.putText(display, fen_short, (10, dh - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _BLACK, 3, cv2.LINE_AA)
    cv2.putText(display, fen_short, (10, dh - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1, cv2.LINE_AA)

    return display


# DEPRECATED — reemplazado por frame-diff vs referencia en vivo (modo B con tecla B).
# Mantenido como referencia; ya no se invoca desde el loop principal.
def _legacy_occupancy_basic_display(warped: np.ndarray,
                                     occupied_grid: list,
                                     occupied_count: int,
                                     cell_metrics: list | None = None) -> np.ndarray:
    """
    [DEPRECATED] Tablero aplanado con overlay semitransparente.
    Reemplazado por _b_build_display + _b_compute_occupancy (modo B en vivo).
    """
    display = warped.copy()
    h, w = display.shape[:2]
    cell_h = h // 8
    cell_w = w // 8

    overlay = display.copy()
    for row in range(8):
        for col in range(8):
            x0 = col * cell_w
            y0 = row * cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            color = (0, 0, 180) if occupied_grid[row][col] else (0, 180, 0)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)

    cv2.addWeighted(overlay, 0.35, display, 0.65, 0, display)

    # Valor de mean_diff por celda (texto pequeño para calibración)
    if cell_metrics is not None:
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.30
        for row in range(8):
            for col in range(8):
                m   = cell_metrics[row][col]
                x0  = col * cell_w
                y0  = row * cell_h
                txt = f"d{m['mean_diff']:.0f}"
                cv2.putText(display, txt,
                            (x0 + 4, y0 + 18), font, scale, _BLACK, 2, cv2.LINE_AA)
                cv2.putText(display, txt,
                            (x0 + 4, y0 + 18), font, scale, _WHITE, 1, cv2.LINE_AA)

    # Título con conteo y umbral
    title = f"Ocupadas: {occupied_count}/64 | Umbral diff: 20"
    cv2.putText(display, title, (18, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, _BLACK, 5, cv2.LINE_AA)
    cv2.putText(display, title, (18, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, _COLOR_YELLOW, 3, cv2.LINE_AA)

    # Leyenda inferior
    dh = display.shape[0]
    legend = "ROJO=ocupada  VERDE=vacia  d=diff_vs_referencia"
    cv2.putText(display, legend, (10, dh - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, _BLACK, 3, cv2.LINE_AA)
    cv2.putText(display, legend, (10, dh - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 255, 200), 1, cv2.LINE_AA)

    return display


def _build_debug_overlay(frame: np.ndarray) -> np.ndarray:
    """
    Overlay de debug sobre el frame original (antes del warp).
    Refresca la detección cada _DEBUG_REFRESH segundos para no bloquear la UI.

    Dibuja:
      - Azul claro / delgado : todos los contornos candidatos con área > 5% del frame
      - Verde / grueso       : contorno finalmente elegido como tablero
      - Círculos amarillos   : 4 corners numerados 0-3 en orden TL, TR, BR, BL
      - Texto inferior       : estrategia usada, área y n_corners del elegido
    """
    debug = frame.copy()
    h, w  = debug.shape[:2]
    img_area = h * w

    # --- Contornos candidatos: Canny básico, solo para visualización ---
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kern  = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kern, iterations=1)

    contours_all, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    min_cand_area = 0.05 * img_area
    candidates = [c for c in contours_all if cv2.contourArea(c) > min_cand_area]
    cv2.drawContours(debug, candidates, -1, (255, 200, 50), 1)   # azul claro, delgado

    # --- Contorno elegido (con caché temporal) ---
    now = time.time()
    if now - _debug_state["last_update"] > _DEBUG_REFRESH:
        corners, strategy = _board_detector._detect_board_corners(frame)
        _debug_state.update({
            "corners":     corners,
            "strategy":    strategy,
            "last_update": now,
        })
    else:
        corners  = _debug_state["corners"]
        strategy = _debug_state["strategy"]

    chosen_area = 0
    n_corners   = 0
    cal_active  = _calibration["corners"] is not None

    if corners is not None:
        ordered = _board_detector._ordenar_esquinas(corners.astype(np.float32))
        pts_int = ordered.astype(np.int32).reshape((-1, 1, 2))

        # Polígono del contorno elegido en verde grueso
        cv2.polylines(debug, [pts_int], isClosed=True,
                      color=(0, 255, 0), thickness=3)

        chosen_area = int(cv2.contourArea(pts_int))
        n_corners   = 4

        # Naranja si calibrado, amarillo si detección live
        _CORNER_COLOR = (0, 165, 255) if cal_active else (0, 255, 255)
        for i, pt in enumerate(ordered):
            cx, cy = int(pt[0]), int(pt[1])
            cv2.circle(debug, (cx, cy), 15, (0, 0, 0),      -1)   # aro negro
            cv2.circle(debug, (cx, cy), 12, _CORNER_COLOR,  -1)   # relleno
            cv2.putText(debug, str(i),
                        (cx - 5, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    # --- Texto de estado ---
    _STRATEGY_LABELS = {
        "chessboard_corners": "Est.1:findChessboard",
        "hough_lines":        "Est.2:HoughLines",
        "contour_improved":   "Est.3:Contorno",
    }
    strat_txt = _STRATEGY_LABELS.get(strategy, "Sin deteccion")

    if corners is not None:
        info_txt = (f"DEBUG  [{strat_txt}]  "
                    f"area={chosen_area}  n_corners={n_corners}")
    else:
        info_txt = "DEBUG  Sin contorno elegido"

    _put_text_outlined(debug, info_txt,
                       (14, h - 52), scale=0.60,
                       color=(0, 255, 255), thickness=2)

    # Indicador activo en zona superior
    _put_text_outlined(debug, "[ D ] DEBUG ON",
                       (14, 72), scale=0.65,
                       color=(0, 255, 255), thickness=2)

    # Indicador de calibración (solo cuando está activa)
    if cal_active:
        _put_text_outlined(debug, "DEBUG [CALIBRADO con Est.1]",
                           (14, h - 80), scale=0.60,
                           color=(255, 255, 0), thickness=2)   # cian BGR

    # --- Borde rojo detectado en cian (solo cuando tracking ON) ---
    # Cian = (255, 255, 0) en BGR. Muestra "lo que ve el detector" vs corners calibrados.
    if _tracking_state["tracking_status"].startswith("ON"):
        red_corners = _tracking_state["_red_border_corners"]
        if red_corners is not None:
            pts_cian = red_corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(debug, [pts_cian], isClosed=True,
                          color=(255, 255, 0), thickness=2)   # cian BGR
            for i, pt in enumerate(red_corners):
                rx, ry = int(pt[0]), int(pt[1])
                cv2.circle(debug, (rx, ry), 9,  (255, 255, 0), -1)   # relleno cian
                cv2.circle(debug, (rx, ry), 9,  (0, 0, 0),     1)    # aro negro
                cv2.putText(debug, str(i), (rx - 5, ry + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)

    return debug


# ---------------------------------------------------------------------------
# Modo B: ocupación en vivo — frame-diff vs tablero vacío de referencia
# ---------------------------------------------------------------------------

def _b_compute_occupancy(warped_cur: np.ndarray,
                          empty_warped: np.ndarray,
                          threshold: int) -> tuple:
    """
    Detector de ocupación de dos etapas:

    Etapa 1 (por píxel): |cur_gray - ref_gray| > 15 → píxel "cambiado"
                          (umbral fijo, no expuesto al usuario).
    Etapa 2 (por celda): change_pct = % de píxeles cambiados en región 60% central.
                          Ocupada si change_pct > threshold (threshold en %).
    Equipo: HSV sobre la misma región 60%, filtrada por changed_mask para aislar
            el color del objeto del fondo de la casilla.

    Returns:
        (occ_grid 8×8 bool, team_grid 8×8 str, change_pcts 8×8 float, count int)
        change_pcts: porcentaje 0.0–100.0 de píxeles cambiados por celda.
        team values: "EMPTY" | "RED" | "GREEN" | "UNKNOWN"
    """
    _PX_THRESH = 15   # umbral per-píxel fijo (no configurable)

    size = warped_cur.shape[0]   # 800 px
    cell = size // 8             # 100 px

    gray_cur = cv2.cvtColor(warped_cur,   cv2.COLOR_BGR2GRAY).astype(np.int16)
    gray_ref = cv2.cvtColor(empty_warped, cv2.COLOR_BGR2GRAY).astype(np.int16)
    if gray_ref.shape != gray_cur.shape:
        gray_ref = cv2.resize(gray_ref.astype(np.uint8),
                              (gray_cur.shape[1], gray_cur.shape[0])).astype(np.int16)

    occ_grid   = []
    team_grid  = []
    pcts       = []
    count      = 0
    margin_occ = max(1, int(cell * 0.20))   # 60% central

    for row in range(8):
        row_o = []
        row_t = []
        row_p = []
        for col in range(8):
            y0, y1 = row * cell, (row + 1) * cell
            x0, x1 = col * cell, (col + 1) * cell
            cy0, cy1 = y0 + margin_occ, y1 - margin_occ
            cx0, cx1 = x0 + margin_occ, x1 - margin_occ

            # --- Etapa 1: máscara de píxeles cambiados ---
            pixel_diff   = np.abs(gray_cur[cy0:cy1, cx0:cx1]
                                  - gray_ref[cy0:cy1, cx0:cx1])
            changed_mask = pixel_diff > _PX_THRESH           # bool (H, W)

            # --- Etapa 2: porcentaje de píxeles cambiados ---
            change_pct = 100.0 * changed_mask.mean()
            occ        = change_pct > threshold
            row_o.append(occ)
            row_p.append(change_pct)
            if occ:
                count += 1

            # --- Equipo (solo si ocupada) ---
            if occ:
                crop = warped_cur[cy0:cy1, cx0:cx1]
                if crop.size == 0:
                    row_t.append("EMPTY")
                else:
                    hsv    = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    chg_u8 = changed_mask.astype(np.uint8) * 255   # 0 | 255

                    # Solo contar píxeles de color DENTRO de la máscara cambiada
                    mask_red = cv2.bitwise_and(
                        cv2.inRange(hsv, (0,   80, 50), (10,  255, 255)) |
                        cv2.inRange(hsv, (165, 80, 50), (180, 255, 255)),
                        chg_u8
                    )
                    mask_green = cv2.bitwise_and(
                        cv2.inRange(hsv, (35, 70, 40), (85, 255, 255)),
                        chg_u8
                    )
                    red_count   = cv2.countNonZero(mask_red)
                    green_count = cv2.countNonZero(mask_green)

                    if red_count > 100 and red_count > green_count * 1.5:
                        row_t.append("RED")
                    elif green_count > 100 and green_count > red_count * 1.5:
                        row_t.append("GREEN")
                    else:
                        row_t.append("EMPTY")
            else:
                row_t.append("EMPTY")

        occ_grid.append(row_o)
        team_grid.append(row_t)
        pcts.append(row_p)

    return occ_grid, team_grid, pcts, count


def _b_build_display(warped: np.ndarray,
                      grid: list,
                      team_grid: list,
                      diffs: list,
                      count: int,
                      threshold: int) -> np.ndarray:
    """
    Construye la imagen de la ventana de ocupación en vivo (modo B).
    Tint por equipo + label R/G/? + notación algebraica + diff + HUD.
    # TODO: orientación asumida con blancas abajo (a1 = bottom-left).
    """
    display  = warped.copy()
    h, w     = display.shape[:2]
    cell_h   = h // 8
    cell_w   = w // 8
    cols     = list("abcdefgh")

    # Contadores por equipo
    n_red     = sum(team_grid[r][c] == "RED"     for r in range(8) for c in range(8))
    n_green   = sum(team_grid[r][c] == "GREEN"   for r in range(8) for c in range(8))
    n_unknown = sum(team_grid[r][c] == "UNKNOWN" for r in range(8) for c in range(8))
    n_empty   = 64 - count

    # Colores tint BGR
    _TINT = {
        "EMPTY":   (180, 180, 180),   # gris claro
        "RED":     (50,  50,  220),   # rojo
        "GREEN":   (50,  200, 50),    # verde
        "UNKNOWN": (50,  220, 220),   # amarillo
    }
    _TEAM_LBL = {"RED": "R", "GREEN": "G", "UNKNOWN": "?", "EMPTY": ""}

    # --- Tint semitransparente (alpha 0.30) ---
    overlay = display.copy()
    for row in range(8):
        for col in range(8):
            x0, y0 = col * cell_w, row * cell_h
            color   = _TINT[team_grid[row][col]]
            cv2.rectangle(overlay, (x0, y0), (x0 + cell_w, y0 + cell_h), color, -1)
    cv2.addWeighted(overlay, 0.30, display, 0.70, 0, display)

    # --- Grilla 8×8 ---
    for i in range(9):
        cv2.line(display, (i * cell_w, 0),     (i * cell_w, h),     (255, 255, 255), 1)
        cv2.line(display, (0,     i * cell_h), (w,     i * cell_h), (255, 255, 255), 1)

    # --- Texto por celda ---
    for row in range(8):
        for col in range(8):
            rank     = 8 - row       # fila 0 del warp = rank 8
            sq       = f"{cols[col]}{rank}"
            x0       = col * cell_w
            y0       = row * cell_h
            team     = team_grid[row][col]
            diff_txt = str(int(round(diffs[row][col])))

            # diff — esquina superior izquierda
            cv2.putText(display, diff_txt, (x0 + 4, y0 + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0),       2, cv2.LINE_AA)
            cv2.putText(display, diff_txt, (x0 + 4, y0 + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv2.LINE_AA)

            # notación algebraica — esquina inferior izquierda
            cv2.putText(display, sq, (x0 + 4, y0 + cell_h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 0),       2, cv2.LINE_AA)
            cv2.putText(display, sq, (x0 + 4, y0 + cell_h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1, cv2.LINE_AA)

            # label de equipo — centro de celda (solo si ocupada)
            lbl = _TEAM_LBL.get(team, "")
            if lbl:
                lx = x0 + cell_w // 2 - 7
                ly = y0 + cell_h // 2 + 7
                cv2.putText(display, lbl, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0),       3, cv2.LINE_AA)
                cv2.putText(display, lbl, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # --- HUD superior ---
    title = (f"OCUPACION  |  Vacias: {n_empty}  |  "
             f"R: {n_red}  |  G: {n_green}  |  ?: {n_unknown}  |  Umbral: {threshold}%")
    _put_text_outlined(display, title, (10, 30), scale=0.52, color=_COLOR_YELLOW)

    # --- HUD inferior ---
    bar_h = 28
    cv2.rectangle(display, (0, h - bar_h), (w, h), (30, 30, 30), -1)
    _put_text_outlined(display, "+ / - : ajustar umbral  |  B: salir modo",
                       (10, h - 8), scale=0.45, color=_WHITE)

    return display


# ---------------------------------------------------------------------------
# Helpers para el modo foto (tecla P)
# ---------------------------------------------------------------------------

def _p_short(s: str) -> str:
    """Reduce un status string a 'OK', 'FALLO' o '?'."""
    if s.startswith("OK"):
        return "OK"
    if "FALLO" in s or "ERROR" in s:
        return "FALLO"
    return "?"


def _p_draw_grid_overlay(warped: np.ndarray) -> np.ndarray:
    """
    Dibuja la grilla 8×8 sobre el tablero aplanado con notación algebraica.
    # TODO: asumimos blancas abajo (a1=bottom-left desde perspectiva blancas).
    #       Si la cámara está del lado de las negras las etiquetas estarán
    #       invertidas — ajustar cuando se determine la orientación real.
    """
    h, w = warped.shape[:2]
    cell_h = h / 8
    cell_w = w / 8
    cols = list("abcdefgh")

    for i in range(9):
        xi = int(round(i * cell_w))
        yi = int(round(i * cell_h))
        cv2.line(warped, (xi, 0), (xi, h), (255, 255, 255), 1)
        cv2.line(warped, (0, yi), (w, yi), (255, 255, 255), 1)

    for row in range(8):
        for col in range(8):
            rank = 8 - row          # fila 0 del warp = rank 8 (fila de las negras)
            sq   = f"{cols[col]}{rank}"
            xt   = int(round(col * cell_w)) + 4
            yt   = int(round(row * cell_h)) + 18
            cv2.putText(warped, sq, (xt + 1, yt + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(warped, sq, (xt, yt),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return warped


def _p_save_cells(warped: np.ndarray, cells_dir: str):
    """
    Recorta las 64 celdas del tablero aplanado y las guarda como <casilla>.jpg.
    # TODO: mismo supuesto de orientación que _p_draw_grid_overlay.
    """
    os.makedirs(cells_dir, exist_ok=True)
    h, w = warped.shape[:2]
    cell_h = h / 8
    cell_w = w / 8
    cols = list("abcdefgh")

    for row in range(8):
        for col in range(8):
            rank = 8 - row
            sq   = f"{cols[col]}{rank}"
            y0   = int(round(row * cell_h))
            y1   = int(round((row + 1) * cell_h))
            x0   = int(round(col * cell_w))
            x1   = int(round((col + 1) * cell_w))
            cv2.imwrite(os.path.join(cells_dir, f"{sq}.jpg"), warped[y0:y1, x0:x1])


def _p_draw_yolo(warped: np.ndarray, raw_detections: list) -> np.ndarray:
    """Dibuja bounding boxes de YOLO sobre el tablero aplanado."""
    if not raw_detections:
        cv2.putText(warped, "YOLO: 0 detecciones", (18, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, _BLACK, 4, cv2.LINE_AA)
        cv2.putText(warped, "YOLO: 0 detecciones", (18, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, _COLOR_RED, 2, cv2.LINE_AA)
        return warped

    for det in raw_detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cls   = det["class"]
        conf  = det["confidence"]
        label = f"{CLASS_TO_FEN.get(cls, '?')} {conf:.2f}"
        color = _COLOR_WHITE if cls.startswith("white_") else _COLOR_BLACK
        cv2.rectangle(warped, (x1, y1), (x2, y2), color, 2)
        text_y = max(y1 - 4, 14)
        cv2.putText(warped, label, (x1 + 2, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _BLACK, 3, cv2.LINE_AA)
        cv2.putText(warped, label, (x1 + 2, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return warped


def _photo_analysis_thread(frame: np.ndarray, cal_corners: np.ndarray,
                            out_dir: str, ts: str):
    """
    Análisis end-to-end disparado por tecla P.
    Cada etapa tiene try/except independiente: un fallo no aborta las siguientes.
    Escribe 8 artefactos a out_dir y actualiza el HUD al finalizar.
    """
    os.makedirs(os.path.join(out_dir, "04_celdas"), exist_ok=True)

    status       = {}
    warped       = None
    pieces_dict  = {}
    raw_dets     = []
    fen          = None

    # 01 — Frame original -------------------------------------------------------
    try:
        cv2.imwrite(os.path.join(out_dir, "01_frame_original.jpg"), frame)
    except Exception as e:
        print(f"[P] 01_frame_original: FALLO — {e}")

    # 02 — Tablero aplanado con matriz calibrada (sin re-detectar) --------------
    try:
        warped = _board_detector._warp_from_corners(frame, cal_corners)
        cv2.imwrite(os.path.join(out_dir, "02_tablero_aplanado.jpg"), warped)
        status["warp"] = "OK"
    except Exception as e:
        status["warp"] = f"FALLO: {e}"
        print(f"[P] 02_tablero_aplanado: FALLO — {e}")

    # 03 — Grid overlay ---------------------------------------------------------
    if warped is not None:
        try:
            cv2.imwrite(os.path.join(out_dir, "03_grid_overlay.jpg"),
                        _p_draw_grid_overlay(warped.copy()))
            status["grid"] = "OK"
        except Exception as e:
            status["grid"] = f"FALLO: {e}"
            print(f"[P] 03_grid_overlay: FALLO — {e}")
    else:
        status["grid"] = "NO_EJECUTADO: warp fallido"

    # 04 — Celdas individuales --------------------------------------------------
    if warped is not None:
        try:
            _p_save_cells(warped, os.path.join(out_dir, "04_celdas"))
            status["celdas"] = "OK"
        except Exception as e:
            status["celdas"] = f"FALLO: {e}"
            print(f"[P] 04_celdas: FALLO — {e}")
    else:
        status["celdas"] = "NO_EJECUTADO: warp fallido"

    # 05 — YOLO -----------------------------------------------------------------
    if warped is not None:
        try:
            pieces_dict, raw_dets = _classifier.detect_on_board(warped)
            cv2.imwrite(os.path.join(out_dir, "05_yolo_resultado.jpg"),
                        _p_draw_yolo(warped.copy(), raw_dets))
            status["yolo"] = f"OK ({len(raw_dets)} dets)"
        except Exception as e:
            status["yolo"] = f"FALLO: {e}"
            print(f"[P] 05_yolo_resultado: FALLO — {e}")
    else:
        status["yolo"] = "NO_EJECUTADO: warp fallido"

    # 06 — FEN ------------------------------------------------------------------
    fen_path = os.path.join(out_dir, "06_fen_intentado.txt")
    if pieces_dict:
        try:
            fen = _classifier.build_fen(pieces_dict)
            valid, reason = _validate_fen(fen)
            with open(fen_path, "w", encoding="utf-8") as f:
                f.write(fen + "\n")
                if not valid:
                    f.write(f"# ADVERTENCIA: FEN invalido — {reason}\n")
            status["fen"] = "OK" if valid else f"OK (invalido: {reason})"
            if not valid:
                fen = None   # no pasar a Stockfish
        except Exception as e:
            fen = None
            status["fen"] = f"FALLO: {e}"
            print(f"[P] 06_fen: FALLO — {e}")
            with open(fen_path, "w", encoding="utf-8") as f:
                f.write(f"FEN_ERROR: {e}\n")
    else:
        status["fen"] = "FALLO: sin detecciones YOLO"
        with open(fen_path, "w", encoding="utf-8") as f:
            f.write("FEN_ERROR: YOLO no produjo piezas detectadas\n")

    # 07 — Stockfish ------------------------------------------------------------
    sf_path = os.path.join(out_dir, "07_mejor_jugada.txt")
    if fen:
        try:
            _engine.set_position(fen)
            uci, san = _engine.get_best_move()
            with open(sf_path, "w", encoding="utf-8") as f:
                f.write(f"Mejor jugada: {san} ({uci})\n")
                f.write(f"FEN: {fen}\n")
            status["stockfish"] = f"OK: {san}"
        except Exception as e:
            status["stockfish"] = f"FALLO: {e}"
            print(f"[P] 07_stockfish: FALLO — {e}")
            with open(sf_path, "w", encoding="utf-8") as f:
                f.write(f"FALLO: {e}\n")
    else:
        status["stockfish"] = "NO_EJECUTADO: FEN invalido"
        with open(sf_path, "w", encoding="utf-8") as f:
            f.write("NO_EJECUTADO: FEN invalido\n")

    # 08 — Resumen --------------------------------------------------------------
    try:
        with open(os.path.join(out_dir, "08_resumen.txt"), "w", encoding="utf-8") as f:
            f.write(f"Captura P: {ts}\n")
            f.write(f"Directorio: {out_dir}\n\n")
            for stage, result in status.items():
                f.write(f"  {stage:<10}: {result}\n")
    except Exception as e:
        print(f"[P] 08_resumen: FALLO — {e}")

    # Actualizar HUD ------------------------------------------------------------
    parts = [
        f"warp={_p_short(status.get('warp',       '?'))}",
        f"grid={_p_short(status.get('grid',       '?'))}",
        f"yolo={_p_short(status.get('yolo',       '?'))}",
        f"fen={_p_short( status.get('fen',        '?'))}",
        f"sf={_p_short(  status.get('stockfish',  '?'))}",
    ]
    final_msg = f"P: capturas/captura_{ts}/ — {', '.join(parts)}"
    fail      = any("FALLO" in v for v in status.values())

    with _state["lock"]:
        _state["msg_text"]   = final_msg
        _state["msg_color"]  = _COLOR_RED if fail else _COLOR_GREEN
        _state["msg_expire"] = time.time() + 3.0

    print(f"[P] Analisis completo guardado en: {out_dir}")
    for stage, result in status.items():
        print(f"  {stage}: {result}")


# ---------------------------------------------------------------------------
# Dashboard web — funciones de soporte
# ---------------------------------------------------------------------------

def _mqtt_publish(topic: str, payload: str):
    """Publica en MQTT si el cliente está conectado, silencia errores."""
    if _mqtt_client is not None:
        try:
            _mqtt_client.publish(topic, payload)
        except Exception:
            pass


def _start_mqtt():
    """Conecta al broker MQTT local y suscribe al tópico de comandos."""
    global _mqtt_client
    if not _MQTT_AVAILABLE:
        print("[MQTT] paho-mqtt no disponible — instalar: pip install paho-mqtt")
        return
    try:
        client = _paho_mqtt.Client(client_id="aura_camera")

        def _on_message(c, userdata, msg):
            try:
                data = json.loads(msg.payload)
                cmd  = data.get("cmd", "")
                if cmd in ("calibrate", "analyze", "photo"):
                    _command_queue.put(cmd)
            except Exception:
                pass

        client.on_message = _on_message
        client.connect(_MQTT_BROKER, _MQTT_PORT, keepalive=60)
        client.subscribe("aura/control/command")
        client.loop_start()
        _mqtt_client = client
        print(f"[MQTT] Conectado a {_MQTT_BROKER}:{_MQTT_PORT}")
    except Exception as exc:
        print(f"[MQTT] No se pudo conectar: {exc}  — dashboard sin MQTT.")


class _MjpegHandler(BaseHTTPRequestHandler):
    """Sirve el feed de cámara como MJPEG en http://localhost:{_MJPEG_PORT}/"""
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                with _mjpeg_lock:
                    data = _mjpeg_buf[0]
                if data:
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                        + data + b"\r\n"
                    )
                    self.wfile.flush()
                time.sleep(0.033)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def log_message(self, *args):
        pass  # silenciar logs de acceso


def _start_mjpeg_server(port: int = _MJPEG_PORT):
    server = ThreadingHTTPServer(("0.0.0.0", port), _MjpegHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[MJPEG] Servidor de video en http://0.0.0.0:{port}/")


def _maybe_publish_board(team_grid):
    """
    Filtra ruido (UNKNOWN transitorio) con buffer de estabilidad.
    Solo publica aura/board/state cuando el estado se mantiene >= _STABLE_FRAMES frames.
    """
    global _last_pub_board

    stable_board = [[None] * 8 for _ in range(8)]
    for r in range(8):
        for c in range(8):
            cell = team_grid[r][c]     # "RED" | "GREEN" | "EMPTY" | "UNKNOWN"
            buf  = _stable_buf[r][c]
            if cell == buf["state"]:
                buf["count"] = min(buf["count"] + 1, _STABLE_FRAMES)
            else:
                buf["state"] = cell
                buf["count"] = 1
            # Publicar estado sólo si es estable y no UNKNOWN
            if buf["count"] >= _STABLE_FRAMES and cell != "UNKNOWN":
                stable_board[r][c] = cell
            else:
                stable_board[r][c] = "EMPTY"

    if stable_board != _last_pub_board:
        _last_pub_board = [row[:] for row in stable_board]
        _mqtt_publish("aura/board/state",
                      json.dumps({"state": stable_board}))


# ---------------------------------------------------------------------------
# Hilo del pipeline
# ---------------------------------------------------------------------------

def _pipeline_thread(image_path: str):
    """Ejecuta run_pipeline() en segundo plano y actualiza _state."""
    result = run_pipeline(image_path)

    san, uci = "", ""
    with _state["lock"]:
        _state["running"] = False
        _state["result"]  = result

        if result["success"]:
            san = result.get("best_move_san", "")
            uci = result.get("best_move_uci", "")
            _state["msg_text"]      = f"Ultima jugada: {san} ({uci})"
            _state["msg_color"]     = _COLOR_GREEN
            _state["pending_board"] = True
        else:
            err = result.get("error") or "Error desconocido"
            if len(err) > 90:
                err = err[:87] + "..."
            _state["msg_text"]  = f"Error: {err}"
            _state["msg_color"] = _COLOR_RED

        _state["msg_expire"] = time.time() + MSG_DURATION

    # Publicar jugada sugerida al dashboard (fuera del lock)
    if result["success"] and uci:
        _mqtt_publish("aura/board/bestmove",
                      json.dumps({"move": uci, "san": san}))


# DEPRECATED — reemplazado por modo B en vivo. Mantenido como referencia.
def _legacy_occupancy_basic_thread(image_path: str, reference_path: str):
    """[DEPRECATED] Ejecuta run_basic_pipeline() en segundo plano."""
    result = run_basic_pipeline(image_path, reference_path)

    with _state["lock"]:
        _state["basic_running"] = False
        _state["basic_result"]  = result

        if result["success"]:
            count = result.get("occupied_count", 0)
            _state["msg_text"]  = f"Ocupacion: {count}/64 casillas"
            _state["msg_color"] = _COLOR_YELLOW
            _state["basic_pending"] = True
        else:
            err = result.get("error") or "Error desconocido"
            if len(err) > 90:
                err = err[:87] + "..."
            _state["msg_text"]  = f"Error: {err}"
            _state["msg_color"] = _COLOR_RED

        _state["msg_expire"] = time.time() + MSG_DURATION


# ---------------------------------------------------------------------------
# Inicialización de cámara
# ---------------------------------------------------------------------------

def _scan_cameras(max_index: int = 5) -> list[int]:
    """Prueba índices 0..max_index y devuelve los que están disponibles."""
    available = []
    print("[Cámara] Escaneando cámaras disponibles...")
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(idx)
                print(f"  Camara {idx}: disponible")
        cap.release()
    if not available:
        print("  (no se encontró ninguna cámara)")
    return available


def _open_camera():
    """
    Detecta las cámaras disponibles, intenta identificar la Logitech C920e
    por nombre y, si no es posible, pregunta al usuario el índice.
    Devuelve (cap, index, label).
    """
    available = _scan_cameras()
    if not available:
        raise RuntimeError("No se encontró ninguna cámara en los índices 0 al 5.")

    # --- Intentar identificar la C920e por nombre de backend ---
    chosen_idx = None
    for idx in available:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        backend_name = cap.getBackendName() if hasattr(cap, "getBackendName") else ""
        backend_code = int(cap.get(cv2.CAP_PROP_BACKEND))
        cap.release()
        print(f"  Camara {idx}: backend={backend_name!r}  (código {backend_code})")

    # OpenCV no expone el nombre del dispositivo directamente; si solo hay una
    # cámara disponible la seleccionamos automáticamente, si hay varias pedimos
    # al usuario que elija.
    if len(available) == 1:
        chosen_idx = available[0]
        print(f"[Cámara] Solo hay una cámara disponible → usando índice {chosen_idx}.")
    else:
        print("\n[Cámara] No se pudo identificar automáticamente la C920e.")
        print("         Ingrese el número de índice correspondiente a la C920e:")
        while chosen_idx is None:
            try:
                val = int(input("  >>> ").strip())
                if val in available:
                    chosen_idx = val
                else:
                    print(f"  Índice {val} no está en la lista de disponibles {available}. "
                          "Intente nuevamente.")
            except ValueError:
                print("  Ingrese un número entero.")

    # --- Abrir y configurar la C920e ---
    cap = cv2.VideoCapture(chosen_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara en índice {chosen_idx}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_AUTOFOCUS,    1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Cámara] C920e abierta en índice {chosen_idx}  ({actual_w}x{actual_h})")

    return cap, chosen_idx, "C920e"


# ---------------------------------------------------------------------------
# Bucle principal
# ---------------------------------------------------------------------------

def main():
    print("[camera_pipeline] Iniciando — esto puede tardar unos segundos "
          "mientras el pipeline carga los modelos...")
    _start_mqtt()
    _start_mjpeg_server()
    cap, cam_index, cam_label = _open_camera()

    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_MAIN, TARGET_W, TARGET_H)

    board_display  = None    # última imagen del tablero completo (2ª ventana)
    basic_display  = None    # última imagen del modo básico (3ª ventana)
    pipeline_start = None    # time.time() de cuando arrancó el pipeline

    print("[camera_pipeline] Listo. ESPACIO: analisis completo | R: guardar referencia | B: detectar ocupacion | Q: salir")
    _prev_tracking_status = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[camera_pipeline] Advertencia: frame inválido, reintentando...")
            time.sleep(0.05)
            continue

        # Redimensionar si la cámara no soportó la resolución objetivo
        fh, fw = frame.shape[:2]
        if fw != TARGET_W or fh != TARGET_H:
            frame = cv2.resize(frame, (TARGET_W, TARGET_H))

        # ---- Leer estado compartido (copia local para no mantener el lock) ----
        with _state["lock"]:
            running       = _state["running"]
            msg_text      = _state["msg_text"]
            msg_color     = _state["msg_color"]
            msg_expire    = _state["msg_expire"]
            pending       = _state["pending_board"]
            result        = _state["result"]
            basic_running = _state["basic_running"]
            basic_pending = _state["basic_pending"]
            basic_result  = _state["basic_result"]

            if pending:
                _state["pending_board"] = False   # consumir la señal
            if basic_pending:
                _state["basic_pending"] = False   # consumir la señal

        # ---- Detectar timeout del pipeline ----
        if running and pipeline_start is not None:
            if time.time() - pipeline_start > PIPELINE_TIMEOUT:
                with _state["lock"]:
                    if _state["running"]:   # verificar que no terminó justo ahora
                        _state["running"]   = False
                        _state["msg_text"]  = (
                            f"Timeout: el pipeline tardó más de "
                            f"{PIPELINE_TIMEOUT:.0f}s"
                        )
                        _state["msg_color"]  = _COLOR_RED
                        _state["msg_expire"] = time.time() + MSG_DURATION
                running = False

        # ---- Construir imagen del tablero completo si hay resultado nuevo ----
        if pending and result is not None and result["success"]:
            board_display = _build_board_display(result)

        # ---- Construir imagen del modo básico si hay resultado nuevo ----
        if basic_pending and basic_result is not None and basic_result["success"]:
            basic_display = _build_basic_display(
                basic_result["_warped_board"],
                basic_result["occupied_grid"],
                basic_result["occupied_count"],
                basic_result.get("cell_metrics"),
            )

        # ---- Tracking por borde rojo continuo (antes del debug overlay y modo B) ----
        _update_red_border_tracking(frame)
        _cur_tracking = _tracking_state["tracking_status"]
        if _cur_tracking != _prev_tracking_status:
            _prev_tracking_status = _cur_tracking
            _mqtt_publish("aura/system/status",
                          json.dumps({"tracking": _cur_tracking}))

        # ---- Debug overlay (antes del HUD, sobre frame limpio de cámara) ----
        if _debug_state["active"]:
            frame = _build_debug_overlay(frame)

        # ---- Modo B: ocupación en vivo (frame-diff vs tablero vacío) ----
        if _mode_b["active"] and _calibration["matrix"] is not None \
                              and _calibration["empty_warped"] is not None:
            try:
                _bsz = 800
                _bm  = int(_bsz * 0.05)
                warped_b = cv2.warpPerspective(frame, _calibration["matrix"], (_bsz, _bsz))
                warped_b = warped_b[_bm:_bsz - _bm, _bm:_bsz - _bm]
                warped_b = cv2.resize(warped_b, (_bsz, _bsz), interpolation=cv2.INTER_AREA)
                _grid_b, _team_b, _pcts_b, _count_b = _b_compute_occupancy(
                    warped_b, _calibration["empty_warped"], _mode_b["threshold"]
                )
                _maybe_publish_board(_team_b)
                cv2.imshow(WINDOW_OCC, _b_build_display(
                    warped_b, _grid_b, _team_b, _pcts_b, _count_b, _mode_b["threshold"]
                ))
            except Exception as _b_err:
                pass   # no abortar el feed principal por error en modo B

        # ---- HUD ----
        _draw_hud(frame, cam_label, running, pipeline_start,
                  msg_text, msg_color, msg_expire,
                  tracking_status=_tracking_state["tracking_status"])

        cv2.imshow(WINDOW_MAIN, frame)

        # ---- Actualizar frame MJPEG para el dashboard ----
        _ok, _jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if _ok:
            with _mjpeg_lock:
                _mjpeg_buf[0] = _jpeg.tobytes()

        # ---- 2ª ventana con el tablero completo (si existe) ----
        if board_display is not None:
            cv2.imshow(WINDOW_BOARD, board_display)

        # ---- 3ª ventana con el modo básico (si existe) ----
        if basic_display is not None:
            cv2.imshow(WINDOW_BASIC, basic_display)

        # ---- Teclas ----
        key = cv2.waitKey(1) & 0xFF

        # Inyectar comandos del dashboard si no llegó tecla física
        if key == 0xFF:
            try:
                cmd = _command_queue.get_nowait()
                if   cmd == "calibrate": key = ord('R')
                elif cmd == "analyze":   key = ord(' ')
                elif cmd == "photo":     key = ord('P')
            except queue.Empty:
                pass

        if key in (ord('q'), ord('Q')):
            break

        elif key == ord(' '):
            with _state["lock"]:
                already_running = _state["running"]

            if already_running:
                print("[camera_pipeline] Pipeline ya en curso, ignorando captura.")
                continue

            # Guardar el frame actual
            save_ok = cv2.imwrite(CAPTURE_PATH, frame)
            if not save_ok:
                print(f"[camera_pipeline] Error: no se pudo guardar {CAPTURE_PATH}")
                continue
            print(f"[camera_pipeline] Frame guardado -> {CAPTURE_PATH}")

            # Lanzar el pipeline completo en un hilo daemon
            with _state["lock"]:
                _state["running"]       = True
                _state["msg_text"]      = ""
                _state["pending_board"] = False

            pipeline_start = time.time()
            t = threading.Thread(target=_pipeline_thread,
                                 args=(CAPTURE_PATH,), daemon=True)
            t.start()

        elif key in (ord('r'), ord('R')):
            # --- Pre-condición 1: detectar el borde rojo del tablero ---
            _corners_rojo = _detect_red_border_corners(frame)

            if _corners_rojo is None:
                print("[camera_pipeline] R: no se detecta el borde rojo del tablero — "
                      "verificar iluminacion")
                with _state["lock"]:
                    _state["msg_text"]   = ("R: no se detecta el borde rojo del tablero — "
                                            "verificar iluminacion")
                    _state["msg_color"]  = _COLOR_RED
                    _state["msg_expire"] = time.time() + 2.0

            else:
                # --- Pre-condición 2: tablero vacío + findChessboardCorners (Est.1) ---
                inner_7x7 = _find_inner_corners_for_calibration(frame)
                if inner_7x7 is None:
                    _calibration["corners"] = None
                    _calibration["matrix"]  = None
                    print("[camera_pipeline] R: calibracion fallida — "
                          "Est.1 no encontro el tablero")
                    with _state["lock"]:
                        _state["msg_text"]   = ("R: calibracion fallida — "
                                                "limpiar tablero y reintentar")
                        _state["msg_color"]  = _COLOR_RED
                        _state["msg_expire"] = time.time() + 2.0

                else:
                    # corners_precisos — extrapolación diagonal a esquinas exteriores 8×8
                    outer_corners    = extrapolate_outer_corners(inner_7x7)
                    corners_precisos = _board_detector._ordenar_esquinas(outer_corners)

                    # Expansión 2.5% hacia afuera desde el centroide: compensa el leve
                    # undershoot que se observa en el debug overlay (el warp corta el borde).
                    _centroid        = corners_precisos.mean(axis=0)
                    corners_precisos = (corners_precisos
                                        + 0.025 * (corners_precisos - _centroid)).astype(
                                            np.float32)

                    # Log en consola
                    inner_extremes = np.array([
                        inner_7x7[0, 0], inner_7x7[0, 6],
                        inner_7x7[6, 6], inner_7x7[6, 0],
                    ], dtype=np.float32)
                    print("[camera_pipeline] R: corners internos extremos "
                          "(pre-extrapol, 1 casilla adentro del borde):")
                    for lbl, pt in zip(
                        ["[0,0]", "[0,6]", "[6,6]", "[6,0]"], inner_extremes
                    ):
                        print(f"  inner {lbl}: ({pt[0]:.1f}, {pt[1]:.1f})")
                    print("[camera_pipeline] R: corners exteriores extrapolados "
                          "(esquinas reales del tablero 8×8):")
                    for lbl, pt in zip(["TL", "TR", "BR", "BL"], corners_precisos):
                        print(f"  outer  {lbl}: ({pt[0]:.1f}, {pt[1]:.1f})")
                    print("[camera_pipeline] R: corners del borde rojo detectado:")
                    for lbl, pt in zip(["TL", "TR", "BR", "BL"], _corners_rojo):
                        print(f"  rojo   {lbl}: ({pt[0]:.1f}, {pt[1]:.1f})")

                    # Homografía: corners del borde rojo → corners precisos del área de juego
                    # Captura la transformación completa entre borde físico y área activa,
                    # incluyendo distorsiones perspectivas entre el borde y las casillas.
                    H_red_to_board = cv2.getPerspectiveTransform(
                        _corners_rojo.astype(np.float32),
                        corners_precisos.astype(np.float32),
                    )
                    _tracking_state["red_to_board_homography"] = H_red_to_board
                    _tracking_state["last_board_corners"]      = corners_precisos

                    # Actualizar calibración con corners precisos
                    matrix = _board_detector._compute_perspective_matrix(corners_precisos)
                    _calibration["corners"]      = corners_precisos
                    _calibration["matrix"]       = matrix
                    _calibration["empty_warped"] = _board_detector._warp_from_corners(
                        frame, corners_precisos
                    )

                    cv2.imwrite(REFERENCE_PATH, frame)
                    print(f"[camera_pipeline] R: calibrado OK + tracking rojo activo — "
                          f"referencia guardada -> {REFERENCE_PATH}")

                    with _state["lock"]:
                        _state["msg_text"]   = "R: calibrado OK + tracking rojo activo"
                        _state["msg_color"]  = _COLOR_GREEN
                        _state["msg_expire"] = time.time() + 2.0

        elif key in (ord('p'), ord('P')):
            if _calibration["matrix"] is None or _calibration["corners"] is None:
                with _state["lock"]:
                    _state["msg_text"]   = "P: requiere calibracion previa con R"
                    _state["msg_color"]  = _COLOR_RED
                    _state["msg_expire"] = time.time() + 2.0
            else:
                ts      = time.strftime("%Y%m%d_%H%M%S")
                out_dir = os.path.join("capturas", f"captura_{ts}")
                # Copias locales para el hilo (evita race condition si R se pulsa durante el análisis)
                frame_copy   = frame.copy()
                corners_copy = _calibration["corners"].copy()

                t = threading.Thread(
                    target=_photo_analysis_thread,
                    args=(frame_copy, corners_copy, out_dir, ts),
                    daemon=True,
                )
                t.start()

                print(f"[camera_pipeline] P: análisis lanzado -> {out_dir}")
                with _state["lock"]:
                    _state["msg_text"]   = f"P: analizando -> capturas/captura_{ts}/ ..."
                    _state["msg_color"]  = _COLOR_YELLOW
                    _state["msg_expire"] = time.time() + MSG_DURATION

        elif key in (ord('b'), ord('B')):
            if _mode_b["active"]:
                # Toggle OFF: desactivar y cerrar ventana
                _mode_b["active"] = False
                cv2.destroyWindow(WINDOW_OCC)
                print("[camera_pipeline] Modo B: desactivado.")
            else:
                # Toggle ON: verificar pre-condición
                if _calibration["matrix"] is None or _calibration["empty_warped"] is None:
                    print("[camera_pipeline] Modo B: requiere calibracion previa con R.")
                    with _state["lock"]:
                        _state["msg_text"]   = "B: requiere calibracion previa con R"
                        _state["msg_color"]  = _COLOR_RED
                        _state["msg_expire"] = time.time() + 2.0
                else:
                    _mode_b["active"] = True
                    print(f"[camera_pipeline] Modo B: activado (umbral={_mode_b['threshold']}).")

        elif key in (ord('+'), ord('=')):   # = es + sin shift en teclados sin numpad
            if _mode_b["active"]:
                _mode_b["threshold"] = min(100, _mode_b["threshold"] + 2)
                print(f"[camera_pipeline] Modo B: umbral -> {_mode_b['threshold']}")

        elif key == ord('-'):
            if _mode_b["active"]:
                _mode_b["threshold"] = max(2, _mode_b["threshold"] - 2)
                print(f"[camera_pipeline] Modo B: umbral -> {_mode_b['threshold']}")

        elif key in (ord('d'), ord('D')):
            _debug_state["active"] = not _debug_state["active"]
            if not _debug_state["active"]:
                # Limpiar caché al desactivar para forzar re-detección al reactivar
                _debug_state.update({"corners": None, "strategy": None,
                                     "last_update": 0.0})
            estado = "ACTIVADO" if _debug_state["active"] else "desactivado"
            print(f"[camera_pipeline] Debug overlay {estado}.")

    # ---- Limpieza ----
    cap.release()
    cv2.destroyAllWindows()
    print("[camera_pipeline] Cerrado.")


if __name__ == "__main__":
    main()
