"""
_aruco_fallback.py — Tracking ArUco (método anterior, conservado como referencia).
Robot Ajedrecista CAETI (UAI)

El tracking activo en camera_pipeline.py usa detección del borde rojo.
Este archivo conserva el código ArUco por si se desea volver a ese método.

Para reactivarlo:
  1. Copiar el bloque de inicialización y las funciones de aquí a camera_pipeline.py.
  2. Reemplazar _detect_red_border_corners / _update_red_border_tracking.
  3. Cambiar _tracking_state["red_to_board_homography"] → "markers_to_board_homography".
  4. Pegar los 4 marcadores ArUco (generate_aruco_markers.py) en las esquinas del tablero.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Inicialización del detector — compatibilidad OpenCV 4.6 / 4.7+
# ---------------------------------------------------------------------------
try:
    _aruco_dict     = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    _aruco_params   = cv2.aruco.DetectorParameters()
    _aruco_detector = cv2.aruco.ArucoDetector(_aruco_dict, _aruco_params)

    def _detect_aruco_markers(frame: np.ndarray):
        return _aruco_detector.detectMarkers(frame)

except AttributeError:
    # OpenCV < 4.7 — API legada
    _aruco_dict   = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    _aruco_params = cv2.aruco.DetectorParameters_create()

    def _detect_aruco_markers(frame: np.ndarray):
        return cv2.aruco.detectMarkers(frame, _aruco_dict, parameters=_aruco_params)


# ---------------------------------------------------------------------------
# Tracking ArUco — función de actualización por frame
# ---------------------------------------------------------------------------

def update_aruco_tracking(frame: np.ndarray,
                           tracking_state: dict,
                           calibration: dict,
                           board_detector) -> None:
    """
    Detecta marcadores ArUco (IDs 0-3) y actualiza calibration["matrix/corners"]
    si los 4 están presentes y la homografía está calibrada.

    Montaje en el tablero:
        ID 0 → TL   ID 1 → TR   ID 2 → BR   ID 3 → BL  (vista desde la cámara)

    tracking_state esperado:
        "markers_to_board_homography"  — H: centros marcadores → esquinas del tablero
        "last_board_corners"
        "tracking_status"              — "OFF" | "ON (4/4)" | "STALE (X/4)"
        "_detected_corners"            — lista para debug overlay
        "_detected_ids"                — lista para debug overlay
    """
    corners_raw, ids_raw, _ = _detect_aruco_markers(frame)

    id_to_center          = {}
    detected_corners_list = []
    detected_ids_list     = []

    if ids_raw is not None:
        for i, mid in enumerate(ids_raw.flatten()):
            mid = int(mid)
            if mid in (0, 1, 2, 3):
                c = corners_raw[i][0]
                id_to_center[mid] = c.mean(axis=0)
                detected_corners_list.append(corners_raw[i])
                detected_ids_list.append(mid)

    tracking_state["_detected_corners"] = detected_corners_list
    tracking_state["_detected_ids"]     = detected_ids_list

    n = len(id_to_center)

    if n == 4 and tracking_state.get("markers_to_board_homography") is not None:
        marker_pts = np.array([
            id_to_center[0], id_to_center[1],
            id_to_center[2], id_to_center[3],
        ], dtype=np.float32).reshape(4, 1, 2)

        H         = tracking_state["markers_to_board_homography"]
        board_pts = cv2.perspectiveTransform(marker_pts, H).reshape(4, 2)
        ordered   = board_detector._ordenar_esquinas(board_pts.astype(np.float32))

        calibration["corners"] = ordered
        calibration["matrix"]  = board_detector._compute_perspective_matrix(ordered)
        tracking_state["last_board_corners"] = ordered
        tracking_state["tracking_status"]    = "ON (4/4)"

    elif calibration.get("matrix") is not None:
        tracking_state["tracking_status"] = f"STALE ({n}/4)"
    else:
        tracking_state["tracking_status"] = "OFF"
