"""
calibration.py
Herramienta interactiva de calibración con cámara web en tiempo real.
Proyecto: Robot Ajedrecista - CAETI (UAI)

Uso:
    python calibration.py

Teclas:
    ESPACIO  →  Capturar frame actual → guardar "captured_frame.jpg" e imprimir FEN
    S        →  Guardar snapshot con timestamp en snapshots/
    Q        →  Salir
"""

import cv2
import numpy as np
import os
import sys
from datetime import datetime

from board_detector import BoardDetector


def crear_carpeta_snapshots() -> str:
    """Crea la carpeta snapshots/ si no existe y retorna su path."""
    carpeta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
    os.makedirs(carpeta, exist_ok=True)
    return carpeta


def dibujar_texto_fen(imagen: np.ndarray, fen: str | None) -> np.ndarray:
    """
    Superpone el FEN detectado sobre la imagen con fondo semitransparente.

    Args:
        imagen: Imagen BGR
        fen: String FEN o None si no se detectó tablero

    Returns:
        Imagen con el texto del FEN dibujado
    """
    resultado = imagen.copy()
    h, w = resultado.shape[:2]

    # Fondo oscuro semitransparente para el texto
    overlay = resultado.copy()
    cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, resultado, 0.4, 0, resultado)

    texto = fen if fen else "Tablero no detectado"
    color = (0, 255, 0) if fen else (0, 0, 255)

    # Escalar el texto para que entre en la imagen
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    escala = 0.5
    grosor = 1

    # Posición del texto
    cv2.putText(resultado, "FEN:", (10, h - 38), fuente, escala, (200, 200, 200), grosor)
    cv2.putText(resultado, texto, (10, h - 15), fuente, escala, color, grosor)

    return resultado


def dibujar_instrucciones(imagen: np.ndarray) -> np.ndarray:
    """Dibuja las instrucciones de teclas en la esquina superior derecha."""
    resultado = imagen.copy()
    h, w = resultado.shape[:2]

    instrucciones = [
        "ESPACIO: Capturar",
        "S: Snapshot",
        "Q: Salir",
    ]

    overlay = resultado.copy()
    cv2.rectangle(overlay, (w - 200, 0), (w, len(instrucciones) * 25 + 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, resultado, 0.4, 0, resultado)

    for i, texto in enumerate(instrucciones):
        cv2.putText(resultado, texto,
                    (w - 195, 20 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return resultado


def construir_grid_visual(detector: BoardDetector,
                          board_image: np.ndarray,
                          cells: list[str] | None) -> np.ndarray:
    """
    Construye la imagen del grid de 64 celdas para la ventana de debug.
    Si no hay celdas clasificadas, devuelve el board_image sin overlay.
    """
    if cells is not None:
        return detector.get_cells_grid_image(board_image, cells)
    return board_image


def main():
    detector = BoardDetector()
    carpeta_snapshots = crear_carpeta_snapshots()

    # Intentar abrir la cámara
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara (índice 0).")
        print("  Verificar que la cámara esté conectada y no esté en uso por otra aplicación.")
        sys.exit(1)

    # Ajustar resolución de captura (puede no ser compatible con todas las cámaras)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ancho_real = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto_real = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Cámara abierta. Resolución: {ancho_real}x{alto_real}")
    print("[INFO] Ventanas: 'Calibración', 'Tablero Aplanado', 'Grid Celdas'")
    print("[INFO] ESPACIO=capturar | S=snapshot | Q=salir")

    # Nombres de ventanas
    VENTANA_PRINCIPAL = "Calibración - Robot Ajedrecista (CAETI)"
    VENTANA_TABLERO   = "Tablero Aplanado"
    VENTANA_GRID      = "Grid de Celdas (64)"

    cv2.namedWindow(VENTANA_PRINCIPAL, cv2.WINDOW_NORMAL)
    cv2.namedWindow(VENTANA_TABLERO,   cv2.WINDOW_NORMAL)
    cv2.namedWindow(VENTANA_GRID,      cv2.WINDOW_NORMAL)

    # Posicionar las ventanas para que no se superpongan
    cv2.moveWindow(VENTANA_PRINCIPAL, 0, 0)
    cv2.moveWindow(VENTANA_TABLERO,   700, 0)
    cv2.moveWindow(VENTANA_GRID,      1300, 0)

    fen_actual = None
    board_image_actual = None
    cells_actual = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] No se pudo leer frame de la cámara.")
            continue

        # Procesar el frame con el detector
        resultado = detector.process_frame(frame)

        fen_actual = resultado["fen"]
        board_image_actual = resultado["board_image"]
        cells_actual = resultado["cells"]

        # Usar debug_image (frame con contorno verde) o el frame sin procesar
        imagen_display = resultado["debug_image"] if resultado["debug_image"] is not None else frame

        # Superponer FEN e instrucciones
        imagen_display = dibujar_texto_fen(imagen_display, fen_actual)
        imagen_display = dibujar_instrucciones(imagen_display)

        # Mostrar ventana principal
        cv2.imshow(VENTANA_PRINCIPAL, imagen_display)

        # Mostrar tablero aplanado (o imagen de placeholder si no se detectó)
        if board_image_actual is not None:
            cv2.imshow(VENTANA_TABLERO, board_image_actual)
            grid_img = construir_grid_visual(detector, board_image_actual, cells_actual)
            cv2.imshow(VENTANA_GRID, grid_img)
        else:
            # Mostrar imagen de "no detectado" en las ventanas secundarias
            placeholder = np.zeros((400, 400, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No detectado",
                        (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            cv2.imshow(VENTANA_TABLERO, placeholder)
            cv2.imshow(VENTANA_GRID, placeholder)

        # Manejo de teclas
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord('q') or tecla == ord('Q'):
            print("[INFO] Saliendo...")
            break

        elif tecla == ord(' '):
            # Capturar frame actual
            path_captura = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "captured_frame.jpg")
            cv2.imwrite(path_captura, frame)
            print(f"\n[CAPTURA] Frame guardado en: {path_captura}")
            if fen_actual:
                print(f"[FEN]     {fen_actual}")
            else:
                print("[FEN]     Tablero no detectado en este frame.")

        elif tecla == ord('s') or tecla == ord('S'):
            # Guardar snapshot con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_archivo = f"snapshot_{timestamp}.jpg"
            path_snapshot = os.path.join(carpeta_snapshots, nombre_archivo)
            cv2.imwrite(path_snapshot, frame)

            # También guardar el tablero aplanado si está disponible
            if board_image_actual is not None:
                nombre_board = f"board_{timestamp}.jpg"
                path_board = os.path.join(carpeta_snapshots, nombre_board)
                cv2.imwrite(path_board, board_image_actual)
                print(f"[SNAPSHOT] {path_snapshot} + {nombre_board}")
            else:
                print(f"[SNAPSHOT] {path_snapshot}")

            if fen_actual:
                print(f"[FEN]      {fen_actual}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
