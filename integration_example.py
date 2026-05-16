"""
integration_example.py
Ejemplo de integración entre el módulo de visión (BoardDetector)
y el motor de ajedrez (ChessEngine) del proyecto principal.
Proyecto: Robot Ajedrecista - CAETI (UAI)

NOTA: Este archivo es solo documentación ejecutable.
      chess_engine.py está en otra carpeta del proyecto principal.
      Ajustar la ruta de importación según la estructura real del repositorio.
"""

import cv2
import sys

# ---------------------------------------------------------------------------
# PASO 0: Importar los módulos del proyecto
# ---------------------------------------------------------------------------

# Importar el detector de visión (este módulo)
from board_detector import BoardDetector

# Importar el motor de ajedrez (ajustar el path según la estructura del repo)
# Opción A: si ambos módulos están en la misma carpeta:
#   from chess_engine import ChessEngine
#
# Opción B: si chess_engine.py está en una carpeta hermana:
#   import sys, os
#   sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'chess_engine'))
#   from chess_engine import ChessEngine
#
# Opción C: si el proyecto usa un paquete instalado:
#   from robot_ajedrez.chess_engine import ChessEngine

# Para que este ejemplo sea ejecutable sin chess_engine.py, usamos un stub:
class ChessEngine:
    """Stub de ChessEngine para que el ejemplo se pueda ejecutar solo."""
    def get_best_move(self, fen: str) -> str:
        print(f"  [ChessEngine stub] Recibido FEN: {fen}")
        return "e2e4"  # Movimiento de ejemplo


# ---------------------------------------------------------------------------
# PASO 1: Inicializar los módulos
# ---------------------------------------------------------------------------

detector = BoardDetector()
engine   = ChessEngine()

print("=" * 60)
print("Ejemplo de integración: Visión → FEN → Motor de ajedrez")
print("=" * 60)


# ---------------------------------------------------------------------------
# PASO 2: Obtener el FEN desde una imagen estática
#         (en producción esto vendría de la cámara en tiempo real)
# ---------------------------------------------------------------------------

def obtener_fen_desde_imagen(path_imagen: str) -> str | None:
    """
    Carga una imagen, detecta el tablero y retorna el FEN.
    Retorna None si no se puede detectar el tablero.
    """
    frame = cv2.imread(path_imagen)
    if frame is None:
        print(f"[ERROR] No se pudo cargar la imagen: {path_imagen}")
        return None

    resultado = detector.process_frame(frame)

    if not resultado["success"]:
        print("[WARN] No se detectó el tablero en la imagen.")
        return None

    return resultado["fen"]


def obtener_fen_desde_camara(indice_camara: int = 0) -> str | None:
    """
    Captura un frame de la cámara web y retorna el FEN detectado.
    Intenta hasta 10 frames antes de rendirse.
    """
    cap = cv2.VideoCapture(indice_camara)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara con índice {indice_camara}.")
        return None

    fen = None
    intentos = 0
    max_intentos = 10

    while intentos < max_intentos:
        ret, frame = cap.read()
        if not ret:
            intentos += 1
            continue

        resultado = detector.process_frame(frame)
        if resultado["success"]:
            fen = resultado["fen"]
            break

        intentos += 1

    cap.release()

    if fen is None:
        print(f"[WARN] No se detectó el tablero luego de {max_intentos} intentos.")

    return fen


# ---------------------------------------------------------------------------
# PASO 3: Pasar el FEN al motor y obtener el mejor movimiento
# ---------------------------------------------------------------------------

def ciclo_completo(fen: str) -> str | None:
    """
    Dado un FEN, consulta al motor de ajedrez y retorna el mejor movimiento
    en notación UCI (ej: "e2e4", "g1f3", "e1g1" para enroque).

    Args:
        fen: Posición en formato FEN

    Returns:
        Movimiento en notación UCI, o None si el motor falla
    """
    print(f"\n[VISION]  FEN detectado: {fen}")

    try:
        movimiento = engine.get_best_move(fen)
        print(f"[ENGINE]  Mejor movimiento: {movimiento}")
        return movimiento
    except Exception as e:
        print(f"[ERROR]   El motor de ajedrez falló: {e}")
        return None


# ---------------------------------------------------------------------------
# PASO 4: Ejemplo en bucle (simula el robot esperando cada jugada)
# ---------------------------------------------------------------------------

def bucle_robot(usar_camara: bool = False, path_imagen: str | None = None):
    """
    Simula el bucle principal del robot ajedrecista:
      1. Capturar imagen del tablero (cámara o archivo)
      2. Detectar posición → FEN
      3. Calcular mejor movimiento
      4. (Aquí iría el código de control del brazo robótico)
      5. Esperar la respuesta del oponente
      6. Repetir
    """
    print("\n[ROBOT] Iniciando bucle principal...")

    if usar_camara:
        print("[ROBOT] Fuente: cámara web")
        fen = obtener_fen_desde_camara(indice_camara=0)
    elif path_imagen:
        print(f"[ROBOT] Fuente: imagen estática → {path_imagen}")
        fen = obtener_fen_desde_imagen(path_imagen)
    else:
        # FEN de ejemplo: posición inicial del ajedrez
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        print(f"[ROBOT] Fuente: FEN de ejemplo (posición inicial)")

    if fen is None:
        print("[ROBOT] No se pudo obtener el FEN. Abortando ciclo.")
        return

    movimiento = ciclo_completo(fen)

    if movimiento:
        print(f"\n[ROBOT] Ejecutar movimiento: {movimiento}")
        # AQUÍ: llamar al módulo de control del brazo robótico
        # por ejemplo: robot_arm.move(movimiento)
        print("[ROBOT] (En producción: enviar movimiento al brazo robótico)")
    else:
        print("[ROBOT] No se obtuvo movimiento. Revisar detección y motor.")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Si se pasa una imagen como argumento, usarla
    if len(sys.argv) > 1:
        bucle_robot(path_imagen=sys.argv[1])
    else:
        # Demostración con FEN hardcodeado (no requiere cámara ni imagen)
        bucle_robot()

    print("\n[INFO] Para usar con imagen real:")
    print("  python integration_example.py foto_tablero.jpg")
    print("\n[INFO] Para integrar con cámara, editar bucle_robot(usar_camara=True)")
