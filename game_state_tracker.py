"""
game_state_tracker.py — Modelo lógico del tablero y tracking de movimientos.
Robot Ajedrecista CAETI (UAI)

Responsabilidades principales
------------------------------
Este módulo mantiene el estado lógico de la partida independientemente del
módulo de visión. Recibe estados de tablero (grillas 8×8 de ocupación/color
producidas por camera_pipeline._b_compute_occupancy) y los traduce al modelo
de piezas con identidades concretas.

Flujo de uso previsto
---------------------
1. Lock-in inicial (tecla I en tracker_app.py):
   - Se toma el estado actual del tablero como snapshot de inicio.
   - Se asume posición estándar de ajedrez: las 16 piezas blancas (equipo RED
     o GREEN según la orientación física del tablero) se ubican en filas 1-2,
     las 16 piezas negras en filas 7-8.
   - A cada casilla ocupada se le asigna una pieza concreta según la
     distribución estándar: torres en a1/h1, caballos en b1/g1, alfiles en
     c1/f1, reina en d1, rey en e1, peones en a2-h2 (y simétricamente para
     las negras).
   - Se instancia un tablero python-chess con la posición inicial estándar
     para el seguimiento paralelo de legalidad.

2. Tracking de movimientos (llamada periódica desde tracker_app.py):
   - Recibe el estado nuevo del tablero (grilla 8×8).
   - Compara contra el estado anterior para detectar qué casilla se vació y
     qué casilla se llenó (movimiento simple) o qué casilla se vació y ninguna
     se llenó (captura al paso / detección de captura).
   - Identifica la pieza que se movió consultando el estado lógico interno.
   - Genera la jugada en notación UCI (ej. "e2e4") y luego en SAN (ej. "e4")
     usando python-chess.
   - Valida la legalidad del movimiento en el tablero python-chess.
     - Si es legal: aplica el movimiento al tablero python-chess y actualiza
       el estado interno.
     - Si es ilegal: emite un WARNING en consola/log pero NO rechaza el
       movimiento (el estado interno se actualiza igualmente para mantenerse
       sincronizado con la realidad física).
   - Detecta capturas: si la casilla destino estaba ocupada antes del
     movimiento, registra la pieza capturada.

3. Estado interno expuesto
   - pieces: dict[str, Piece]  casilla algebraica → Pieza
   - move_history: list[dict]  historial (se puebla en Fase 3)
   - chess_board: chess.Board  tablero python-chess sincronizado

Dependencias externas
---------------------
- python-chess >= 1.10.0  (validación de legalidad, generación SAN, export PGN)

Integración con el módulo de visión
------------------------------------
- La grilla 8×8 de entrada es la misma que produce
  camera_pipeline._b_compute_occupancy():
    team_grid: list[list[str]]  "RED" | "GREEN" | "EMPTY" | "UNKNOWN"
  donde fila 0 = rank 8 (lado de las negras) y fila 7 = rank 1 (lado de las
  blancas), asumiendo cámara sobre el lado de las blancas.
- camera_pipeline / tracker_app son los responsables de llamar a este módulo;
  game_state_tracker NO importa ni conoce camera_pipeline ni cv2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import chess


# ---------------------------------------------------------------------------
# Dataclass de pieza
# ---------------------------------------------------------------------------

@dataclass
class Piece:
    """Representa una pieza con identidad permanente a lo largo de la partida."""
    color: str          # "WHITE" | "BLACK"
    piece_type: str     # letra SAN: "K" | "Q" | "R" | "B" | "N" | "P"
    original_square: str   # casilla donde estaba al inicio, ej. "a1", "e2"


# ---------------------------------------------------------------------------
# Posición estándar de ajedrez (las 32 casillas iniciales)
# ---------------------------------------------------------------------------

STARTING_POSITION: dict[str, tuple[str, str]] = {
    # Blancas — fila 1 (piezas mayores)
    "a1": ("WHITE", "R"), "b1": ("WHITE", "N"), "c1": ("WHITE", "B"),
    "d1": ("WHITE", "Q"), "e1": ("WHITE", "K"), "f1": ("WHITE", "B"),
    "g1": ("WHITE", "N"), "h1": ("WHITE", "R"),
    # Blancas — fila 2 (peones)
    "a2": ("WHITE", "P"), "b2": ("WHITE", "P"), "c2": ("WHITE", "P"),
    "d2": ("WHITE", "P"), "e2": ("WHITE", "P"), "f2": ("WHITE", "P"),
    "g2": ("WHITE", "P"), "h2": ("WHITE", "P"),
    # Negras — fila 7 (peones)
    "a7": ("BLACK", "P"), "b7": ("BLACK", "P"), "c7": ("BLACK", "P"),
    "d7": ("BLACK", "P"), "e7": ("BLACK", "P"), "f7": ("BLACK", "P"),
    "g7": ("BLACK", "P"), "h7": ("BLACK", "P"),
    # Negras — fila 8 (piezas mayores)
    "a8": ("BLACK", "R"), "b8": ("BLACK", "N"), "c8": ("BLACK", "B"),
    "d8": ("BLACK", "Q"), "e8": ("BLACK", "K"), "f8": ("BLACK", "B"),
    "g8": ("BLACK", "N"), "h8": ("BLACK", "R"),
}

# Set de las 32 casillas de inicio válidas (filas 1, 2, 7, 8)
VALID_INITIAL_SQUARES: set[str] = set(STARTING_POSITION.keys())


# ---------------------------------------------------------------------------
# Helper de conversión de índice a notación algebraica
# ---------------------------------------------------------------------------

def cell_to_square(row: int, col: int) -> str:
    """
    Convierte índice (row, col) del team_grid a notación algebraica.

    Convención (idéntica a camera_pipeline._b_build_display):
        row=0 → rank 8  (fila superior del warp, lado de las negras)
        row=7 → rank 1  (fila inferior del warp, lado de las blancas)
        col=0 → file a  (columna izquierda)
        col=7 → file h  (columna derecha)

    Ejemplos:
        cell_to_square(0, 0) → "a8"
        cell_to_square(7, 0) → "a1"
        cell_to_square(7, 7) → "h1"
        cell_to_square(0, 7) → "h8"
        cell_to_square(6, 4) → "e2"
        cell_to_square(1, 4) → "e7"
    """
    cols = "abcdefgh"
    rank = 8 - row
    return f"{cols[col]}{rank}"


# ---------------------------------------------------------------------------
# Clase principal del tracker
# ---------------------------------------------------------------------------

class GameStateTracker:
    """Mantiene el estado lógico de la partida: identidades de piezas e historial."""

    def __init__(self) -> None:
        self.pieces: dict[str, Piece] = {}       # casilla → Pieza
        self.is_initialized: bool = False
        self.chess_board: chess.Board | None = None
        self.move_history: list = []              # se popula en Fase 3

    # ------------------------------------------------------------------
    # Inicialización desde snapshot de visión
    # ------------------------------------------------------------------

    def initialize_from_snapshot(
        self, team_grid: list[list[str]]
    ) -> tuple[bool, str]:
        """
        Inicializa el tracker a partir del estado de tablero captado por visión.

        Args:
            team_grid: Matriz 8×8 de strings ("RED" | "GREEN" | "EMPTY" | "UNKNOWN").
                       Misma estructura que devuelve _b_compute_occupancy():
                         fila 0 = rank 8, fila 7 = rank 1, col 0 = file a.
                       RED  = equipo blancas (cubos rojos)
                       GREEN = equipo negras (cubos verdes)

        Returns:
            (True, mensaje_ok)      si la inicialización fue exitosa.
            (False, mensaje_error)  si hay algún problema de validación.
        """
        new_pieces: dict[str, Piece] = {}

        for row in range(8):
            for col in range(8):
                team = team_grid[row][col]

                if team == "EMPTY":
                    continue

                if team == "UNKNOWN":
                    return False, "Hay celdas en estado UNKNOWN — limpiar tablero y reintentar"

                square = cell_to_square(row, col)
                rank   = 8 - row   # rank numérico (1-8)

                # Validar que la casilla pertenece a la posición estándar inicial
                if square not in VALID_INITIAL_SQUARES:
                    return False, f"Cubo detectado en {square}, fuera de posición inicial"

                # Validar coherencia de color:
                #   RED  (blancas) → solo rank 1 o 2
                #   GREEN (negras) → solo rank 7 u 8
                if team == "RED" and rank not in (1, 2):
                    return (
                        False,
                        f"Cubo RED en {square} no corresponde — rojas en filas 1-2, verdes en 7-8",
                    )
                if team == "GREEN" and rank not in (7, 8):
                    return (
                        False,
                        f"Cubo GREEN en {square} no corresponde — rojas en filas 1-2, verdes en 7-8",
                    )

                color, piece_type = STARTING_POSITION[square]
                new_pieces[square] = Piece(
                    color=color,
                    piece_type=piece_type,
                    original_square=square,
                )

        # Todas las validaciones pasaron — confirmar estado
        self.pieces         = new_pieces
        self.chess_board    = chess.Board()   # posición inicial estándar de python-chess
        self.is_initialized = True
        self.move_history   = []

        return True, f"Tracker inicializado con {len(self.pieces)} piezas"

    # ------------------------------------------------------------------
    # Acceso a piezas
    # ------------------------------------------------------------------

    def get_piece_at(self, square: str) -> Piece | None:
        """Devuelve la Piece en `square`, o None si la casilla está vacía."""
        return self.pieces.get(square)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reinicia el tracker a estado vacío (como si nunca se inicializara)."""
        self.pieces         = {}
        self.is_initialized  = False
        self.chess_board    = None
        self.move_history   = []


# ---------------------------------------------------------------------------
# Tests rápidos (ejecutar: python game_state_tracker.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def _make_grid() -> list[list[str]]:
        """Crea una grilla 8×8 completamente vacía."""
        return [["EMPTY"] * 8 for _ in range(8)]

    tracker = GameStateTracker()

    # ------------------------------------------------------------------
    # Test 1: cubo rojo en a1 (row=7, col=0) y verde en e7 (row=1, col=4)
    #   → inicialización exitosa, a1=torre blanca, e7=peón negro
    # ------------------------------------------------------------------
    g1 = _make_grid()
    g1[7][0] = "RED"    # a1  (rank=1 → row=7, file a → col=0)
    g1[1][4] = "GREEN"  # e7  (rank=7 → row=1, file e → col=4)
    ok, msg = tracker.initialize_from_snapshot(g1)
    assert ok, f"Test 1 FALLÓ: {msg}"
    assert tracker.pieces["a1"].piece_type == "R",      "Test 1: a1 debería ser torre (R)"
    assert tracker.pieces["a1"].color      == "WHITE",  "Test 1: a1 debería ser blanca"
    assert tracker.pieces["e7"].piece_type == "P",      "Test 1: e7 debería ser peón (P)"
    assert tracker.pieces["e7"].color      == "BLACK",  "Test 1: e7 debería ser negra"
    assert tracker.is_initialized,                       "Test 1: tracker debería estar inicializado"
    assert tracker.chess_board is not None,              "Test 1: chess_board debería existir"
    print(f"[Test 1] OK — {msg}")

    # ------------------------------------------------------------------
    # Test 2: cubo rojo en d4 → fuera de posición inicial
    #   (rank=4 → row=4, file d → col=3)
    # ------------------------------------------------------------------
    tracker.reset()
    g2 = _make_grid()
    g2[4][3] = "RED"    # d4
    ok, msg = tracker.initialize_from_snapshot(g2)
    assert not ok,         "Test 2: debería fallar (d4 fuera de posición inicial)"
    assert "d4" in msg,    f"Test 2: el mensaje debería mencionar 'd4', got: {msg}"
    assert not tracker.is_initialized, "Test 2: tracker NO debería estar inicializado"
    print(f"[Test 2] OK — {msg}")

    # ------------------------------------------------------------------
    # Test 3: cubo verde en a2 → color incoherente (rank 2 no es 7 u 8)
    #   (rank=2 → row=6, file a → col=0)
    # ------------------------------------------------------------------
    tracker.reset()
    g3 = _make_grid()
    g3[6][0] = "GREEN"  # a2
    ok, msg = tracker.initialize_from_snapshot(g3)
    assert not ok, "Test 3: debería fallar (GREEN en rank 2)"
    assert "GREEN" in msg or "a2" in msg, f"Test 3: mensaje inesperado: {msg}"
    assert not tracker.is_initialized, "Test 3: tracker NO debería estar inicializado"
    print(f"[Test 3] OK — {msg}")

    # ------------------------------------------------------------------
    # Test 4: tablero completamente vacío → éxito con 0 piezas
    # ------------------------------------------------------------------
    tracker.reset()
    g4 = _make_grid()
    ok, msg = tracker.initialize_from_snapshot(g4)
    assert ok,                       f"Test 4 FALLÓ: {msg}"
    assert len(tracker.pieces) == 0, "Test 4: debería haber 0 piezas"
    assert tracker.is_initialized,   "Test 4: tracker debería estar inicializado"
    print(f"[Test 4] OK — {msg}")

    print("\nTodos los tests pasaron correctamente.")
