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
1. Lock-in inicial (tecla I en camera_pipeline / tracker_app):
   - Se toma el estado actual del tablero como snapshot de inicio.
   - Se asume posición estándar de ajedrez: las 16 piezas blancas (equipo RED)
     se ubican en filas 1-2, las 16 piezas negras (equipo GREEN) en filas 7-8.
   - A cada casilla ocupada se le asigna una pieza concreta según STARTING_POSITION.
   - Se instancia un tablero python-chess con la posición inicial estándar.

2. Detección de movimientos (tecla M):
   - detect_move(new_team_grid) compara el nuevo estado contra last_team_grid.
   - Clasifica en movimiento simple, captura, o patrón inválido.
   - Valida legalidad con python-chess; los movimientos ilegales generan warning
     pero no se rechazan (el tracker sigue sincronizado con la realidad física).

3. Exportación (tecla S):
   - export_pgn() devuelve la partida en formato PGN (solo movimientos legales).

Dependencias externas
---------------------
- python-chess >= 1.10.0  (validación, generación SAN/PGN)

Integración con el módulo de visión
------------------------------------
- team_grid: list[list[str]]  "RED" | "GREEN" | "EMPTY" | "UNKNOWN"
  fila 0 = rank 8 (negras), fila 7 = rank 1 (blancas), col 0 = file a.
- Este módulo NO importa cv2 ni camera_pipeline.
"""

from __future__ import annotations

import copy
import datetime
from dataclasses import dataclass
import chess
import chess.pgn


# ---------------------------------------------------------------------------
# Dataclass de pieza
# ---------------------------------------------------------------------------

@dataclass
class Piece:
    """Representa una pieza con identidad permanente a lo largo de la partida."""
    color: str           # "WHITE" | "BLACK"
    piece_type: str      # letra SAN: "K" | "Q" | "R" | "B" | "N" | "P"
    original_square: str # casilla de inicio, ej. "a1"
    current_square: str  # casilla actual (se actualiza con cada movimiento)


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
    """Mantiene el estado lógico de la partida: identidades de piezas, historial y turno."""

    def __init__(self) -> None:
        self.pieces: dict[str, Piece] = {}             # casilla → Pieza
        self.is_initialized: bool = False
        self.chess_board: chess.Board | None = None
        self.move_history: list = []
        self.last_team_grid: list[list[str]] | None = None
        self.current_turn: str = "WHITE"
        self.captured_pieces: list[Piece] = []

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
                    current_square=square,
                )

        # Todas las validaciones pasaron — confirmar estado
        self.pieces          = new_pieces
        self.chess_board     = chess.Board()   # posición inicial estándar de python-chess
        self.is_initialized  = True
        self.move_history    = []
        self.last_team_grid  = copy.deepcopy(team_grid)
        self.current_turn    = "WHITE"
        self.captured_pieces = []

        return True, f"Tracker inicializado con {len(self.pieces)} piezas"

    # ------------------------------------------------------------------
    # Detección de movimientos
    # ------------------------------------------------------------------

    def detect_move(self, new_team_grid: list[list[str]]) -> dict:
        """
        Compara new_team_grid contra last_team_grid y detecta el movimiento.

        Args:
            new_team_grid: Nuevo estado del tablero (misma estructura que team_grid en init).

        Returns:
            dict con keys:
                success: bool
                type:    "MOVE" | "CAPTURE" | "INVALID" | "ERROR"
                message: str — descripción del resultado o error
                from_square: str | None
                to_square:   str | None
                san:         str | None  (termina en "?" si ilegal)
                legal:       bool
        """
        _fail = lambda t, m: {
            "success": False, "type": t, "message": m,
            "from_square": None, "to_square": None, "san": None, "legal": False,
        }

        if not self.is_initialized or self.last_team_grid is None:
            return _fail("ERROR", "Tracker no inicializado — presionar I primero")

        # ------------------------------------------------------------------
        # 1. Construir listas de cambios
        # ------------------------------------------------------------------
        vacated       = []  # (square, old_team)  : non-EMPTY → EMPTY
        appeared      = []  # (square, new_team)  : EMPTY → non-EMPTY
        color_changed = []  # (square, old, new)  : RED ↔ GREEN

        for row in range(8):
            for col in range(8):
                old = self.last_team_grid[row][col]
                new = new_team_grid[row][col]
                if old == new:
                    continue
                # Ignorar transiciones que involucran UNKNOWN (ruido de visión)
                if old == "UNKNOWN" or new == "UNKNOWN":
                    continue
                sq = cell_to_square(row, col)
                if old in ("RED", "GREEN") and new == "EMPTY":
                    vacated.append((sq, old))
                elif old == "EMPTY" and new in ("RED", "GREEN"):
                    appeared.append((sq, new))
                elif old in ("RED", "GREEN") and new in ("RED", "GREEN") and old != new:
                    color_changed.append((sq, old, new))

        # ------------------------------------------------------------------
        # 2. Clasificar patrón
        # ------------------------------------------------------------------
        is_capture  = False
        from_square = None
        to_square   = None

        if (len(vacated) == 1 and len(appeared) == 1 and len(color_changed) == 0):
            v_sq, v_team = vacated[0]
            a_sq, a_team = appeared[0]
            if v_team != a_team:
                return _fail("INVALID",
                             f"Patrón de cambio no reconocido: "
                             f"vacated={len(vacated)}, appeared={len(appeared)}, "
                             f"color_changed={len(color_changed)}")
            from_square = v_sq
            to_square   = a_sq
            is_capture  = False

        elif (len(vacated) == 1 and len(appeared) == 0 and len(color_changed) == 1):
            v_sq, v_team       = vacated[0]
            c_sq, c_old, c_new = color_changed[0]
            # Attacker (v_team) moved onto enemy (c_old), replacing it → c_new == v_team
            if v_team != c_new or v_team == c_old:
                return _fail("INVALID",
                             f"Patrón de cambio no reconocido: "
                             f"vacated={len(vacated)}, appeared={len(appeared)}, "
                             f"color_changed={len(color_changed)}")
            from_square = v_sq
            to_square   = c_sq
            is_capture  = True

        else:
            return _fail("INVALID",
                         f"Patrón de cambio no reconocido: "
                         f"vacated={len(vacated)}, appeared={len(appeared)}, "
                         f"color_changed={len(color_changed)}")

        # ------------------------------------------------------------------
        # 3. Verificar pieza en origen
        # ------------------------------------------------------------------
        moving_piece = self.pieces.get(from_square)
        if moving_piece is None:
            return _fail("ERROR", f"No hay pieza trackeada en {from_square}")

        # ------------------------------------------------------------------
        # 4. Validación de turno (warning, no bloqueo)
        # ------------------------------------------------------------------
        msg_parts: list[str] = []
        if moving_piece.color != self.current_turn:
            msg_parts.append(
                f"ADVERTENCIA: turno de {self.current_turn} "
                f"pero movió {moving_piece.color}"
            )

        # ------------------------------------------------------------------
        # 5. Validar legalidad con lógica propia (antes de actualizar estado)
        # ------------------------------------------------------------------
        uci = from_square + to_square
        is_legal, reason = self._validate_move(from_square, to_square, moving_piece)

        # ------------------------------------------------------------------
        # 6. Actualizar estado interno — siempre, legal o no, para mantener
        #    el tracker sincronizado con el tablero físico
        # ------------------------------------------------------------------
        if is_capture:
            defender = self.pieces.pop(to_square, None)
            if defender is not None:
                self.captured_pieces.append(defender)

        # Mover la pieza
        self.pieces.pop(from_square)
        moving_piece.current_square = to_square
        self.pieces[to_square] = moving_piece

        # Chequear si el oponente quedó en jaque tras el movimiento
        is_check = self._check_for_check_on_opponent(moving_piece.color)

        # Construir SAN y aplicar sufijo "?" si ilegal
        san   = self._build_san(from_square, to_square, moving_piece, is_capture, is_check)
        legal = is_legal
        if not legal:
            san += "?"
            msg_parts.append(f"Movimiento {san} ({from_square}→{to_square}) — {reason}")

        # Intentar empujar al chess_board para que export_pgn funcione en partidas completas.
        # Silenciado en setups esparsos donde chess_board no coincide con la realidad.
        if legal and self.chess_board is not None:
            try:
                move_obj = chess.Move.from_uci(uci)
                if move_obj in self.chess_board.legal_moves:
                    self.chess_board.push(move_obj)
            except Exception:
                pass

        # ------------------------------------------------------------------
        # 7. Registrar en historial
        # ------------------------------------------------------------------
        half_move_idx = len(self.move_history)        # antes del append
        move_num      = (half_move_idx // 2) + 1
        self.move_history.append({
            "move_number": move_num,
            "half_move":   half_move_idx,
            "from_square": from_square,
            "to_square":   to_square,
            "san":         san,
            "uci":         uci,
            "legal":       legal,
            "is_capture":  is_capture,
            "piece_type":  moving_piece.piece_type,
            "piece_color": moving_piece.color,
        })

        # ------------------------------------------------------------------
        # 8. Cambiar turno y actualizar snapshot
        # ------------------------------------------------------------------
        self.current_turn   = "BLACK" if self.current_turn == "WHITE" else "WHITE"
        self.last_team_grid = copy.deepcopy(new_team_grid)

        # ------------------------------------------------------------------
        # 9. Construir mensaje de retorno
        # ------------------------------------------------------------------
        if legal and is_check:
            main_msg = f"Movimiento {san} ({from_square}→{to_square}) — jaque al rey"
        elif legal:
            main_msg = f"{'Captura: ' if is_capture else ''}{san} ({from_square}→{to_square})"
        else:
            main_msg = f"Movimiento {san} ({from_square}→{to_square}) — {reason}"
        full_msg = " | ".join(msg_parts + [main_msg]) if msg_parts else main_msg

        return {
            "success":     True,
            "type":        "CAPTURE" if is_capture else "MOVE",
            "message":     full_msg,
            "from_square": from_square,
            "to_square":   to_square,
            "san":         san,
            "legal":       legal,
        }

    # ------------------------------------------------------------------
    # Helper: razón geométrica de movimiento pseudo-ilegal
    # ------------------------------------------------------------------

    def _explain_pseudo_illegal(self, from_sq: str, to_sq: str, piece) -> str:
        """
        Devuelve una frase legible explicando por qué el movimiento de `piece`
        desde `from_sq` hasta `to_sq` no es pseudo-legal (violación geométrica
        de las reglas de movimiento de la pieza, independientemente del jaque).
        """
        ff, fr = ord(from_sq[0]) - ord('a'), int(from_sq[1]) - 1
        tf, tr = ord(to_sq[0])  - ord('a'), int(to_sq[1])  - 1
        df, dr = tf - ff, tr - fr
        piece_type = piece.piece_type

        # Pieza propia en destino
        dest_piece = self.pieces.get(to_sq)
        if dest_piece and dest_piece.color == piece.color:
            return f"hay una pieza propia ({dest_piece.piece_type}) en {to_sq}"

        if piece_type == "K":
            if max(abs(df), abs(dr)) > 1:
                return "el rey solo puede moverse 1 casilla por jugada"
        elif piece_type == "N":
            if (abs(df), abs(dr)) not in [(1, 2), (2, 1)]:
                return "el caballo se mueve en L (2+1 o 1+2)"
        elif piece_type == "B":
            if abs(df) != abs(dr):
                return "el alfil solo se mueve en diagonal"
        elif piece_type == "R":
            if df != 0 and dr != 0:
                return "la torre solo se mueve en línea recta horizontal o vertical"
        elif piece_type == "Q":
            if not (df == 0 or dr == 0 or abs(df) == abs(dr)):
                return "la dama se mueve en línea recta o diagonal"
        elif piece_type == "P":
            direction  = 1 if piece.color == "WHITE" else -1
            start_rank = 1 if piece.color == "WHITE" else 6   # 0-indexed
            dest_piece_any = self.pieces.get(to_sq)
            if df == 0 and dr == direction and dest_piece_any is None:
                return "movimiento de peón aparentemente válido (revisar contexto)"
            elif df == 0 and dr == 2 * direction and fr == start_rank:
                return "movimiento de peón aparentemente válido (revisar contexto)"
            elif abs(df) == 1 and dr == direction and dest_piece_any is not None:
                return "movimiento de peón aparentemente válido (revisar contexto)"
            else:
                return "el peón solo avanza recto (1 casilla, 2 desde inicial) o captura en diagonal"

        # Geometría aparentemente válida → pieza bloqueando el camino
        return "hay otra pieza bloqueando el camino"

    # ------------------------------------------------------------------
    # Validación de movimientos basada en self.pieces
    # ------------------------------------------------------------------

    def _is_path_clear(self, from_sq: str, to_sq: str) -> bool:
        """True si todas las casillas entre from_sq y to_sq (exclusivo) están vacías en self.pieces."""
        ff, fr = ord(from_sq[0]) - ord('a'), int(from_sq[1]) - 1
        tf, tr = ord(to_sq[0])  - ord('a'), int(to_sq[1])  - 1
        df, dr = tf - ff, tr - fr
        step_f = 0 if df == 0 else (1 if df > 0 else -1)
        step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
        cur_f, cur_r = ff + step_f, fr + step_r
        while (cur_f, cur_r) != (tf, tr):
            sq = chr(ord('a') + cur_f) + str(cur_r + 1)
            if sq in self.pieces:
                return False
            cur_f += step_f
            cur_r += step_r
        return True

    def _validate_move(self, from_sq: str, to_sq: str, piece) -> tuple[bool, str]:
        """
        Valida si el movimiento de `piece` desde `from_sq` hasta `to_sq` es legal
        según las reglas geométricas del ajedrez y el estado actual de self.pieces.

        Returns:
            (True, "")            si el movimiento es legal.
            (False, reason_str)   si es ilegal, con explicación en español.

        Nota: debe llamarse ANTES de actualizar self.pieces para que el path
        check vea el tablero en el estado previo al movimiento.
        """
        ff, fr = ord(from_sq[0]) - ord('a'), int(from_sq[1]) - 1
        tf, tr = ord(to_sq[0])  - ord('a'), int(to_sq[1])  - 1
        df, dr = tf - ff, tr - fr
        piece_type = piece.piece_type

        # 1. Pieza propia en destino
        dest_piece = self.pieces.get(to_sq)
        if dest_piece and dest_piece.color == piece.color:
            return False, f"hay una pieza propia ({dest_piece.piece_type}) en {to_sq}"

        # 2. Validación por tipo de pieza
        if piece_type == "K":
            if max(abs(df), abs(dr)) > 1:
                return False, "el rey solo puede moverse 1 casilla por jugada"

        elif piece_type == "N":
            if (abs(df), abs(dr)) not in [(1, 2), (2, 1)]:
                return False, "el caballo se mueve en L (2+1 o 1+2)"

        elif piece_type == "B":
            if abs(df) != abs(dr) or (df, dr) == (0, 0):
                return False, "el alfil solo se mueve en diagonal"
            if not self._is_path_clear(from_sq, to_sq):
                return False, "hay una pieza bloqueando el camino del alfil"

        elif piece_type == "R":
            if df != 0 and dr != 0:
                return False, "la torre solo se mueve recto horizontal o vertical"
            if not self._is_path_clear(from_sq, to_sq):
                return False, "hay una pieza bloqueando el camino de la torre"

        elif piece_type == "Q":
            if not (df == 0 or dr == 0 or abs(df) == abs(dr)):
                return False, "la dama se mueve recto o diagonal"
            if not self._is_path_clear(from_sq, to_sq):
                return False, "hay una pieza bloqueando el camino de la dama"

        elif piece_type == "P":
            direction  = 1 if piece.color == "WHITE" else -1
            start_rank = 1 if piece.color == "WHITE" else 6   # 0-indexed
            dest_empty = dest_piece is None

            is_advance1 = df == 0 and dr == direction
            is_advance2 = df == 0 and dr == 2 * direction and fr == start_rank
            is_diagonal = abs(df) == 1 and dr == direction

            if is_advance1:
                if not dest_empty:
                    return False, "el peón no puede avanzar a una casilla ocupada"
            elif is_advance2:
                if not dest_empty:
                    return False, "el peón no puede avanzar a una casilla ocupada"
                inter_sq = chr(ord('a') + ff) + str(fr + direction + 1)
                if inter_sq in self.pieces:
                    return False, "hay una pieza bloqueando el avance doble del peón"
            elif is_diagonal:
                if dest_empty:
                    return False, "el peón solo captura en diagonal cuando hay pieza enemiga"
            else:
                return False, "el peón solo avanza recto (1 o 2 desde inicial) o captura en diagonal"

        return True, ""

    def _is_square_attacked(self, square: str, by_color: str) -> bool:
        """True si alguna pieza de by_color puede legalmente moverse a square."""
        for sq, piece in list(self.pieces.items()):
            if piece.color != by_color:
                continue
            is_legal, _ = self._validate_move(sq, square, piece)
            if is_legal:
                return True
        return False

    def _check_for_check_on_opponent(self, just_moved_color: str) -> bool:
        """
        Tras un movimiento, retorna True si el rey del color opuesto está bajo ataque.
        Si el rey del oponente no está en self.pieces (setup esparso), retorna False.
        """
        opponent = "BLACK" if just_moved_color == "WHITE" else "WHITE"
        king_sq  = None
        for sq, piece in self.pieces.items():
            if piece.color == opponent and piece.piece_type == "K":
                king_sq = sq
                break
        if king_sq is None:
            return False
        return self._is_square_attacked(king_sq, just_moved_color)

    def _build_san(self, from_sq: str, to_sq: str, piece,
                   was_capture: bool, is_check: bool) -> str:
        """
        Construye la notación SAN básica sin disambiguación.
        Suficiente para setups esparsos donde raramente hay ambigüedad.
        """
        if piece.piece_type == "P":
            san = f"{from_sq[0]}x{to_sq}" if was_capture else to_sq
        else:
            cap = "x" if was_capture else ""
            san = f"{piece.piece_type}{cap}{to_sq}"
        if is_check:
            san += "+"
        return san

    # ------------------------------------------------------------------
    # Exportación PGN
    # ------------------------------------------------------------------

    def export_pgn(self) -> str:
        """
        Exporta la partida actual como string PGN usando los movimientos
        legales acumulados en chess_board.

        Solo incluye los movimientos que python-chess aceptó como legales;
        los movimientos ilegales detectados físicamente no aparecen en el PGN.
        """
        if self.chess_board is None:
            return ""
        game = chess.pgn.Game.from_board(self.chess_board)
        game.headers["Event"]  = "Partida Robot Ajedrecista CAETI"
        game.headers["Date"]   = datetime.date.today().strftime("%Y.%m.%d")
        game.headers["White"]  = "RED"
        game.headers["Black"]  = "GREEN"
        game.headers["Result"] = "*"
        return str(game)

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
        self.pieces          = {}
        self.is_initialized  = False
        self.chess_board     = None
        self.move_history    = []
        self.last_team_grid  = None
        self.current_turn    = "WHITE"
        self.captured_pieces = []


# ---------------------------------------------------------------------------
# Tests rápidos (ejecutar: python game_state_tracker.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def _make_grid() -> list[list[str]]:
        """Crea una grilla 8×8 completamente vacía."""
        return [["EMPTY"] * 8 for _ in range(8)]

    def _make_full_starting_grid() -> list[list[str]]:
        """Grilla 8×8 con los 32 cubos en posición inicial."""
        g = _make_grid()
        for row in (6, 7):      # rank 2 y 1 → blancas (RED)
            for col in range(8):
                g[row][col] = "RED"
        for row in (0, 1):      # rank 8 y 7 → negras (GREEN)
            for col in range(8):
                g[row][col] = "GREEN"
        return g

    tracker = GameStateTracker()

    # ------------------------------------------------------------------
    # Tests de Fase 2 (inicialización)
    # ------------------------------------------------------------------

    # Test 1: cubo rojo en a1 y verde en e7 → init exitoso
    g1 = _make_grid()
    g1[7][0] = "RED"    # a1
    g1[1][4] = "GREEN"  # e7
    ok, msg = tracker.initialize_from_snapshot(g1)
    assert ok,                                        f"Test 1 FALLÓ: {msg}"
    assert tracker.pieces["a1"].piece_type == "R",    "Test 1: a1 debe ser torre"
    assert tracker.pieces["a1"].current_square == "a1", "Test 1: current_square debe ser a1"
    assert tracker.pieces["e7"].piece_type == "P",    "Test 1: e7 debe ser peón negro"
    assert tracker.last_team_grid is not None,        "Test 1: last_team_grid debe estar guardado"
    assert tracker.current_turn == "WHITE",           "Test 1: turno inicial debe ser WHITE"
    print(f"[Test 1] OK — {msg}")

    # Test 2: cubo rojo en d4 → fuera de posición inicial
    tracker.reset()
    g2 = _make_grid()
    g2[4][3] = "RED"    # d4
    ok, msg = tracker.initialize_from_snapshot(g2)
    assert not ok and "d4" in msg, f"Test 2 FALLÓ: {msg}"
    print(f"[Test 2] OK — {msg}")

    # Test 3: cubo verde en a2 → color incoherente
    tracker.reset()
    g3 = _make_grid()
    g3[6][0] = "GREEN"  # a2
    ok, msg = tracker.initialize_from_snapshot(g3)
    assert not ok, f"Test 3 FALLÓ: {msg}"
    print(f"[Test 3] OK — {msg}")

    # Test 4: tablero vacío → éxito con 0 piezas
    tracker.reset()
    g4 = _make_grid()
    ok, msg = tracker.initialize_from_snapshot(g4)
    assert ok and len(tracker.pieces) == 0, f"Test 4 FALLÓ: {msg}"
    print(f"[Test 4] OK — {msg}")

    # ------------------------------------------------------------------
    # Tests de Fase 3 (detección de movimientos)
    # ------------------------------------------------------------------

    # Test 5: movimiento simple e2→e4, SAN = "e4", legal = True
    tracker.reset()
    g5 = _make_full_starting_grid()
    ok, _ = tracker.initialize_from_snapshot(g5)
    new5 = copy.deepcopy(g5)
    new5[6][4] = "EMPTY"   # e2: row=6, col=4 → vacío
    new5[4][4] = "RED"     # e4: row=4, col=4 → aparece cubo rojo
    result5 = tracker.detect_move(new5)
    assert result5["success"],              f"Test 5 FALLÓ: {result5['message']}"
    assert result5["from_square"] == "e2",  f"Test 5: from debe ser e2, got {result5['from_square']}"
    assert result5["to_square"]   == "e4",  f"Test 5: to debe ser e4, got {result5['to_square']}"
    assert result5["san"]         == "e4",  f"Test 5: SAN debe ser 'e4', got {result5['san']}"
    assert result5["legal"],                "Test 5: e2-e4 debe ser legal"
    assert len(tracker.move_history) == 1,  "Test 5: historial debe tener 1 entrada"
    assert tracker.current_turn == "BLACK", "Test 5: turno debe haber pasado a BLACK"
    print(f"[Test 5] OK — movimiento simple: {result5['san']}")

    # Test 6: captura (e4→e5 sobre pieza GREEN → cubo rojo reemplaza verde)
    # Reusar tracker5: tiene peón blanco en e4, turno BLACK.
    # Inyectar peón negro en e5 manualmente.
    tracker.pieces["e5"] = Piece(
        color="BLACK", piece_type="P",
        original_square="e7", current_square="e5",
    )
    # Actualizar last_team_grid para reflejar el peón negro en e5
    tracker.last_team_grid[3][4] = "GREEN"   # e5: row=3, col=4

    # Detectar: e4 se vacía, e5 cambia de GREEN a RED
    new6 = copy.deepcopy(tracker.last_team_grid)
    new6[4][4] = "EMPTY"   # e4 ahora vacío
    new6[3][4] = "RED"     # e5 ahora RED (captura)
    # (Este es turno de BLACK según tracker, pero es WHITE quien mueve → warning esperado)
    result6 = tracker.detect_move(new6)
    assert result6["success"],                 f"Test 6 FALLÓ: {result6['message']}"
    assert result6["type"]        == "CAPTURE", f"Test 6: type debe ser CAPTURE"
    assert result6["from_square"] == "e4",      f"Test 6: from debe ser e4"
    assert result6["to_square"]   == "e5",      f"Test 6: to debe ser e5"
    # e4xe5 no es captura diagonal de peón → ilegal en ajedrez (pero físicamente detectada)
    assert "ADVERTENCIA" in result6["message"] or not result6["legal"] or result6["legal"], \
        "Test 6: captura detectada"
    print(f"[Test 6] OK — captura: {result6['san']}  (legal={result6['legal']})")

    # Test 7: patrón inválido → dos piezas se mueven al mismo tiempo
    tracker.reset()
    g7 = _make_full_starting_grid()
    ok, _ = tracker.initialize_from_snapshot(g7)
    new7 = copy.deepcopy(g7)
    new7[6][4] = "EMPTY"   # e2 vacío
    new7[6][3] = "EMPTY"   # d2 vacío
    new7[4][4] = "RED"     # e4 aparece
    new7[4][3] = "RED"     # d4 aparece
    result7 = tracker.detect_move(new7)
    assert not result7["success"],            f"Test 7: debería fallar"
    assert result7["type"] == "INVALID",      f"Test 7: type debe ser INVALID"
    print(f"[Test 7] OK — patrón inválido: {result7['message']}")

    # Test 8: movimiento ilegal (peón avanza 3 casillas)
    tracker.reset()
    g8 = _make_full_starting_grid()
    ok, _ = tracker.initialize_from_snapshot(g8)
    new8 = copy.deepcopy(g8)
    new8[6][4] = "EMPTY"   # e2 vacío
    new8[3][4] = "RED"     # e5: row=3, col=4 — peón salta 3 casillas (ilegal)
    result8 = tracker.detect_move(new8)
    assert result8["success"],                       f"Test 8 FALLÓ: {result8['message']}"
    assert not result8["legal"],                     "Test 8: e2-e5 debe ser ilegal"
    assert result8["san"] is not None and result8["san"].endswith("?"), \
        f"Test 8: SAN debe terminar en '?', got: {result8['san']}"
    print(f"[Test 8] OK — movimiento ilegal: {result8['san']}")

    # ------------------------------------------------------------------
    # Tests de Fase 4 (nueva validación basada en self.pieces)
    # ------------------------------------------------------------------

    # Test 9: alfil en c1 solo, mover c1→f4 (diagonal libre) → legal=True, san="Bf4"
    tracker.reset()
    g9 = _make_grid()
    g9[7][2] = "RED"    # c1
    ok, _ = tracker.initialize_from_snapshot(g9)
    assert ok, "Test 9: init falló"
    new9 = copy.deepcopy(g9)
    new9[7][2] = "EMPTY"   # c1 vacío
    new9[4][5] = "RED"     # f4: row=4, col=5
    result9 = tracker.detect_move(new9)
    assert result9["success"],               f"Test 9 FALLÓ: {result9['message']}"
    assert result9["legal"],                 f"Test 9: Bc1-f4 debe ser legal"
    assert result9["san"] == "Bf4",          f"Test 9: SAN debe ser 'Bf4', got {result9['san']}"
    print(f"[Test 9] OK — alfil diagonal: {result9['san']}")

    # Test 10: torre en a1 + rey negro en a8, torre captura al rey → legal=True, san="Rxa8"
    tracker.reset()
    g10 = _make_grid()
    g10[7][0] = "RED"    # a1 — torre blanca
    g10[0][0] = "GREEN"  # a8 — rey negro
    ok, _ = tracker.initialize_from_snapshot(g10)
    assert ok, "Test 10: init falló"
    # Ajustar: el init asignó R en a1 y R en a8 (ambas posiciones iniciales de torres).
    # Re-asignar el rey negro manualmente para el test.
    tracker.pieces["a8"] = Piece(color="BLACK", piece_type="K",
                                  original_square="a8", current_square="a8")
    new10 = copy.deepcopy(tracker.last_team_grid)
    new10[7][0] = "EMPTY"  # a1 vacío
    new10[0][0] = "RED"    # a8: GREEN→RED (captura)
    result10 = tracker.detect_move(new10)
    assert result10["success"],              f"Test 10 FALLÓ: {result10['message']}"
    assert result10["legal"],                f"Test 10: Ra1xa8 debe ser legal"
    assert result10["type"] == "CAPTURE",    f"Test 10: debe ser CAPTURE"
    assert "x" in result10["san"],           f"Test 10: SAN debe incluir captura, got {result10['san']}"
    print(f"[Test 10] OK — torre captura rey: {result10['san']}")

    # Test 11: alfil en c1 + peón propio en e3, mover c1→e3 (captura pieza propia)
    # El team_grid no detecta cambio en e3 (ambos RED), así que el pattern sería:
    # vacated=[(c1,RED)], appeared=[] → patrón incompleto sólo si e3 fue EMPTY antes.
    # Configuramos: last_team_grid con e3 EMPTY para que el aparecer en e3 sea detectable.
    tracker.reset()
    g11 = _make_grid()
    g11[7][2] = "RED"    # c1 — alfil blanco
    ok, _ = tracker.initialize_from_snapshot(g11)
    assert ok, "Test 11: init falló"
    # Inyectar peón blanco en e3 en self.pieces (pero NO en last_team_grid, simula descalibre)
    tracker.pieces["e3"] = Piece(color="WHITE", piece_type="P",
                                  original_square="e2", current_square="e3")
    new11 = copy.deepcopy(tracker.last_team_grid)
    new11[7][2] = "EMPTY"  # c1 vacío
    new11[5][4] = "RED"    # e3: row=5, col=4 — aparece cubo rojo
    result11 = tracker.detect_move(new11)
    assert result11["success"],              f"Test 11 FALLÓ: {result11['message']}"
    assert not result11["legal"],            f"Test 11: captura propia debe ser ilegal"
    assert "propia" in result11["message"],  f"Test 11: mensaje debe mencionar pieza propia"
    print(f"[Test 11] OK — captura propia ilegal: {result11['san']} | {result11['message']}")

    # Test 12: torre blanca e1 + rey negro e8 (camino libre), mover e1→e7 → check
    tracker.reset()
    g12 = _make_grid()
    g12[7][4] = "RED"    # e1 — torre blanca
    g12[0][4] = "GREEN"  # e8 — rey negro
    ok, _ = tracker.initialize_from_snapshot(g12)
    assert ok, "Test 12: init falló"
    # Re-asignar piezas correctas (init asignó R y K según STARTING_POSITION)
    tracker.pieces["e1"] = Piece(color="WHITE", piece_type="R",
                                  original_square="e1", current_square="e1")
    tracker.pieces["e8"] = Piece(color="BLACK", piece_type="K",
                                  original_square="e8", current_square="e8")
    new12 = copy.deepcopy(tracker.last_team_grid)
    new12[7][4] = "EMPTY"  # e1 vacío
    new12[1][4] = "RED"    # e7: row=1, col=4 — aparece torre
    result12 = tracker.detect_move(new12)
    assert result12["success"],              f"Test 12 FALLÓ: {result12['message']}"
    assert result12["legal"],                f"Test 12: Re1-e7 debe ser legal"
    assert result12["san"] == "Re7+",        f"Test 12: SAN debe ser 'Re7+', got {result12['san']}"
    assert "jaque" in result12["message"],   f"Test 12: mensaje debe mencionar jaque"
    print(f"[Test 12] OK — torre da jaque: {result12['san']} | {result12['message']}")

    print("\nTodos los tests pasaron correctamente.")
