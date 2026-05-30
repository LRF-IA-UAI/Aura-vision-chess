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
   - Se instancia un tablero python-chess con la posición inicial para el
     seguimiento paralelo de legalidad.

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
   - board_matrix: dict[str, str]  casilla algebraica → letra FEN
     (ej. {"e4": "P", "d7": "n", ...})
   - move_history: list[dict]  historial de movimientos detectados, cada
     elemento con claves: {"uci", "san", "legal", "captured", "timestamp"}
   - chess_board: chess.Board  tablero python-chess sincronizado (puede
     desviarse si hay movimientos ilegales sucesivos)

Dependencias externas
---------------------
- python-chess >= 1.10.0  (validación de legalidad, generación SAN, export PGN)

Integración con el módulo de visión
------------------------------------
- La grilla 8×8 de entrada es la misma que produce
  camera_pipeline._b_compute_occupancy():
    occ_grid:  list[list[bool]]  True = celda ocupada
    team_grid: list[list[str]]   "RED" | "GREEN" | "EMPTY" | "UNKNOWN"
  donde fila 0 = rank 8 (lado de las negras) y fila 7 = rank 1 (lado de las
  blancas), asumiendo cámara sobre el lado de las blancas.
- tracker_app.py es el responsable de llamar a este módulo; game_state_tracker
  NO importa ni conoce camera_pipeline ni cv2.

Notas de implementación (Fase 2)
---------------------------------
- El lock-in puede fallar si el tablero no está en posición estándar; en ese
  caso se debe notificar al usuario y no inicializar el tracking.
- Para manejar el enroque se necesita detectar dos movimientos simultáneos
  (rey y torre); esto requiere lógica especial en el comparador de estados.
- Las piezas no se re-identifican tras una captura: si una pieza blanca es
  capturada y una negra ocupa su lugar, el módulo de visión lo detectará como
  "celda que cambió de equipo", no como "celda vació + celda llenó".
"""
