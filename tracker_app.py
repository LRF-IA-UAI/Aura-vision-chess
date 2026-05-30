"""
tracker_app.py — Aplicación principal de tracking de partidas de ajedrez.
Robot Ajedrecista CAETI (UAI)

Propósito
---------
Este módulo es el punto de entrada de la extensión de tracking. Conecta el
módulo de visión existente (camera_pipeline / board_detector) con el modelo
lógico de game_state_tracker para producir un registro completo de la partida
en formato PGN.

Es el equivalente funcional de camera_pipeline.py pero orientado a tracking
en lugar de sugerencia de jugadas: no usa Stockfish, no calcula la mejor
jugada, y el loop principal se enfoca en detectar y registrar los movimientos
que ocurren físicamente en el tablero.

Teclas del loop principal
-------------------------
R — Calibrar el tablero (reutiliza el flujo existente de camera_pipeline):
      Detecta el borde rojo del tablero físico, corre findChessboardCorners
      para obtener las esquinas subpixel del área de juego, calcula y guarda
      la homografía borde-rojo → área de juego. Requiere tablero vacío o con
      piezas (la homografía depende solo del borde, no de las piezas).
      Sin calibración previa, I, S y el tracking automático no funcionan.

I — Lock-in de posición inicial:
      Congela el estado actual del tablero como posición de inicio de partida.
      Invoca game_state_tracker para asignar identidades de piezas basándose
      en la distribución estándar de ajedrez (filas 1-2 = blancas, 7-8 = negras).
      A partir de este momento el tracker comienza a comparar frames sucesivos
      en busca de movimientos.
      Debe presionarse con el tablero en posición estándar inicial; si el número
      de piezas detectadas no corresponde a los 32 esperados se mostrará un
      warning pero se intentará el lock-in igualmente.

S — Guardar la partida actual en PGN:
      Exporta el historial de movimientos registrado hasta el momento en un
      archivo .pgn con timestamp (ej. "partida_20250530_143022.pgn").
      El PGN incluye headers básicos: Event, Date, White, Black, Result (? si
      la partida no terminó).
      Si no hay movimientos registrados todavía muestra un mensaje de aviso.

Q — Salir:
      Cierra la cámara, destruye las ventanas OpenCV y termina el proceso.
      Si hay movimientos sin guardar, pregunta al usuario si desea guardar
      antes de salir (comportamiento a definir en Fase 2).

Arquitectura del loop principal
--------------------------------
1. Leer frame de cámara.
2. Ejecutar _update_red_border_tracking(frame) para mantener la homografía
   actualizada frame a frame (reutilizado de camera_pipeline).
3. Si el tracking está activo (estado ON o STALE) y el lock-in fue realizado:
   a. Warpar el frame usando la homografía calibrada.
   b. Llamar a _b_compute_occupancy(warped, empty_warped, threshold) para
      obtener occ_grid y team_grid.
   c. Filtrar ruido con el buffer de estabilidad (_STABLE_FRAMES consecutivos).
   d. Comparar el estado estable con el último estado registrado.
   e. Si hay cambio, invocar game_state_tracker.process_state(occ_grid,
      team_grid) para que detecte y registre el movimiento.
4. Dibujar HUD sobre el frame:
   - Estado de calibración y tracking (ON / STALE / OFF).
   - Último movimiento detectado en notación SAN.
   - Número de movimientos registrados y turno actual (blancas/negras).
   - Instrucciones de teclas.
5. Mostrar el frame (ventana principal) y el tablero aplanado con overlay de
   ocupación (ventana secundaria, equivalente al modo B de camera_pipeline).
6. Procesar teclas (R, I, S, Q).

Dependencias
------------
- camera_pipeline (funciones de tracking y visión):
    _detect_red_border_corners, _update_red_border_tracking,
    _b_compute_occupancy, _b_build_display
    Estados compartidos: _calibration, _tracking_state
- board_detector.BoardDetector (warp y corner detection)
- game_state_tracker.GameStateTracker (lógica de partida)
- python-chess (exportación PGN, via game_state_tracker)
- opencv-python, numpy

Archivos generados
------------------
- tablero_vacio.jpg       referencia del tablero vacío (creado por R)
- partida_YYYYMMDD_HHMMSS.pgn  exportación de la partida (creado por S)

Notas para Fase 2
-----------------
- Separar camera_pipeline en un módulo reutilizable (vision_core.py) para que
  tracker_app y camera_pipeline compartan el código de visión sin duplicarlo.
- Evaluar si el buffer de estabilidad de camera_pipeline es suficiente para
  el tracking o si se necesita un buffer más largo (más frames = menos falsos
  positivos pero más latencia en la detección).
- El overlay del tablero aplanado en la ventana secundaria podría incluir flechas
  que muestren el último movimiento detectado (casilla origen → casilla destino).
"""
