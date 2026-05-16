"""
board_detector.py
Módulo de visión por computadora para detección del tablero de ajedrez.
Proyecto: Robot Ajedrecista - CAETI (UAI)
"""

import cv2
import numpy as np


class BoardDetector:
    """
    Detecta un tablero de ajedrez en una imagen y genera el FEN correspondiente.
    Pipeline: imagen BGR → detección → perspectiva → celdas → clasificación → FEN
    """

    def __init__(self):
        # Tamaño de salida del tablero aplanado (debe ser múltiplo de 8)
        self.board_size = 800
        self.cell_size = self.board_size // 8  # 100 píxeles por celda

        # Área mínima del contorno para considerarse un tablero (evita detecciones falsas)
        self.min_board_area = 10000

        # Fracción de la celda que se usa para clasificación (0.6 = 60% central)
        self.cell_roi_fraction = 0.6

    # -------------------------------------------------------------------------
    # 1. Detección del tablero y corrección de perspectiva
    # -------------------------------------------------------------------------

    def detect_board(self, image: np.ndarray) -> np.ndarray | None:
        """
        Detecta el tablero de ajedrez en la imagen y devuelve una vista cenital 800x800.

        Intenta 3 estrategias en orden de prioridad:
          1. Detección por grilla de ajedrez (findChessboardCorners / findChessboardCornersSB)
          2. Detección por líneas Hough sobre imagen en escala de grises + threshold adaptativo
          3. Detección por contorno mejorada (múltiples thresholds, filtro aspect ratio)

        Args:
            image: Imagen BGR de OpenCV (numpy array)

        Returns:
            np.ndarray 800x800 con el tablero aplanado, o None si no se detecta
        """
        if image is None or image.size == 0:
            return None

        corners, strategy = self._detect_board_corners(image)
        if corners is None:
            return None

        return self._warp_from_corners(image, corners)

    def _detect_board_corners(self, image: np.ndarray) -> tuple:
        """
        Intenta detectar las 4 esquinas del tablero usando las 3 estrategias.

        Returns:
            (corners np.ndarray shape (4,2), strategy_name str)
            o (None, None) si ninguna estrategia funcionó
        """
        corners = self._detect_by_chessboard_corners(image)
        if corners is not None:
            return corners, "chessboard_corners"

        corners = self._detect_by_hough_lines(image)
        if corners is not None:
            return corners, "hough_lines"

        corners = self._detect_by_contour_improved(image)
        if corners is not None:
            return corners, "contour_improved"

        return None, None

    # -------------------------------------------------------------------------
    # Estrategia 1: findChessboardCorners / findChessboardCornersSB
    # -------------------------------------------------------------------------

    def _detect_by_chessboard_corners(self, image: np.ndarray) -> np.ndarray | None:
        """
        Busca el patrón interno de 7x7 esquinas interiores del tablero y
        extrapola las 4 esquinas externas del área de juego.

        Returns:
            Array (4, 2) con las 4 esquinas externas, o None
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pattern = (7, 7)

        # Probar con distintas escalas para mejorar la detección
        for scale in [1.0, 0.75, 0.5]:
            h, w = gray.shape
            if scale != 1.0:
                scaled = cv2.resize(gray, (int(w * scale), int(h * scale)))
            else:
                scaled = gray.copy()

            # Ecualizar histograma para mejorar contraste
            scaled_eq = cv2.equalizeHist(scaled)

            ret, corners = None, None

            # Intentar primero findChessboardCornersSB (más robusto, OpenCV 4+)
            try:
                ret, corners = cv2.findChessboardCornersSB(scaled_eq, pattern)
                if not ret:
                    ret, corners = cv2.findChessboardCornersSB(scaled, pattern)
            except cv2.error:
                ret = False

            # Si falla, usar findChessboardCorners estándar
            if not ret:
                flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
                         cv2.CALIB_CB_NORMALIZE_IMAGE +
                         cv2.CALIB_CB_FAST_CHECK)
                ret, corners = cv2.findChessboardCorners(scaled_eq, pattern, flags)
                if not ret:
                    ret, corners = cv2.findChessboardCorners(scaled, pattern, flags)

            if not ret or corners is None:
                continue

            # Escalar las esquinas de vuelta a la resolución original
            if scale != 1.0:
                corners = corners / scale

            # Refinar posición de las esquinas con subpixel
            corners_f32 = corners.reshape(-1, 1, 2).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_f32 = cv2.cornerSubPix(gray, corners_f32, (11, 11), (-1, -1), criteria)

            # Reorganizar en grilla 7x7: corners_2d[fila][col] = (x, y)
            corners_2d = corners_f32.reshape(7, 7, 2)

            # Vectores de paso: un paso en dirección columna y fila
            step_col = (corners_2d[0, 6] - corners_2d[0, 0]) / 6.0
            step_row = (corners_2d[6, 0] - corners_2d[0, 0]) / 6.0

            # Extrapolar una celda hacia afuera en cada dirección
            tl = corners_2d[0, 0] - step_col - step_row
            tr = corners_2d[0, 6] + step_col - step_row
            br = corners_2d[6, 6] + step_col + step_row
            bl = corners_2d[6, 0] - step_col + step_row

            outer_corners = np.array([tl, tr, br, bl], dtype=np.float32)

            # Validar que las esquinas estén dentro de los límites de la imagen
            ih, iw = image.shape[:2]
            margin = -50  # permitir un margen pequeño fuera
            if np.any(outer_corners < margin) or np.any(outer_corners[:, 0] > iw - margin) or \
               np.any(outer_corners[:, 1] > ih - margin):
                continue

            return outer_corners.reshape(4, 2)

        return None

    # -------------------------------------------------------------------------
    # Estrategia 2: threshold adaptativo + HoughLinesP
    # -------------------------------------------------------------------------

    def _detect_by_hough_lines(self, image: np.ndarray) -> np.ndarray | None:
        """
        Detecta la grilla del tablero buscando líneas horizontales y verticales
        regulares con transformada de Hough.

        Returns:
            Array (4, 2) con las 4 esquinas del área de juego, o None
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_h, img_w = gray.shape
        img_area = img_h * img_w

        # Preprocesamiento: suavizar + threshold adaptativo para resaltar la grilla
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        edges = cv2.Canny(thresh, 30, 100, apertureSize=3)

        # Dilatar para conectar bordes discontinuos
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=60,
            minLineLength=min(img_w, img_h) * 0.15,
            maxLineGap=30
        )

        if lines is None or len(lines) < 8:
            return None

        # Clasificar líneas en horizontales y verticales
        h_positions = []  # coordenada y central de cada línea horizontal
        v_positions = []  # coordenada x central de cada línea vertical

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
            if angle < 20:          # horizontal
                h_positions.append((y1 + y2) / 2.0)
            elif angle > 70:        # vertical
                v_positions.append((x1 + x2) / 2.0)

        if len(h_positions) < 4 or len(v_positions) < 4:
            return None

        h_clusters = self._cluster_lines(sorted(h_positions), gap=img_h * 0.04)
        v_clusters = self._cluster_lines(sorted(v_positions), gap=img_w * 0.04)

        if len(h_clusters) < 2 or len(v_clusters) < 2:
            return None

        # Tomar líneas más extremas como bordes del tablero
        top    = min(h_clusters)
        bottom = max(h_clusters)
        left   = min(v_clusters)
        right  = max(v_clusters)

        board_w = right - left
        board_h = bottom - top

        if board_w <= 0 or board_h <= 0:
            return None

        # Filtrar por aspect ratio (tablero aproximadamente cuadrado: 0.7 a 1.3)
        ratio = board_w / board_h
        if not (0.7 < ratio < 1.3):
            return None

        board_area = board_w * board_h
        if board_area < 0.15 * img_area:
            return None

        corners = np.array([
            [left,  top],
            [right, top],
            [right, bottom],
            [left,  bottom]
        ], dtype=np.float32)

        return corners

    def _cluster_lines(self, positions: list, gap: float) -> list:
        """
        Agrupa posiciones cercanas en clústeres y devuelve la mediana de cada grupo.
        """
        if not positions:
            return []
        clusters = []
        group = [positions[0]]
        for pos in positions[1:]:
            if pos - group[-1] < gap:
                group.append(pos)
            else:
                clusters.append(float(np.median(group)))
                group = [pos]
        clusters.append(float(np.median(group)))
        return clusters

    # -------------------------------------------------------------------------
    # Estrategia 3: detección por contorno mejorada
    # -------------------------------------------------------------------------

    def _detect_by_contour_improved(self, image: np.ndarray) -> np.ndarray | None:
        """
        Detecta el contorno cuadrangular más grande de la imagen probando
        múltiples valores de threshold y epsilon para approxPolyDP.
        Filtra por aspect ratio (~cuadrado) y área mínima (15% de la imagen).

        Returns:
            Array (4, 2) con las 4 esquinas, o None
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        img_h, img_w = gray.shape
        img_area = img_h * img_w
        min_area = 0.15 * img_area

        # Máscara para reducir la interferencia del marco rojo
        masked = self._mask_dominant_color(image, blur)

        kernel = np.ones((3, 3), np.uint8)

        # Lista de configuraciones de preprocesamiento para probar
        preproc_variants = []

        # Variantes con distintos umbrales de Canny
        for low, high in [(30, 90), (50, 150), (80, 200), (100, 250)]:
            edges = cv2.Canny(blur, low, high)
            edges = cv2.dilate(edges, kernel, iterations=1)
            preproc_variants.append(edges)

        # Variante con Otsu
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges_otsu = cv2.Canny(otsu, 50, 150)
        edges_otsu = cv2.dilate(edges_otsu, kernel, iterations=1)
        preproc_variants.append(edges_otsu)

        # Variante con imagen sin el color dominante (marco rojo)
        if masked is not None:
            masked_blur = cv2.GaussianBlur(masked, (5, 5), 0)
            for low, high in [(50, 150), (80, 200)]:
                edges_m = cv2.Canny(masked_blur, low, high)
                edges_m = cv2.dilate(edges_m, kernel, iterations=1)
                preproc_variants.append(edges_m)

        for edges in preproc_variants:
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

            for contour in contours_sorted[:10]:
                area = cv2.contourArea(contour)
                if area < min_area:
                    break

                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue

                # Probar distintos valores de epsilon
                for eps_factor in [0.01, 0.02, 0.03, 0.04, 0.05]:
                    approx = cv2.approxPolyDP(
                        contour, eps_factor * perimeter, True
                    )

                    if len(approx) != 4:
                        continue

                    pts = approx.reshape(4, 2).astype(np.float32)

                    # Filtrar por aspect ratio (tablero aproximadamente cuadrado)
                    x, y, w, h = cv2.boundingRect(approx)
                    if h == 0:
                        continue
                    ratio = w / h
                    if not (0.8 <= ratio <= 1.2):
                        continue

                    return pts

        return None

    def _mask_dominant_color(self, image: np.ndarray, gray: np.ndarray) -> np.ndarray | None:
        """
        Crea una versión en escala de grises con el color dominante del marco
        (detectado automáticamente) reemplazado por gris neutro.
        Ayuda cuando hay un marco de color que interfiere con los contornos.
        """
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Detectar píxeles muy saturados (probablemente el marco de color)
            saturation = hsv[:, :, 1]
            sat_mask = saturation > 100

            if sat_mask.sum() < 1000:
                return None

            # Obtener el hue dominante entre los píxeles saturados
            hues = hsv[:, :, 0][sat_mask]
            hist, bins = np.histogram(hues, bins=18, range=(0, 180))
            dominant_bin = np.argmax(hist)
            dom_hue_low  = int(bins[dominant_bin])
            dom_hue_high = int(bins[dominant_bin + 1])

            # Ampliar el rango de hue para capturar bien el color
            margin = 10
            lo = max(0, dom_hue_low - margin)
            hi = min(180, dom_hue_high + margin)

            mask = cv2.inRange(hsv, (lo, 80, 80), (hi, 255, 255))

            # Para rojo (hue cerca de 0 o 180), incluir ambos extremos
            if dom_hue_low <= 10 or dom_hue_high >= 170:
                mask2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
                mask3 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
                mask = cv2.bitwise_or(mask, cv2.bitwise_or(mask2, mask3))

            masked_gray = gray.copy()
            masked_gray[mask > 0] = 128
            return masked_gray

        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Helper: aplicar transformación de perspectiva
    # -------------------------------------------------------------------------

    def _compute_perspective_matrix(self, corners: np.ndarray) -> np.ndarray:
        """
        Calcula la matriz de perspectiva que transforma las 4 esquinas del
        tablero al cuadrado de salida (board_size x board_size).

        Args:
            corners: Array (4, 2) en cualquier orden

        Returns:
            Matriz 3x3 de perspectiva (float64)
        """
        esquinas = self._ordenar_esquinas(corners.astype(np.float32))
        destino = np.array([
            [0, 0],
            [self.board_size - 1, 0],
            [self.board_size - 1, self.board_size - 1],
            [0, self.board_size - 1]
        ], dtype=np.float32)
        return cv2.getPerspectiveTransform(esquinas, destino)

    def _warp_from_corners(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Aplica warpPerspective usando las 4 esquinas detectadas.
        Después recorta un margen interior del 3% de cada lado para eliminar
        el borde decorativo (ej. rojo) que queda incluido en la transformación.

        Args:
            image:   Imagen original BGR
            corners: Array (4, 2) con las esquinas (en cualquier orden)

        Returns:
            Imagen board_size x board_size con el tablero aplanado y borde recortado
        """
        M = self._compute_perspective_matrix(corners)
        warped = cv2.warpPerspective(image, M, (self.board_size, self.board_size))

        # Recortar borde decorativo: 5% de cada lado
        margin = int(self.board_size * 0.05)
        warped = warped[margin:self.board_size - margin,
                        margin:self.board_size - margin]
        warped = cv2.resize(warped, (self.board_size, self.board_size),
                            interpolation=cv2.INTER_AREA)
        return warped

    def _ordenar_esquinas(self, puntos: np.ndarray) -> np.ndarray:
        """
        Ordena 4 puntos en el orden: top-left, top-right, bottom-right, bottom-left.

        Usa el centroide como referencia para clasificar cada punto según su
        posición relativa. Este método es robusto ante rotaciones de la imagen,
        a diferencia del método suma/diferencia que falla cuando el tablero
        está inclinado.

        Args:
            puntos: Array de shape (4, 2) con las coordenadas

        Returns:
            Array de shape (4, 2) ordenado: TL, TR, BR, BL
        """
        centroide = puntos.mean(axis=0)          # (cx, cy)
        dx = puntos[:, 0] - centroide[0]         # desplazamiento horizontal al centro
        dy = puntos[:, 1] - centroide[1]         # desplazamiento vertical al centro

        ordenados = np.zeros((4, 2), dtype=np.float32)
        # Top-left:     x-cy < 0  y  y-cy < 0  →  x+y mínimo relativo al centroide
        ordenados[0] = puntos[np.argmin(dx + dy)]   # top-left
        # Top-right:    x-cy > 0  y  y-cy < 0  →  x-y mínimo (derecha y arriba)
        ordenados[1] = puntos[np.argmin(dy - dx)]   # top-right  (min de y-x ≡ min de -(x-y))
        # Bottom-right: x-cy > 0  y  y-cy > 0  →  x+y máximo relativo al centroide
        ordenados[2] = puntos[np.argmax(dx + dy)]   # bottom-right
        # Bottom-left:  x-cy < 0  y  y-cy > 0  →  x-y máximo (izquierda y abajo)
        ordenados[3] = puntos[np.argmax(dy - dx)]   # bottom-left (max de y-x)

        return ordenados

    # -------------------------------------------------------------------------
    # 2. División en celdas
    # -------------------------------------------------------------------------

    def split_into_cells(self, board_image: np.ndarray,
                         original_image: np.ndarray | None = None,
                         perspective_M: np.ndarray | None = None) -> list[np.ndarray]:
        """
        Divide el tablero en 64 celdas iguales con margen interior del 8%.

        Cuando se proporcionan `original_image` y `perspective_M`, cada celda
        se extrae directamente de la imagen original mediante su propia
        transformación de perspectiva (más precisa que slicing sobre el warp).
        Sin esos parámetros, hace slice + resize sobre board_image.

        Orden: a8→h8, a7→h7, ..., a1→h1.
        Salida: 64 arrays de exactamente 80x80 px BGR.

        Args:
            board_image:   Imagen aplanada del tablero (resultado de warpPerspective)
            original_image: Imagen original BGR (opcional)
            perspective_M:  Matriz 3x3 de perspectiva usada para crear board_image (opcional)

        Returns:
            Lista de 64 arrays 80x80 BGR
        """
        out_size     = 80
        margin_frac  = 0.08
        cell_board   = self.board_size / 8   # tamaño de celda en board-space (100 px)

        use_precise = (original_image is not None) and (perspective_M is not None)
        M_inv = np.linalg.inv(perspective_M) if use_precise else None

        dst_corners = np.array([
            [0,            0],
            [out_size - 1, 0],
            [out_size - 1, out_size - 1],
            [0,            out_size - 1],
        ], dtype=np.float32)

        celdas = []
        for fila in range(8):
            for col in range(8):
                if use_precise:
                    # Esquinas de la celda en board-space (con margen)
                    mx = cell_board * margin_frac
                    my = cell_board * margin_frac
                    bx0 = col  * cell_board + mx
                    by0 = fila * cell_board + my
                    bx1 = (col  + 1) * cell_board - mx
                    by1 = (fila + 1) * cell_board - my

                    board_pts = np.array([
                        [bx0, by0], [bx1, by0],
                        [bx1, by1], [bx0, by1],
                    ], dtype=np.float32)

                    # Proyectar a imagen original mediante M_inv
                    orig_pts = cv2.perspectiveTransform(
                        board_pts.reshape(1, -1, 2), M_inv
                    )[0]

                    M_cell = cv2.getPerspectiveTransform(orig_pts, dst_corners)
                    celda  = cv2.warpPerspective(
                        original_image, M_cell, (out_size, out_size),
                        flags=cv2.INTER_LINEAR
                    )
                else:
                    h, w = board_image.shape[:2]
                    ch, cw = h / 8, w / 8
                    y0, x0 = fila * ch, col * cw
                    y1 = int(round(y0 + ch * margin_frac))
                    y2 = int(round(y0 + ch * (1 - margin_frac)))
                    x1 = int(round(x0 + cw * margin_frac))
                    x2 = int(round(x0 + cw * (1 - margin_frac)))
                    celda = cv2.resize(board_image[y1:y2, x1:x2],
                                       (out_size, out_size),
                                       interpolation=cv2.INTER_AREA)

                celdas.append(celda)

        return celdas

    # -------------------------------------------------------------------------
    # Visualización del grid sobre el tablero aplanado
    # -------------------------------------------------------------------------

    def visualize_grid(self, board_image: np.ndarray,
                       cells: list[np.ndarray]) -> np.ndarray:
        """
        Dibuja el grid 8x8 sobre el tablero aplanado con notación algebraica.

        Por cada casilla dibuja:
          - Líneas blancas finas delimitando la celda bruta
          - Etiqueta de notación algebraica (a8..h1) en la esquina superior
            izquierda de cada casilla

        Args:
            board_image: Imagen aplanada del tablero
            cells:       Lista de 64 celdas (usada solo para validar el count)

        Returns:
            Copia de board_image con grid y etiquetas superpuestos
        """
        anotado = board_image.copy()
        h, w = anotado.shape[:2]
        cell_h = h / 8
        cell_w = w / 8

        columnas = list("abcdefgh")

        # Líneas del grid
        for i in range(9):
            x = int(round(i * cell_w))
            y = int(round(i * cell_h))
            cv2.line(anotado, (x, 0),    (x, h),    (255, 255, 255), 1)
            cv2.line(anotado, (0, y),    (w, y),    (255, 255, 255), 1)

        # Notación algebraica en cada celda
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        thickness  = 1
        for fila in range(8):
            for col in range(8):
                numero = 8 - fila          # fila 0 → rango 8, fila 7 → rango 1
                letra  = columnas[col]
                etiq   = f"{letra}{numero}"

                x_text = int(round(col * cell_w)) + 4
                y_text = int(round(fila * cell_h)) + 14

                # Sombra negra para legibilidad
                cv2.putText(anotado, etiq,
                            (x_text + 1, y_text + 1), font, font_scale,
                            (0, 0, 0), thickness + 1, cv2.LINE_AA)
                # Texto blanco
                cv2.putText(anotado, etiq,
                            (x_text, y_text), font, font_scale,
                            (255, 255, 255), thickness, cv2.LINE_AA)

        return anotado

    # -------------------------------------------------------------------------
    # DEBUG: visualizar esquinas detectadas
    # -------------------------------------------------------------------------

    def debug_corners(self, image: np.ndarray, save_path: str = "debug_corners.jpg") -> np.ndarray:
        """
        Detecta las 4 esquinas del tablero, las dibuja sobre la imagen original
        con colores distintos y numeradas, imprime coordenadas en consola y
        guarda el resultado en disco.

        Orden de colores:
          1 (top-left)     → rojo   (0, 0, 255)
          2 (top-right)    → verde  (0, 255, 0)
          3 (bottom-right) → azul   (255, 0, 0)
          4 (bottom-left)  → amarillo (0, 255, 255)

        Args:
            image:     Imagen BGR de OpenCV
            save_path: Ruta donde guardar la imagen de debug

        Returns:
            Imagen con las esquinas dibujadas (o la imagen original si no se detectaron)
        """
        debug = image.copy()

        corners_raw, strategy = self._detect_board_corners(image)
        if corners_raw is None:
            print("[DEBUG CORNERS] No se detectaron esquinas en la imagen.")
            cv2.imwrite(save_path, debug)
            return debug

        esquinas = self._ordenar_esquinas(corners_raw.astype(np.float32))

        nombres = ["top-left", "top-right", "bottom-right", "bottom-left"]
        colores = [
            (0,   0,   255),   # 1 top-left:     rojo
            (0,   255,   0),   # 2 top-right:    verde
            (255,   0,   0),   # 3 bottom-right: azul
            (0,   255, 255),   # 4 bottom-left:  amarillo
        ]

        print("\n" + "=" * 55)
        print(f"  ESQUINAS DETECTADAS  (estrategia: {strategy})")
        print("=" * 55)
        print(f"  {'#':<4} {'Rol':<14} {'X':>7} {'Y':>7}  Color")
        print("  " + "-" * 50)

        for i, (punto, nombre, color) in enumerate(zip(esquinas, nombres, colores), start=1):
            x, y = int(punto[0]), int(punto[1])
            color_nombre = ["rojo", "verde", "azul", "amarillo"][i - 1]
            print(f"  {i:<4} {nombre:<14} {x:>7} {y:>7}  {color_nombre}")

            # Círculo relleno + borde blanco para visibilidad
            cv2.circle(debug, (x, y), 14, (255, 255, 255), -1)
            cv2.circle(debug, (x, y), 12, color, -1)

            # Número y nombre del rol
            cv2.putText(debug, str(i),
                        (x - 5, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(debug, nombre,
                        (x + 18, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(debug, nombre,
                        (x + 18, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        print("=" * 55 + "\n")

        # Polígono del contorno detectado
        pts_int = esquinas.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(debug, [pts_int], isClosed=True, color=(0, 255, 0), thickness=2)

        # Etiqueta de estrategia
        strategy_labels = {
            "chessboard_corners": "Est.1: Grilla ajedrez",
            "hough_lines":        "Est.2: Lineas Hough",
            "contour_improved":   "Est.3: Contorno mejorado",
        }
        label = strategy_labels.get(strategy, strategy)
        cv2.putText(debug, label, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imwrite(save_path, debug)
        print(f"[DEBUG CORNERS] Imagen guardada en: {save_path}")

        return debug

    def get_cells_grid_image(self, board_image: np.ndarray, cells: list[str]) -> np.ndarray:
        """
        Genera una imagen visual del grid de 64 celdas con su clasificación.
        Útil para debug y calibración.

        Args:
            board_image: Tablero aplanado 800x800
            cells: Lista de 64 clasificaciones

        Returns:
            Imagen 800x800 con el tablero y el color de clasificación superpuesto
        """
        grid = board_image.copy()

        for idx, clasificacion in enumerate(cells):
            fila = idx // 8
            columna = idx % 8
            x = columna * self.cell_size
            y = fila * self.cell_size

            # Color del overlay según clasificación
            if clasificacion == "white":
                color = (200, 200, 255)   # Azul claro para piezas blancas
            elif clasificacion == "black":
                color = (255, 100, 100)   # Rojo claro para piezas negras
            else:
                continue  # No dibujar overlay en casillas vacías

            overlay = grid.copy()
            cv2.rectangle(overlay, (x, y), (x + self.cell_size, y + self.cell_size), color, -1)
            cv2.addWeighted(overlay, 0.3, grid, 0.7, 0, grid)

            # Etiqueta de texto
            label = "W" if clasificacion == "white" else "B"
            cv2.putText(grid, label,
                        (x + 38, y + 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Dibujar líneas del grid
        for i in range(9):
            pos = i * self.cell_size
            cv2.line(grid, (pos, 0), (pos, self.board_size), (50, 50, 50), 1)
            cv2.line(grid, (0, pos), (self.board_size, pos), (50, 50, 50), 1)

        return grid
