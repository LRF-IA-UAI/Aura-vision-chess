"""
generate_aruco_markers.py — Genera marcadores ArUco para tracking continuo de perspectiva.
Robot Ajedrecista CAETI (UAI)

Salida:
    aruco_marker_0.png .. aruco_marker_3.png  (600×600 px, alta res para imprimir)
    aruco_markers_imprimir.pdf                 (A4 con los 4 marcadores a 4cm×4cm)

Uso:
    python generate_aruco_markers.py

Montaje en el tablero (vista desde la cámara):
    ID 0 → esquina superior-izquierda
    ID 1 → esquina superior-derecha
    ID 2 → esquina inferior-derecha
    ID 3 → esquina inferior-izquierda
"""

import sys
import os
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Compatibilidad OpenCV 4.6 / 4.7+
# ---------------------------------------------------------------------------

def _get_aruco_dict():
    try:
        return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    except AttributeError:
        return cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)


def _make_marker(aruco_dict, marker_id: int, size: int) -> np.ndarray:
    """Genera imagen de marcador ArUco; maneja API antigua y nueva de OpenCV."""
    try:
        return cv2.aruco.generateImageMarker(aruco_dict, marker_id, size)
    except AttributeError:
        return cv2.aruco.drawMarker(aruco_dict, marker_id, size)


# ---------------------------------------------------------------------------
# Generación de PNGs individuales (600×600 px)
# ---------------------------------------------------------------------------

def generate_png_markers(output_dir: str = ".") -> dict:
    """Genera aruco_marker_0.png .. aruco_marker_3.png a 600×600 px."""
    aruco_dict = _get_aruco_dict()
    markers = {}
    for mid in range(4):
        img = _make_marker(aruco_dict, mid, 600)
        path = os.path.join(output_dir, f"aruco_marker_{mid}.png")
        cv2.imwrite(path, img)
        markers[mid] = img
        print(f"  [OK] {path}")
    return markers


# ---------------------------------------------------------------------------
# PDF con reportlab (preferido — control preciso del tamaño de impresión)
# ---------------------------------------------------------------------------

def _generate_pdf_reportlab(markers: dict, output_path: str):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib import colors
    import tempfile

    c = rl_canvas.Canvas(output_path, pagesize=A4)
    page_w, page_h = A4          # 595.27 × 841.89 puntos

    MARKER_SIZE = 4 * cm         # 4 cm en puntos
    GAP         = 1.5 * cm      # separación entre marcadores
    MARGIN_X    = 2.5 * cm
    MARGIN_Y_T  = 3.0 * cm      # margen desde arriba
    LABEL_H     = 0.75 * cm     # altura reservada para etiqueta

    # Título
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(page_w / 2, page_h - MARGIN_Y_T * 0.55,
                        "Marcadores ArUco — Robot Ajedrecista CAETI (UAI)")

    # Instrucciones
    c.setFont("Helvetica", 9)
    c.drawCentredString(
        page_w / 2, page_h - MARGIN_Y_T * 0.55 - 16,
        "Imprimir al 100% sin 'ajustar a pagina'. "
        "Borde blanco es parte del marcador — no recortar.")
    c.drawCentredString(
        page_w / 2, page_h - MARGIN_Y_T * 0.55 - 30,
        "Pegar sobre el BORDE del tablero: ID0=TL  ID1=TR  ID2=BR  ID3=BL "
        "(vista desde la camara)")

    # Distribución 2×2 que refleja la posición física en el tablero:
    #   col=0 → izquierda   col=1 → derecha
    #   row=0 → arriba      row=1 → abajo
    layout = [(0, 0, 0), (1, 1, 0), (3, 0, 1), (2, 1, 1)]

    for mid, col, row in layout:
        x = MARGIN_X + col * (MARKER_SIZE + GAP)
        y = (page_h - MARGIN_Y_T - 40
             - row * (MARKER_SIZE + LABEL_H + GAP)
             - MARKER_SIZE)

        # Guardar marcador en PNG temporal (reportlab necesita un archivo)
        img_rgb = cv2.cvtColor(markers[mid], cv2.COLOR_GRAY2RGB)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            cv2.imwrite(tf.name, img_rgb)
            tmp = tf.name

        c.drawImage(tmp, x, y, width=MARKER_SIZE, height=MARKER_SIZE,
                    preserveAspectRatio=True)
        os.unlink(tmp)

        # Líneas de corte punteadas
        cut = 0.25 * cm
        c.setStrokeColor(colors.grey)
        c.setLineWidth(0.4)
        c.setDash([3, 3])
        c.rect(x - cut, y - cut,
               MARKER_SIZE + 2 * cut, MARKER_SIZE + 2 * cut)
        c.setDash()

        # Etiqueta debajo del marcador
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(colors.black)
        c.drawCentredString(x + MARKER_SIZE / 2, y - LABEL_H + 3, f"ID {mid}")

    c.save()
    print(f"  [OK] {output_path}  (reportlab, marcadores a 4cm x 4cm)")


# ---------------------------------------------------------------------------
# PDF con matplotlib (fallback)
# ---------------------------------------------------------------------------

def _generate_pdf_matplotlib(markers: dict, output_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # A4 en pulgadas (8.27 × 11.69)
    A4_W_IN, A4_H_IN = 8.27, 11.69
    A4_W_CM, A4_H_CM = 21.0, 29.7

    fig = plt.figure(figsize=(A4_W_IN, A4_H_IN), facecolor="white")

    # Conversión cm → fracción de figura
    def fw(x): return x / A4_W_CM
    def fh(x): return x / A4_H_CM

    MARKER_CM = 4.0
    GAP_CM    = 1.5
    MX_CM     = 2.5    # margen izquierdo
    MT_CM     = 4.0    # margen desde arriba (deja espacio para título)
    LABEL_CM  = 0.6

    # Distribución que refleja posición física:
    #   ID 0=TL, ID 1=TR, ID 3=BL, ID 2=BR
    layout = [(0, 0, 0), (1, 1, 0), (3, 0, 1), (2, 1, 1)]

    for mid, col, row in layout:
        x_cm = MX_CM + col * (MARKER_CM + GAP_CM)
        # y desde arriba → convertir a fracción desde abajo
        y_top_cm = MT_CM + row * (MARKER_CM + LABEL_CM + GAP_CM)
        y_bot_cm = A4_H_CM - y_top_cm - MARKER_CM

        left   = fw(x_cm)
        bottom = fh(y_bot_cm)
        width  = fw(MARKER_CM)
        height = fh(MARKER_CM)

        ax = fig.add_axes([left, bottom, width, height])
        ax.imshow(markers[mid], cmap="gray", vmin=0, vmax=255,
                  interpolation="nearest")
        ax.axis("off")

        # Borde punteado (línea de corte visual)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#888888")
            spine.set_linewidth(0.8)
            spine.set_linestyle("--")

        # Etiqueta debajo del marcador
        fig.text(left + width / 2,
                 fh(y_bot_cm) - fh(LABEL_CM) * 0.55,
                 f"ID {mid}",
                 ha="center", va="center",
                 fontsize=9, fontweight="bold", color="black")

    # Título
    fig.text(0.5, 1 - fh(1.2),
             "Marcadores ArUco — Robot Ajedrecista CAETI (UAI)",
             ha="center", va="top", fontsize=11, fontweight="bold")
    fig.text(0.5, 1 - fh(2.0),
             "Imprimir al 100% sin 'ajustar a pagina'. "
             "Borde blanco es parte del marcador — no recortar.",
             ha="center", va="top", fontsize=8)
    fig.text(0.5, 1 - fh(2.7),
             "Pegar sobre el BORDE del tablero: ID0=TL  ID1=TR  ID2=BR  ID3=BL "
             "(vista desde la camara)",
             ha="center", va="top", fontsize=8)

    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, dpi=150)
    plt.close(fig)
    print(f"  [OK] {output_path}  (matplotlib, ~4cm x 4cm al imprimir en A4)")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def main():
    print("Generando marcadores ArUco — DICT_4X4_50, IDs 0-3")
    print()
    print("PNGs (600×600 px):")
    markers = generate_png_markers(".")

    pdf_path = "aruco_markers_imprimir.pdf"
    print()
    print("PDF (A4):")
    try:
        _generate_pdf_reportlab(markers, pdf_path)
    except ImportError:
        try:
            _generate_pdf_matplotlib(markers, pdf_path)
        except ImportError:
            print("  [WARN] Ninguna librería PDF disponible.")
            print("         Instalar: pip install reportlab")
            print("         O bien:   pip install matplotlib")

    print()
    print("Archivos generados:")
    for mid in range(4):
        print(f"  aruco_marker_{mid}.png")
    print(f"  {pdf_path}")
    print()
    print("Montaje en el tablero (vista desde la camara):")
    print("  ID 0 → esquina superior-izquierda (TL)")
    print("  ID 1 → esquina superior-derecha   (TR)")
    print("  ID 2 → esquina inferior-derecha   (BR)")
    print("  ID 3 → esquina inferior-izquierda (BL)")


if __name__ == "__main__":
    main()
