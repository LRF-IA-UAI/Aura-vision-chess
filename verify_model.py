import sys
import torch

# PyTorch 2.6+ cambió weights_only=True por default, lo que rompe
# el formato de checkpoints de ultralytics. Parche local seguro:
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})

from ultralytics import YOLO
import cv2

image_path = sys.argv[1] if len(sys.argv) > 1 else "foto_tablero.jpg"

# --- Cargar modelo ---
model = YOLO("chess_model.pt")

# --- Clases disponibles ---
names = model.names
print("=" * 50)
print(f"Clases del modelo ({len(names)} total):")
for idx, name in names.items():
    print(f"  {idx:2d}: {name}")
print("=" * 50)

# --- Inferencia ---
results = model(image_path, conf=0.25)
result = results[0]

boxes  = result.boxes
total  = len(boxes)
print(f"\nDetecciones en '{image_path}': {total} piezas\n")

from collections import Counter
counts = Counter()
for box in boxes:
    cls_id = int(box.cls)
    counts[names[cls_id]] += 1

if counts:
    print("Por clase:")
    for cls_name, n in sorted(counts.items()):
        print(f"  {cls_name:<30} {n}")
else:
    print("  (ninguna)")

print("=" * 50)

# --- Guardar imagen con bboxes ---
annotated = result.plot()
out_path = "verify_result.jpg"
cv2.imwrite(out_path, annotated)
print(f"\nImagen anotada guardada en: {out_path}")

# --- Mostrar ventana ---
cv2.imshow("Detecciones - verify_model", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
