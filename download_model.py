import requests
import os
import sys

OUTPUT = "chess_model.pt"
downloaded = False

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# --- Intento 1: huggingface_hub (no requiere token para repos públicos) ---
print("=" * 60)
print("Intento 1: huggingface_hub library (sin token)")
print("=" * 60)

HF_REPOS = [
    ("arnabdhar/YOLOv8-nano-chess",              "best.pt"),
    ("keremberke/yolov8n-chess-pieces-detection", "best.pt"),
    ("keremberke/yolov8s-chess-pieces-detection", "best.pt"),
    ("Dibol/chess_detection",                    "best.pt"),
    ("foduucom/stockfish-part-1-yolov8l",        "best.pt"),
]

try:
    from huggingface_hub import hf_hub_download
    for repo_id, filename in HF_REPOS:
        print(f"  Probando repo: {repo_id}")
        try:
            cached = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=None,
                local_dir=".",
                local_dir_use_symlinks=False,
            )
            if os.path.exists(cached):
                if cached != OUTPUT:
                    os.replace(cached, OUTPUT)
                size = os.path.getsize(OUTPUT)
                print(f"  OK: {repo_id} ({size} bytes) -> {OUTPUT}")
                downloaded = True
                break
        except Exception as e:
            print(f"  Error: {e}")
except ImportError:
    print("  huggingface_hub no instalado, saltando.")

# --- Intento 2: requests con headers de navegador ---
if not downloaded:
    print()
    print("=" * 60)
    print("Intento 2: requests con User-Agent de navegador")
    print("=" * 60)

    URLS = [
        "https://huggingface.co/arnabdhar/YOLOv8-nano-chess/resolve/main/best.pt",
        "https://huggingface.co/keremberke/yolov8n-chess-pieces-detection/resolve/main/best.pt",
        "https://huggingface.co/keremberke/yolov8s-chess-pieces-detection/resolve/main/best.pt",
        "https://huggingface.co/Dibol/chess_detection/resolve/main/best.pt",
        "https://github.com/andrewda/chess-cv/releases/download/v1.0/best.pt",
        "https://github.com/gurcuff91/chess-vision/releases/download/v0.1.0/chess_model.pt",
    ]

    for url in URLS:
        print(f"  Probando: {url}")
        try:
            r = requests.get(url, stream=True, timeout=30, headers=HEADERS, allow_redirects=True)
            print(f"    Status: {r.status_code}")
            if r.status_code == 200:
                content_type = r.headers.get("content-type", "")
                content_len  = r.headers.get("content-length", "?")
                print(f"    Content-Type: {content_type}, Content-Length: {content_len}")
                with open(OUTPUT, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                size = os.path.getsize(OUTPUT)
                print(f"  OK: descargado ({size} bytes) -> {OUTPUT}")
                downloaded = True
                break
            else:
                print(f"    Fallo HTTP {r.status_code}")
        except Exception as e:
            print(f"    Error: {e}")

# --- Intento 3: ultralytics hub ---
if not downloaded:
    print()
    print("=" * 60)
    print("Intento 3: ultralytics hub")
    print("=" * 60)
    try:
        from ultralytics import YOLO
        for model_id in [
            "arnabdhar/YOLOv8-nano-chess",
            "keremberke/yolov8n-chess-pieces-detection",
        ]:
            print(f"  Probando: {model_id}")
            try:
                model = YOLO(model_id)
                model.save(OUTPUT)
                if os.path.exists(OUTPUT):
                    size = os.path.getsize(OUTPUT)
                    print(f"  OK: {model_id} ({size} bytes) -> {OUTPUT}")
                    downloaded = True
                    break
            except Exception as e:
                print(f"  Error: {e}")
    except ImportError:
        print("  ultralytics no instalado.")

# --- Resultado final ---
print()
print("=" * 60)
if downloaded:
    print(f"RESULTADO: Modelo guardado en '{OUTPUT}'")
else:
    print("RESULTADO: No se pudo descargar el modelo.")
    print()
    print("Alternativas manuales:")
    print("  1. Crear cuenta gratuita en HuggingFace y ejecutar:")
    print("     huggingface-cli login")
    print("     python -c \"from huggingface_hub import hf_hub_download; "
          "hf_hub_download('arnabdhar/YOLOv8-nano-chess', 'best.pt', local_dir='.')\"")
    print("  2. Roboflow Universe (requiere API key gratuita):")
    print("     https://universe.roboflow.com/search?q=chess+pieces+yolov8")
print("=" * 60)
