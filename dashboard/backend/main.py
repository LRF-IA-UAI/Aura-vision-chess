"""
dashboard/backend/main.py — AURA Dashboard Backend
FastAPI + WebSocket + MQTT bridge + MJPEG proxy + static file serving

Ejecutar:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import httpx
import paho.mqtt.client as mqtt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("aura")

app = FastAPI(title="AURA Dashboard")

# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    def __init__(self):
        self._clients: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._clients.append(ws)
        log.info(f"WS connected ({len(self._clients)} total)")

    def disconnect(self, ws: WebSocket):
        self._clients.remove(ws)
        log.info(f"WS disconnected ({len(self._clients)} remaining)")

    async def broadcast(self, data: dict):
        if not self._clients:
            return
        payload = json.dumps(data)
        dead = []
        for ws in self._clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.remove(ws)


manager = ConnectionManager()

# ---------------------------------------------------------------------------
# MQTT client
# ---------------------------------------------------------------------------

_MQTT_BROKER   = "localhost"
_MQTT_PORT     = 1883
_MJPEG_URL     = "http://localhost:8765/"

_mqtt_loop: Optional[asyncio.AbstractEventLoop] = None
_mqtt_client: Optional[mqtt.Client] = None

SUBSCRIBE_TOPICS = [
    "aura/board/state",
    "aura/board/bestmove",
    "aura/system/status",
    "aura/robot/status",
]


def _on_mqtt_message(client, userdata, msg):
    """Called from paho's network thread — schedule broadcast on the main loop."""
    if _mqtt_loop is None:
        return
    try:
        data = json.loads(msg.payload)
        data["_topic"] = msg.topic

        # Normalise topic to a clean type field
        topic_map = {
            "aura/board/state":    "board",
            "aura/board/bestmove": "bestmove",
            "aura/system/status":  "status",
            "aura/robot/status":   "arm",
        }
        data["type"] = topic_map.get(msg.topic, msg.topic)

        asyncio.run_coroutine_threadsafe(
            manager.broadcast(data), _mqtt_loop
        )
    except Exception as e:
        log.warning(f"MQTT message error: {e}")


def _on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        log.info(f"MQTT connected to {_MQTT_BROKER}:{_MQTT_PORT}")
        for topic in SUBSCRIBE_TOPICS:
            client.subscribe(topic)
    else:
        log.warning(f"MQTT connect failed: rc={rc}")


def _start_mqtt():
    global _mqtt_client
    try:
        client = mqtt.Client(client_id="aura_dashboard")
        client.on_connect = _on_mqtt_connect
        client.on_message = _on_mqtt_message
        client.connect_async(_MQTT_BROKER, _MQTT_PORT, keepalive=60)
        client.loop_start()
        _mqtt_client = client
    except Exception as e:
        log.warning(f"MQTT unavailable: {e} — board state won't stream via MQTT")


@app.on_event("startup")
async def startup():
    global _mqtt_loop
    _mqtt_loop = asyncio.get_running_loop()
    _start_mqtt()
    log.info("AURA backend started")


@app.on_event("shutdown")
async def shutdown():
    if _mqtt_client:
        _mqtt_client.loop_stop()
        _mqtt_client.disconnect()


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")

            if msg_type == "cmd":
                # Forward keyboard command to camera pipeline via MQTT
                cmd = msg.get("cmd")
                if cmd in ("calibrate", "analyze", "photo") and _mqtt_client:
                    _mqtt_client.publish(
                        "aura/control/command", json.dumps({"cmd": cmd})
                    )

            elif msg_type == "forcemove":
                # Forward forced move to robot arm via MQTT
                uci = msg.get("uci", "")
                if len(uci) == 4 and _mqtt_client:
                    _mqtt_client.publish(
                        "aura/robot/command", json.dumps({"move": uci})
                    )
                # Also broadcast optimistic board update to all clients
                await manager.broadcast({"type": "forcemove", "uci": uci})

    except WebSocketDisconnect:
        manager.disconnect(ws)


# ---------------------------------------------------------------------------
# MJPEG proxy
# ---------------------------------------------------------------------------

@app.get("/api/stream")
async def mjpeg_stream():
    """Proxies the MJPEG stream from camera_pipeline.py (port 8765)."""
    async def generate():
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", _MJPEG_URL) as response:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        yield chunk
        except Exception as e:
            log.warning(f"MJPEG proxy error: {e}")

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache"},
    )


# ---------------------------------------------------------------------------
# REST status endpoint
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def get_status():
    return {
        "mqtt_connected": _mqtt_client is not None and _mqtt_client.is_connected(),
        "ws_clients": len(manager._clients),
    }


# ---------------------------------------------------------------------------
# Serve React build (production)
# ---------------------------------------------------------------------------

_DIST = Path(__file__).parent.parent / "frontend" / "dist"

if _DIST.exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="static")
else:
    @app.get("/")
    async def root():
        return {
            "message": "AURA backend running. Build frontend: cd dashboard/frontend && npm run build"
        }
