"""Serveur d'écoute en continu (WebSocket) pour Dhikron.

À lancer sur un Pod GPU RunPod (pas en serverless : il faut une connexion ouverte) :
    STREAM_TOKEN_SECRET=... uvicorn stream_server:app --host 0.0.0.0 --port 8000

Protocole :
  1. Le site ouvre wss://<pod>/ws et envoie {"type":"start","token":"..."}
  2. Puis du son PCM 16 bits, mono, 16 kHz, en paquets binaires (~100 ms)
  3. Le serveur répond {"type":"text","committed":"...","pending":"..."} au fil de l'eau
  4. Le site envoie {"type":"stop"} ; le serveur répond {"type":"final",...} et ferme
"""
import asyncio
import json
import os
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

import engine
from stream_session import MAX_SESSION_SECONDS, StreamSession, verify_token

SECRET = os.environ.get("STREAM_TOKEN_SECRET", "")
STEP_SECONDS = 0.4
MAX_BYTES_PER_MESSAGE = 64 * 1024

app = FastAPI()
gpu_lock = asyncio.Lock()  # un seul calcul GPU à la fois, les sessions se relaient


def infer_words(samples):
    return engine.greedy_words(engine.get_logits(samples))


async def run_step(session: StreamSession, final: bool = False):
    async with gpu_lock:
        return await asyncio.to_thread(session.step, infer_words, final)


@app.get("/health")
def health():
    return {"ok": True, "device": engine.DEVICE, "auth": bool(SECRET)}


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        first = await asyncio.wait_for(websocket.receive_text(), timeout=10)
        msg = json.loads(first)
    except Exception:
        await websocket.close(code=4400)
        return
    if msg.get("type") != "start" or not verify_token(msg.get("token", ""), SECRET):
        await websocket.send_json({"type": "error", "error": "unauthorized"})
        await websocket.close(code=4401)
        return

    session = StreamSession()
    stopped = asyncio.Event()
    started = time.time()

    async def receiver():
        try:
            while True:
                data = await websocket.receive()
                if data.get("type") == "websocket.disconnect":
                    break
                if data.get("bytes") is not None:
                    if len(data["bytes"]) <= MAX_BYTES_PER_MESSAGE:
                        session.feed(data["bytes"])
                elif data.get("text"):
                    try:
                        if json.loads(data["text"]).get("type") == "stop":
                            break
                    except Exception:
                        pass
                if session.seconds > MAX_SESSION_SECONDS or time.time() - started > MAX_SESSION_SECONDS + 60:
                    break
        except WebSocketDisconnect:
            pass
        finally:
            stopped.set()

    recv_task = asyncio.create_task(receiver())
    try:
        while not stopped.is_set():
            await asyncio.sleep(STEP_SECONDS)
            if session.ready():
                out = await run_step(session)
                if out:
                    await websocket.send_json(out)
        out = await run_step(session, final=True)
        await websocket.send_json(out or {"type": "final", "committed": " ".join(session.committed), "pending": ""})
        await websocket.close()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        recv_task.cancel()
