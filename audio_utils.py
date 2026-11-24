"""Décodage audio robuste pour le worker Dhikron.

Le navigateur envoie du webm/opus (Chrome, Android) ou du mp4/aac (iPhone),
pas du wav. On passe donc toujours par ffmpeg pour obtenir un signal
mono 16 kHz en float32, quel que soit le format reçu.
"""
import base64
import subprocess

import numpy as np

SAMPLE_RATE = 16000
MIN_SECONDS = 0.3  # en dessous : rien d'exploitable, on ne lance pas le modèle


def b64_to_bytes(audio_b64: str) -> bytes:
    # Retire un éventuel préfixe « data:audio/...;base64, »
    if "," in audio_b64[:100]:
        audio_b64 = audio_b64.split(",", 1)[1]
    return base64.b64decode(audio_b64)


def decode_to_16k_mono(audio_bytes: bytes) -> np.ndarray:
    """Convertit n'importe quel format audio en float32 mono 16 kHz."""
    proc = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",
            "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-f", "f32le", "pipe:1",
        ],
        input=audio_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        raise ValueError(proc.stderr.decode("utf-8", "ignore")[-300:] or "ffmpeg decode failed")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def is_too_short(samples: np.ndarray) -> bool:
    return samples.size < int(MIN_SECONDS * SAMPLE_RATE)


def is_silent(samples: np.ndarray, threshold: float = 0.005) -> bool:
    if samples.size == 0:
        return True
    return float(np.sqrt(np.mean(np.square(samples)))) < threshold
