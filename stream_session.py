"""Suivi d'une récitation en continu : logique pure, sans réseau ni modèle (testable).

Le son arrive par petits paquets. Toutes les ~0,4 s on relance le modèle sur
l'audio depuis le dernier mot confirmé (« ancre »). Un mot est confirmé quand
il se termine au moins STABLE_SECONDS avant la fin du son reçu : à ce stade
le modèle ne changera plus d'avis. Le reste est envoyé comme « provisoire ».
"""
import base64
import hashlib
import hmac
import json
import time

import numpy as np

from audio_utils import SAMPLE_RATE
from ctc_utils import FRAME_SECONDS

SAMPLES_PER_FRAME = int(FRAME_SECONDS * SAMPLE_RATE)  # 320
STABLE_SECONDS = 0.6      # marge avant de figer un mot
MIN_NEW_SECONDS = 0.25    # pas de calcul si trop peu de son nouveau
MAX_WINDOW_SECONDS = 12   # au-delà, on fige de force pour garder le calcul rapide
MAX_SESSION_SECONDS = 20 * 60
SILENCE_RMS = 0.005


def pcm16_to_float(data: bytes) -> np.ndarray:
    if len(data) % 2:
        data = data[:-1]
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


class StreamSession:
    def __init__(self):
        self.buffer = np.zeros(0, dtype=np.float32)  # son depuis l'ancre
        self.total_samples = 0
        self.samples_at_last_step = 0
        self.committed: list[str] = []
        self.pending: list[str] = []

    def feed(self, pcm: bytes) -> None:
        samples = pcm16_to_float(pcm)
        self.buffer = np.concatenate([self.buffer, samples])
        self.total_samples += samples.size

    @property
    def seconds(self) -> float:
        return self.total_samples / SAMPLE_RATE

    def ready(self) -> bool:
        return (self.total_samples - self.samples_at_last_step) >= MIN_NEW_SECONDS * SAMPLE_RATE

    def step(self, infer_words, final: bool = False):
        """infer_words(samples) -> [(mot, trame_début, trame_fin)]. Renvoie le message à envoyer ou None."""
        self.samples_at_last_step = self.total_samples
        if self.buffer.size < 0.2 * SAMPLE_RATE:
            return None
        if not final and float(np.sqrt(np.mean(np.square(self.buffer[-SAMPLE_RATE:])))) < SILENCE_RMS and not self.pending:
            # Silence sans mot en attente : on jette le silence (garde 0,5 s) sans lancer le modèle
            self.buffer = self.buffer[-SAMPLE_RATE // 2:]
            return None

        words = infer_words(self.buffer)
        n_frames = self.buffer.size // SAMPLES_PER_FRAME
        cutoff = n_frames - int(STABLE_SECONDS / FRAME_SECONDS)

        if final:
            stable = words
        else:
            stable = [w for w in words if w[2] < cutoff]
            # Fenêtre trop longue : on fige tout sauf le dernier mot
            if not stable and self.buffer.size > MAX_WINDOW_SECONDS * SAMPLE_RATE and len(words) > 1:
                stable = words[:-1]
        # On ne fige que le début : un mot stable ne peut pas suivre un mot instable
        k = 0
        while k < len(stable) and k < len(words) and stable[k] is words[k]:
            k += 1
        stable = words[:k]

        if stable:
            self.committed.extend(w[0] for w in stable)
            cut = (stable[-1][2] + 1) * SAMPLES_PER_FRAME
            self.buffer = self.buffer[cut:]
        self.pending = [w[0] for w in words[k:]] if not final else []
        # Garde-fou mémoire : jamais plus que la fenêtre maximale
        max_samples = int(MAX_WINDOW_SECONDS * 1.5 * SAMPLE_RATE)
        if self.buffer.size > max_samples:
            self.buffer = self.buffer[-max_samples:]

        return {
            "type": "final" if final else "text",
            "committed": " ".join(self.committed),
            "pending": " ".join(self.pending),
        }


# --- Jeton d'accès signé (émis par la fonction Supabase « stream-token ») ---

def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def verify_token(token: str, secret: str, now: float | None = None):
    """Jeton « payload.signature » : payload = base64url(JSON {uid, exp}), signature = HMAC-SHA256 hex.

    Renvoie l'identifiant utilisateur si le jeton est valide et non expiré, sinon None.
    """
    if not token or not secret or token.count(".") != 1:
        return None
    payload_b64, sig = token.split(".")
    expected = hmac.new(secret.encode(), payload_b64.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return None
    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("uid"), str):
        return None
    if float(payload.get("exp", 0)) < (now if now is not None else time.time()):
        return None
    return payload["uid"]
