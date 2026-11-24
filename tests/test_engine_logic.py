"""Tests sans GPU ni vrai modèle : faux vocabulaire et faux logits."""
import base64
import hashlib
import hmac
import json
import os
import sys
import time
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ctc_utils  # noqa: E402
from stream_session import SAMPLES_PER_FRAME, StreamSession, verify_token  # noqa: E402

LETTERS = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
VOCAB = {"<pad>": 0, "|": 1, **{c: i + 2 for i, c in enumerate(LETTERS)}}
ID_TO_CHAR = {i: c for c, i in VOCAB.items()}


def fake_logits(text, frames_per_tok=4, trailing_blank=0):
    seq = []
    for i, w in enumerate(text.split()):
        if i:
            seq.append(1)
        seq += [VOCAB[c] for c in w if c in VOCAB]
    T = len(seq) * frames_per_tok * 2 + trailing_blank
    L = torch.full((T, len(VOCAB)), -8.0)
    t = 0
    for tok in seq:
        for _ in range(frames_per_tok):
            L[t, tok] = 8.0
            t += 1
        for _ in range(frames_per_tok):
            L[t, 0] = 8.0
            t += 1
    for _ in range(trailing_blank):
        L[t, 0] = 8.0
        t += 1
    return L


def test_greedy_words_and_spans():
    words = ctc_utils.greedy_words(fake_logits("الحمد لله رب"), ID_TO_CHAR, 0, 1)
    assert [w[0] for w in words] == ["الحمد", "لله", "رب"]
    assert all(w[1] <= w[2] for w in words)
    assert words[0][2] < words[1][1] < words[2][1]


def test_word_scores_detects_replaced_word():
    expected = "ٱلحمد لله رب العالمين"
    ok = ctc_utils.word_scores(fake_logits("الحمد لله رب العالمين"), expected, VOCAB, 0, 1)
    bad = ctc_utils.word_scores(fake_logits("الحمد لله ملك العالمين"), expected, VOCAB, 0, 1)
    assert [s["score"] for s in ok] == [1.0, 1.0, 1.0, 1.0]
    assert bad[2]["score"] < 0.5 and bad[0]["score"] == 1.0


def _session_with_audio(seconds):
    s = StreamSession()
    pcm = (np.sin(np.arange(int(seconds * 16000)) / 5) * 8000).astype("<i2").tobytes()
    s.feed(pcm)
    return s


def test_stream_commits_only_stable_words():
    s = _session_with_audio(3.0)
    n_frames = s.buffer.size // SAMPLES_PER_FRAME  # 150 trames
    # 3 mots : 2 finis tôt, le dernier finit à la toute fin (encore en cours)
    words = [("الحمد", 5, 40), ("لله", 45, 80), ("رب", 120, n_frames - 2)]
    out = s.step(lambda _s: words)
    assert out["committed"] == "الحمد لله"
    assert out["pending"] == "رب"
    # L'audio déjà confirmé a été retiré du tampon
    assert s.buffer.size == 3.0 * 16000 - 81 * SAMPLES_PER_FRAME


def test_stream_final_commits_everything():
    s = _session_with_audio(1.0)
    out = s.step(lambda _s: [("رب", 10, 45)], final=True)
    assert out == {"type": "final", "committed": "رب", "pending": ""}


def test_stream_skips_silence_without_model():
    s = StreamSession()
    s.feed(np.zeros(16000, dtype="<i2").tobytes())
    called = []
    assert s.step(lambda _s: called.append(1) or []) is None
    assert not called


def _token(uid, exp, secret):
    p = base64.urlsafe_b64encode(json.dumps({"uid": uid, "exp": exp}).encode()).decode().rstrip("=")
    return p + "." + hmac.new(secret.encode(), p.encode(), hashlib.sha256).hexdigest()


def test_verify_token():
    now = time.time()
    assert verify_token(_token("u1", now + 60, "s3cret"), "s3cret") == "u1"
    assert verify_token(_token("u1", now - 1, "s3cret"), "s3cret") is None  # expiré
    assert verify_token(_token("u1", now + 60, "autre"), "s3cret") is None  # mauvaise signature
    assert verify_token("n'importe quoi", "s3cret") is None
    assert verify_token(_token("u1", now + 60, "s3cret"), "") is None  # serveur sans secret : refus


def test_websocket_end_to_end_with_fake_model(monkeypatch):
    fake = types.ModuleType("engine")
    fake.DEVICE = "cpu"
    fake.get_logits = lambda samples: samples
    fake.greedy_words = lambda samples: [("الحمد", 2, 20)] if samples.size > 16000 else []
    sys.modules["engine"] = fake
    monkeypatch.setenv("STREAM_TOKEN_SECRET", "s3cret")
    import importlib
    import stream_server
    importlib.reload(stream_server)
    from fastapi.testclient import TestClient

    client = TestClient(stream_server.app)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "start", "token": "faux"}))
        assert ws.receive_json()["error"] == "unauthorized"

    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "start", "token": _token("u1", time.time() + 60, "s3cret")}))
        pcm = (np.sin(np.arange(1600) / 5) * 8000).astype("<i2").tobytes()  # 100 ms
        for _ in range(20):
            ws.send_bytes(pcm)
        time.sleep(0.6)
        ws.send_text(json.dumps({"type": "stop"}))
        msgs = []
        while True:
            m = ws.receive_json()
            msgs.append(m)
            if m["type"] == "final":
                break
        assert msgs[-1]["committed"].startswith("الحمد")
