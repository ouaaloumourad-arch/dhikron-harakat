# Dhikron — moteur coranique

Modèle : `rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final` (CTC, avec harakât).

## 1. Worker RunPod serverless (bilan de fin de récitation)

Image : `Dockerfile` → `handler.py`.
Entrée : `{"audio": "data:audio/webm;base64,...", "expected_text": "...", "align": true}`
Sortie : `transcription`, `transcription_raw`, `has_harakat`, `status` et, si `align`,
`word_scores` = `[{"word": ..., "score": 0-1}]` (probabilité que chaque mot attendu ait été récité).

Accepte webm/opus (Chrome, Android), mp4/aac (iPhone), wav… (décodage ffmpeg).

Conseil : dans l'endpoint RunPod, **min workers = 1** pour supprimer l'attente au démarrage.

## 2. Serveur d'écoute en continu (pendant la récitation)

Image : `Dockerfile.stream` → `stream_server.py` (WebSocket). À lancer sur un **Pod GPU**
(pas en serverless), port **8000** exposé en HTTP.

Variable d'environnement obligatoire : `STREAM_TOKEN_SECRET` (même valeur que le secret
Supabase du site). Sans elle, le serveur refuse toutes les connexions.

Vérification : `https://<pod-id>-8000.proxy.runpod.net/health` → `{"ok": true, "auth": true}`.

## 3. Mesurer la précision

Voir `evaluate.py` : dossier d'enregistrements + fichiers `.json` avec les fautes attendues,
puis `python3 evaluate.py tests_audio/`.

## Tests (sans GPU)

```
pip install torch torchaudio numpy pytest fastapi httpx
python3 -m pytest -q tests
```
