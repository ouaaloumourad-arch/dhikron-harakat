import base64
import io
import os
import re
import tempfile
import torch
import torchaudio
import runpod
from transformers import AutoProcessor, AutoModelForCTC

# Model reference
MODEL_ID = "rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading processor and model from {MODEL_ID} on {DEVICE}...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCTC.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()
print("Model ready!")

# Harakat unicode characters
FATHA = '\u064e'
DAMMA = '\u064f'
KASRA = '\u0650'
SUKUN = '\u0652'
SHADDA = '\u0651'
FATHATAN = '\u064b'
DAMMATAN = '\u064c'
KASRATAN = '\u064d'
HARAKAT_SET = {FATHA, DAMMA, KASRA, SUKUN, SHADDA, FATHATAN, DAMMATAN, KASRATAN}

def strip_harakat(text: str) -> str:
    return "".join(c for c in text if c not in HARAKAT_SET)

def transcribe_audio(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        waveform, sample_rate = torchaudio.load(tmp.name)

    # Convert to mono if multi-channel
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to 16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    input_values = processor(
        waveform.squeeze(0).numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    ).input_values.to(DEVICE)

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

def handler(job):
    job_input = job.get("input", {})
    audio_b64 = job_input.get("audio_base64") or job_input.get("audio")

    if not audio_b64:
        return {"error": "Missing audio_base64 or audio in input"}

    # Handle data:audio/...;base64, prefix
    if "," in audio_b64:
        audio_b64 = audio_b64.split(",", 1)[1]

    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception as e:
        return {"error": f"Base64 decode error: {str(e)}"}

    try:
        predicted_text = transcribe_audio(audio_bytes)
        unvocalized_text = strip_harakat(predicted_text)

        expected_text = job_input.get("expected_text", "")

        return {
            "transcription": predicted_text,
            "transcription_raw": unvocalized_text,
            "expected_text": expected_text,
            "has_harakat": any(c in HARAKAT_SET for c in predicted_text)
        }
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}

runpod.serverless.start({"handler": handler})
