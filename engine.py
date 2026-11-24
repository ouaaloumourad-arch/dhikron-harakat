"""Chargement du modèle coranique, partagé par le worker RunPod et le serveur d'écoute en continu."""
import torch
from transformers import AutoProcessor, AutoModelForCTC

import ctc_utils
from audio_utils import SAMPLE_RATE

MODEL_ID = "rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = DEVICE == "cuda"

print(f"Loading processor and model from {MODEL_ID} on {DEVICE}...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCTC.from_pretrained(MODEL_ID).to(DEVICE)
if USE_FP16:
    model = model.half()  # ~2x plus rapide sur GPU
model.eval()

VOCAB = processor.tokenizer.get_vocab()
ID_TO_CHAR = {i: c for c, i in VOCAB.items()}
PAD_ID = processor.tokenizer.pad_token_id
WORD_SEP = processor.tokenizer.word_delimiter_token or "|"
SEP_ID = VOCAB.get(WORD_SEP)

# Préchauffage : le premier passage sur GPU est lent, on le fait au démarrage
with torch.inference_mode():
    _dummy = torch.zeros(1, SAMPLE_RATE, device=DEVICE)
    model(_dummy.half() if USE_FP16 else _dummy)
print("Model ready!")


def get_logits(samples):
    """samples : float32 mono 16 kHz -> logits (T, C) en float32."""
    inputs = processor(samples, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(DEVICE)
    if USE_FP16:
        inputs = inputs.half()
    with torch.inference_mode():
        return model(inputs).logits.float()[0]


def transcribe(logits) -> str:
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.decode(predicted_ids)


def greedy_words(logits):
    return ctc_utils.greedy_words(logits, ID_TO_CHAR, PAD_ID, SEP_ID)


def word_scores(logits, expected_text: str):
    return ctc_utils.word_scores(logits, expected_text, VOCAB, PAD_ID, SEP_ID)
