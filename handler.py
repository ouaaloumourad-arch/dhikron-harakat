import torch
import torchaudio
import runpod
from transformers import AutoProcessor, AutoModelForCTC

from audio_utils import b64_to_bytes, decode_to_16k_mono, is_silent, is_too_short, SAMPLE_RATE

# Model reference
MODEL_ID = "rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = DEVICE == "cuda"

print(f"Loading processor and model from {MODEL_ID} on {DEVICE}...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCTC.from_pretrained(MODEL_ID).to(DEVICE)
if USE_FP16:
    model = model.half()  # ~2x plus rapide sur GPU, même résultat en pratique
model.eval()

VOCAB = processor.tokenizer.get_vocab()
PAD_ID = processor.tokenizer.pad_token_id
WORD_SEP = processor.tokenizer.word_delimiter_token or "|"
WORD_SEP_ID = VOCAB.get(WORD_SEP)

# Préchauffage : le premier passage sur GPU est lent, on le fait au démarrage
with torch.inference_mode():
    _dummy = torch.zeros(1, SAMPLE_RATE, device=DEVICE)
    model(_dummy.half() if USE_FP16 else _dummy)
print("Model ready!")

# Harakat unicode characters
FATHA = 'َ'
DAMMA = 'ُ'
KASRA = 'ِ'
SUKUN = 'ْ'
SHADDA = 'ّ'
FATHATAN = 'ً'
DAMMATAN = 'ٌ'
KASRATAN = 'ٍ'
HARAKAT_SET = {FATHA, DAMMA, KASRA, SUKUN, SHADDA, FATHATAN, DAMMATAN, KASRATAN}

# Graphies du Mushaf uthmani absentes du vocabulaire du modèle
UTHMANI_MAP = {"ٱ": "ا"}  # alif wasla -> alif


def strip_harakat(text: str) -> str:
    return "".join(c for c in text if c not in HARAKAT_SET)


def get_logits(samples):
    inputs = processor(samples, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(DEVICE)
    if USE_FP16:
        inputs = inputs.half()
    with torch.inference_mode():
        return model(inputs).logits.float()


def transcribe(logits) -> str:
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]


def word_scores(logits, expected_text: str):
    """Pour chaque mot attendu, probabilité moyenne (0-1) qu'il ait bien été récité.

    Alignement forcé CTC du verset attendu sur l'audio. Ajouté à la réponse
    seulement si l'appel le demande (input.align = true) : ne change rien
    pour les clients actuels.
    """
    words, targets, word_of_token = [], [], []
    for raw in expected_text.split():
        chars = [UTHMANI_MAP.get(c, c) for c in raw]
        ids = [VOCAB[c] for c in chars if c in VOCAB]
        if not ids:
            continue
        if targets and WORD_SEP_ID is not None:
            targets.append(WORD_SEP_ID)
            word_of_token.append(None)
        targets.extend(ids)
        word_of_token.extend([len(words)] * len(ids))
        words.append(raw)
    if not targets:
        return []

    log_probs = torch.log_softmax(logits.unsqueeze(0), dim=-1).cpu()  # (1, T, C)
    tgt = torch.tensor([targets], dtype=torch.int32)
    aligned, scores = torchaudio.functional.forced_align(log_probs, tgt, blank=PAD_ID)
    spans = torchaudio.functional.merge_tokens(aligned[0], scores[0].exp(), blank=PAD_ID)

    per_word = [[] for _ in words]
    for k, span in enumerate(spans):
        w = word_of_token[k] if k < len(word_of_token) else None
        if w is not None:
            per_word[w].append(float(span.score))
    return [
        {"word": words[i], "score": round(sum(s) / len(s), 3) if s else 0.0}
        for i, s in enumerate(per_word)
    ]


def empty_result(expected_text: str, reason: str):
    return {
        "transcription": "",
        "transcription_raw": "",
        "expected_text": expected_text,
        "has_harakat": False,
        "status": reason,
    }


def handler(job):
    job_input = job.get("input", {})
    audio_b64 = job_input.get("audio_base64") or job_input.get("audio")
    expected_text = job_input.get("expected_text", "") or ""

    if not audio_b64:
        return {"error": "Missing audio_base64 or audio in input"}

    try:
        audio_bytes = b64_to_bytes(audio_b64)
    except Exception as e:
        return {"error": f"Base64 decode error: {str(e)}"}

    try:
        samples = decode_to_16k_mono(audio_bytes)
    except Exception as e:
        return {"error": f"Audio decode error: {str(e)[:300]}"}

    # Morceau vide ou silencieux : réponse immédiate, sans passer par le modèle
    if is_too_short(samples):
        return empty_result(expected_text, "too_short")
    if is_silent(samples):
        return empty_result(expected_text, "silence")

    try:
        logits = get_logits(samples)
        predicted_text = transcribe(logits)
        result = {
            "transcription": predicted_text,
            "transcription_raw": strip_harakat(predicted_text),
            "expected_text": expected_text,
            "has_harakat": any(c in HARAKAT_SET for c in predicted_text),
            "status": "success",
        }
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}

    if job_input.get("align") and expected_text:
        try:
            result["word_scores"] = word_scores(logits[0], expected_text)
        except Exception as e:  # l'alignement ne doit jamais casser la transcription
            result["word_scores_error"] = str(e)[:300]

    return result


runpod.serverless.start({"handler": handler})
