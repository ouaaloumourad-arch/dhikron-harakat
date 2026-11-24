import runpod

import engine
from audio_utils import b64_to_bytes, decode_to_16k_mono, is_silent, is_too_short

# Harakat unicode characters
HARAKAT_SET = {'َ', 'ُ', 'ِ', 'ْ', 'ّ', 'ً', 'ٌ', 'ٍ'}


def strip_harakat(text: str) -> str:
    return "".join(c for c in text if c not in HARAKAT_SET)


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
        logits = engine.get_logits(samples)
        predicted_text = engine.transcribe(logits)
        result = {
            "transcription": predicted_text,
            "transcription_raw": strip_harakat(predicted_text),
            "expected_text": expected_text,
            "has_harakat": any(c in HARAKAT_SET for c in predicted_text),
            "status": "success",
        }
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}

    # Score par mot sur le verset attendu, seulement si demandé (input.align = true)
    if job_input.get("align") and expected_text:
        try:
            result["word_scores"] = engine.word_scores(logits, expected_text)
        except Exception as e:  # l'alignement ne doit jamais casser la transcription
            result["word_scores_error"] = str(e)[:300]

    return result


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
