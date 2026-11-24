"""Mesure la précision du moteur sur tes propres récitations.

Prépare un dossier, par exemple tests_audio/, avec pour chaque récitation :
  fatiha_ahmed.webm   (ou .wav, .m4a, .mp3...)
  fatiha_ahmed.json   {"expected_text": "بِسْمِ ٱللَّهِ ...", "wrong_words": [3, 7]}

« wrong_words » = positions (à partir de 0) des mots du verset volontairement
ratés ou sautés dans l'enregistrement ; [] pour une récitation sans faute.

Lance ensuite (sur une machine avec GPU, par exemple ton Pod RunPod) :
    python3 evaluate.py tests_audio/

Tu obtiens, pour plusieurs seuils :
  - fautes détectées : % des vraies fautes repérées (plus c'est haut, mieux c'est)
  - fausses alertes  : % des mots justes signalés à tort (plus c'est bas, mieux c'est)
"""
import json
import sys
import time
from pathlib import Path

import engine
from audio_utils import decode_to_16k_mono

AUDIO_EXT = {".wav", ".webm", ".m4a", ".mp4", ".mp3", ".ogg", ".opus", ".aac"}
THRESHOLDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]


def main(folder: str):
    items = []
    for audio in sorted(Path(folder).iterdir()):
        if audio.suffix.lower() not in AUDIO_EXT:
            continue
        label = audio.with_suffix(".json")
        if not label.exists():
            print(f"(ignoré, pas de {label.name})")
            continue
        meta = json.loads(label.read_text(encoding="utf-8"))
        t0 = time.time()
        logits = engine.get_logits(decode_to_16k_mono(audio.read_bytes()))
        scores = engine.word_scores(logits, meta["expected_text"])
        elapsed = time.time() - t0
        wrong = set(meta.get("wrong_words", []))
        # Les mots sans score (signes de pause seuls) sont ignorés
        words = [w for w in meta["expected_text"].split() if any(c in engine.VOCAB for c in w)]
        if len(words) != len(scores):
            print(f"(attention {audio.name} : {len(words)} mots, {len(scores)} scores)")
        items.append((audio.name, scores, wrong, elapsed))
        low = [i for i, s in enumerate(scores) if s["score"] < 0.75]
        print(f"{audio.name}: {len(scores)} mots, {elapsed:.2f}s, mots douteux (<0,75) : {low}, fautes attendues : {sorted(wrong)}")

    if not items:
        print("Aucune récitation trouvée.")
        return

    print("\nseuil | fautes détectées | fausses alertes")
    for th in THRESHOLDS:
        tp = fn = fp = tn = 0
        for _, scores, wrong, _ in items:
            for i, s in enumerate(scores):
                flagged = s["score"] < th
                if i in wrong:
                    tp += flagged
                    fn += not flagged
                else:
                    fp += flagged
                    tn += not flagged
        rec = 100 * tp / (tp + fn) if tp + fn else float("nan")
        far = 100 * fp / (fp + tn) if fp + tn else float("nan")
        print(f" {th:.2f} | {rec:6.1f} % ({tp}/{tp + fn}) | {far:6.1f} % ({fp}/{fp + tn})")
    avg = sum(i[3] for i in items) / len(items)
    print(f"\nTemps moyen par récitation : {avg:.2f}s ({len(items)} récitations)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests_audio")
