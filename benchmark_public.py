"""Mesure automatique sur des récitations existantes (récitateurs EveryAyah), sans rien enregistrer.

Pour chaque verset et chaque récitateur, 3 tests :
  1. Verset juste          -> aucun mot ne devrait être signalé (mesure des FAUSSES ALERTES)
  2. Mot remplacé          -> le texte attendu contient un mot d'un autre verset : comme si le
                              récitant avait dit un autre mot (mesure des FAUTES DÉTECTÉES)
  3. Mot sauté             -> le texte attendu contient un mot en plus que le récitant n'a pas dit

Lancer sur le Pod GPU (accès Internet nécessaire) :
    python3 benchmark_public.py            # rapide : ~60 versets x 4 récitateurs
    python3 benchmark_public.py --full     # plus de sourates
"""
import json
import random
import sys
import time
import urllib.request

import engine
from audio_utils import decode_to_16k_mono

RECITERS = [
    "Alafasy_128kbps",
    "Husary_128kbps",
    "Minshawi_Murattal_128kbps",
    "Abdul_Basit_Murattal_192kbps",
]
QUICK = {1: None, 112: None, 113: None, 114: None, 36: range(1, 21), 2: range(1, 21), 67: range(1, 16)}
FULL = {**QUICK, 18: range(1, 31), 55: range(1, 41), 78: None, 87: None, 93: None, 103: None}
THRESHOLDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]


def fetch(url, binary=True, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "dhikron-benchmark"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
                return data if binary else json.loads(data)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(1 + k)


def surah_text(n):
    data = fetch(f"https://api.alquran.cloud/v1/surah/{n}/quran-uthmani", binary=False)
    ayahs = [a["text"] for a in data["data"]["ayahs"]]
    if n not in (1, 9) and ayahs and ayahs[0].startswith("بِسْمِ"):
        # alquran.cloud colle la basmala au 1er verset ; EveryAyah ne la récite pas dans ce fichier
        ayahs[0] = " ".join(ayahs[0].split()[4:])
    return ayahs


def scored(words):
    return [w for w in words if any(c in engine.VOCAB for c in w)]


def main(full=False):
    random.seed(7)
    plan = FULL if full else QUICK
    texts = {s: surah_text(s) for s in plan}
    all_words = [w for t in texts.values() for a in t for w in scored(a.split()) if len(w) > 3]

    correct_scores, error_scores, latencies = [], [], []
    for reciter in RECITERS:
        for s, ayahs in texts.items():
            rng = plan[s] or range(1, len(ayahs) + 1)
            for a in rng:
                if a > len(ayahs):
                    continue
                text = ayahs[a - 1]
                words = text.split()
                if len(scored(words)) < 2:
                    continue
                try:
                    audio = fetch(f"https://everyayah.com/data/{reciter}/{s:03d}{a:03d}.mp3")
                except Exception as e:
                    print(f"  {reciter} {s}:{a} indisponible ({e})")
                    continue
                t0 = time.time()
                logits = engine.get_logits(decode_to_16k_mono(audio))
                latencies.append(time.time() - t0)

                # 1. verset juste
                correct_scores += [x["score"] for x in engine.word_scores(logits, text)]

                # 2. mot remplacé
                i = random.randrange(len(words))
                swapped = list(words)
                swapped[i] = random.choice([w for w in all_words if w != words[i]])
                sc = engine.word_scores(logits, " ".join(swapped))
                target = len(scored(swapped[: i + 1])) - 1
                if 0 <= target < len(sc):
                    error_scores.append(sc[target]["score"])

                # 3. mot sauté (mot en trop dans le texte attendu)
                j = random.randrange(len(words) + 1)
                extra = random.choice(all_words)
                inserted = words[:j] + [extra] + words[j:]
                sc = engine.word_scores(logits, " ".join(inserted))
                target = len(scored(inserted[: j + 1])) - 1
                if 0 <= target < len(sc):
                    error_scores.append(sc[target]["score"])
        print(f"{reciter} : fait ({len(latencies)} versets au total)")

    print(f"\n{len(correct_scores)} mots justes, {len(error_scores)} fautes simulées")
    print("seuil | fautes détectées | fausses alertes")
    for th in THRESHOLDS:
        det = 100 * sum(s < th for s in error_scores) / max(1, len(error_scores))
        far = 100 * sum(s < th for s in correct_scores) / max(1, len(correct_scores))
        print(f" {th:.2f} |     {det:5.1f} %     |    {far:5.1f} %")
    lat = sorted(latencies)
    print(f"\nTemps de calcul par verset : médiane {lat[len(lat) // 2]:.3f}s, max {lat[-1]:.3f}s")
    print("\nÀ retenir : choisis le seuil avec le plus de fautes détectées pour moins de ~2 % de fausses alertes.")
    print("Attention : récitateurs professionnels -> les fausses alertes seront plus élevées sur des voix ordinaires.")


if __name__ == "__main__":
    main(full="--full" in sys.argv)
