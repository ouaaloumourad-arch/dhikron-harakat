"""Fonctions CTC pures (sans chargement de modèle) : testables sur CPU avec de faux logits."""
import torch
import torchaudio

FRAME_SECONDS = 0.02  # wav2vec2 : une trame toutes les 20 ms

# Graphies du Mushaf uthmani absentes du vocabulaire du modèle
UTHMANI_MAP = {"ٱ": "ا"}  # alif wasla -> alif


def greedy_words(logits, id_to_char: dict, pad_id: int, sep_id):
    """Décodage glouton mot par mot avec la position de chaque mot dans le temps.

    Renvoie une liste de (mot, trame_début, trame_fin).
    """
    ids = torch.argmax(logits, dim=-1).tolist()
    words, cur, start, last, prev = [], [], None, None, None
    for t, i in enumerate(ids):
        if i == prev:
            if i not in (pad_id, sep_id) and cur:
                last = t
            continue
        prev = i
        if i == pad_id:
            continue
        if i == sep_id:
            if cur:
                words.append(("".join(cur), start, last))
                cur, start, last = [], None, None
            continue
        ch = id_to_char.get(i, "")
        if not ch or ch.startswith("<") or ch.startswith("["):
            continue
        if not cur:
            start = t
        cur.append(ch)
        last = t
    if cur:
        words.append(("".join(cur), start, last))
    return words


def word_scores(logits, expected_text: str, vocab: dict, pad_id: int, sep_id):
    """Pour chaque mot attendu, probabilité moyenne (0-1) qu'il ait bien été récité.

    Alignement forcé CTC du verset attendu sur l'audio.
    """
    words, targets, word_of_token = [], [], []
    for raw in expected_text.split():
        chars = [UTHMANI_MAP.get(c, c) for c in raw]
        ids = [vocab[c] for c in chars if c in vocab]
        if not ids:
            continue
        if targets and sep_id is not None:
            targets.append(sep_id)
            word_of_token.append(None)
        targets.extend(ids)
        word_of_token.extend([len(words)] * len(ids))
        words.append(raw)
    if not targets:
        return []

    log_probs = torch.log_softmax(logits.unsqueeze(0), dim=-1).cpu()  # (1, T, C)
    tgt = torch.tensor([targets], dtype=torch.int32)
    aligned, scores = torchaudio.functional.forced_align(log_probs, tgt, blank=pad_id)
    spans = torchaudio.functional.merge_tokens(aligned[0], scores[0].exp(), blank=pad_id)

    per_word = [[] for _ in words]
    for k, span in enumerate(spans):
        w = word_of_token[k] if k < len(word_of_token) else None
        if w is not None:
            per_word[w].append(float(span.score))
    return [
        {"word": words[i], "score": round(sum(s) / len(s), 3) if s else 0.0}
        for i, s in enumerate(per_word)
    ]
