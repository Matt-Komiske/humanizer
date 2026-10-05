"""Meaning-preservation check: cosine similarity of sentence embeddings (all-MiniLM-L6-v2)."""
import re
import threading

_model = None
_lock = threading.Lock()


def _get():
    global _model
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            from .detectors import _snapshot, device
            _model = SentenceTransformer(_snapshot("sentence-transformers/all-MiniLM-L6-v2"), device=device())
        return _model


def similarity(a, b):
    m = _get()
    with _lock:
        ea, eb = m.encode([a, b], normalize_embeddings=True)
    return float((ea * eb).sum())


_NUM_WORDS = r"(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|hundred|thousand|million|billion|percent|dozen)"
_NUM = re.compile(r"\b(?:\d[\d,.]*%?|" + _NUM_WORDS + r")\b", re.IGNORECASE)
_PROPER = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b")


def new_facts(source, candidate):
    """Numbers or proper names in the candidate that aren't in the source: a sign the model
    invented a fact. Embedding similarity alone doesn't catch these."""
    src = source.lower()
    added = {m.group(0).lower() for m in _NUM.finditer(candidate)} - {m.group(0).lower() for m in _NUM.finditer(source)}
    for m in _PROPER.finditer(candidate):
        w = m.group(1)
        if w.lower() not in src and w not in ("I", "I'm", "I've"):
            added.add(w)
    return added


def faithful(source, candidate, min_sim):
    return similarity(source, candidate) >= min_sim and not new_facts(source, candidate)
