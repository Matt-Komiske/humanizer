"""Humanizing pipeline: LLM rewrite -> rule cleanup -> detector-guided candidate selection.

Approach (from DIPPER / Adversarial Paraphrasing research): rewrite each paragraph several
times, keep the candidate that local detectors score as most human while its meaning stays
close to the original, and retry paragraphs that still score as AI."""
import re
import time

from . import detectors, meaning, rewriter, rules, structure

MODES = {
    # candidates per round, max rounds
    "quick": (0, 0),        # rules only, no AI model needed
    "balanced": (3, 2),
    "thorough": (6, 3),
    "max": (9, 4),          # slow; also steers on the held-out detector (nothing is held out)
}
GUIDE_DETECTORS = ["superannotate", "binoculars"]   # steer the rewrite
JUDGE_DETECTORS = ["desklib"]                 # held out: never used for steering
TARGET = 0.35          # stop retrying a paragraph once its guide score is below this
MIN_SIM = 0.82         # minimum meaning similarity for a candidate to be accepted
MIN_WORDS_TO_SCORE = 25

BOLDER = ("The previous attempt still read as machine-written. Be bolder: restructure more "
          "sentences, vary rhythm more sharply, and choose less predictable (but accurate) wording. "
          "Keep every fact.")


def split_chunks(para, max_words=170):
    """Split an over-long paragraph into sentence-aligned chunks."""
    if len(para.split()) <= max_words:
        return [para]
    chunks, cur = [], []
    for s in re.split(r"(?<=[.!?])\s+", para):
        if cur and len(" ".join(cur + [s]).split()) > max_words:
            chunks.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def guide_score(text, names):
    if not names or len(text.split()) < MIN_WORDS_TO_SCORE:
        return 0.0
    s = detectors.score_all(text, names)
    return sum(s.values()) / len(s)


def humanize(text, mode="balanced", tone="standard", model=None, base_model=None, restructure=True,
             progress=lambda *a: None, cancel=lambda: False):
    t0 = time.time()
    n_cand, rounds = MODES.get(mode, MODES["balanced"])
    use_ai = n_cand > 0 and rewriter.available(model or rewriter.DEFAULT_MODEL)
    if n_cand > 0 and not use_ai:
        progress("AI model not available. Falling back to rule-based mode.", 0)
    steer = GUIDE_DETECTORS + (JUDGE_DETECTORS if mode == "max" else [])
    guide = [d for d in steer if d in detectors.available()] if use_ai else []
    base_model = base_model or rewriter.BASE_MODEL
    use_base = use_ai and rewriter.available(base_model)

    paras = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    notes = []
    if use_ai and restructure and len(paras) >= 3:
        paras, notes = structure.restructure(paras, tone=tone, model=model,
                                             progress=lambda m, f: progress(m, 0.15 * f))
    # One state per chunk. Paragraphs too short to rewrite just get the rule pass.
    states = []
    for i, para in enumerate(paras):
        if not use_ai or len(para.split()) < 6:
            states.append({"para": i, "orig": para, "best": rules.humanize(para), "active": False})
            continue
        context = "\n\n".join(paras[max(0, i - 1):i + 2]) if len(paras) > 1 else ""
        for chunk in split_chunks(para):
            best = rules.humanize(chunk)
            # small handicap so an equally-scored real rewrite beats the lightly edited original
            score = guide_score(best, guide) + 0.05 if guide else None
            states.append({"para": i, "orig": chunk, "best": best, "score": score, "current": chunk,
                           "context": context, "active": True, "cands": [], "pool": [(score, best)]})

    # Work round by round across ALL chunks, grouped by model, so Ollama loads each model
    # once per round instead of swapping between them for every candidate.
    n_base = (n_cand * 2 + 2) // 3 if use_base else 0
    n_chat = n_cand - n_base
    for r in range(rounds):
        active = [s for s in states if s["active"]]
        if not active:
            break
        jobs = [(s, "base", c) for c in range(n_base) for s in active] + \
               [(s, "chat", c) for c in range(n_chat) for s in active]
        for k, (s, kind, c) in enumerate(jobs):
            if cancel():
                raise RuntimeError("cancelled")
            progress(f"Round {r + 1} of {rounds}: rewrite {k + 1} of {len(jobs)}"
                     f" ({'base' if kind == 'base' else 'chat'} model)", 0.15 + 0.7 * (r + k / len(jobs)) / rounds)
            temp = 0.9 + 0.1 * (c % 3) + 0.1 * r
            try:
                if kind == "base":   # base model rewrites the original
                    raw = rewriter.rewrite_base(s["orig"], tone=tone, model=base_model, temperature=temp)
                else:                # chat model refines the best version so far
                    raw = rewriter.rewrite(s["current"], tone=tone, context=s["context"], model=model,
                                           temperature=temp, extra=BOLDER if r else "")
            except Exception as e:
                progress(f"Rewrite failed: {e}", r / rounds)
                continue
            cand = rules.humanize(raw)
            sim = meaning.similarity(s["orig"], cand)
            if sim >= MIN_SIM and not meaning.new_facts(s["orig"], cand):
                s["cands"].append((guide_score(cand, guide), -sim, cand))
        for s in active:
            s["pool"] += [(sc, c) for sc, _, c in s["cands"]]
            if s["cands"]:
                score, _, cand = min(s["cands"])
                if s["score"] is None or score < s["score"]:
                    s["best"], s["score"] = cand, score
            s["cands"] = []
            s["current"] = s["best"]
            if s["score"] is not None and s["score"] < TARGET:
                s["active"] = False

    # Whole-essay selection: a paragraph that scores human alone can still read as AI next to
    # the others, so choose each paragraph's version by how the ASSEMBLED essay scores.
    def assemble():
        return structure.final_cleanup([" ".join(s["best"] for s in states if s["para"] == i)
                                        for i in range(len(paras))])
    if guide and any(len(s.get("pool", [])) > 1 for s in states):
        doc_score = guide_score("\n\n".join(assemble()), guide)
        for sweep in range(2):
            improved = False
            for k, s in enumerate(states):
                for _, cand in sorted(s.get("pool", []), key=lambda x: x[0])[:5]:
                    if cand == s["best"]:
                        continue
                    if cancel():
                        raise RuntimeError("cancelled")
                    progress(f"Whole-essay check: paragraph {s['para'] + 1}", 0.85 + 0.15 * (sweep + k / len(states)) / 2)
                    old = s["best"]
                    s["best"] = cand
                    sc = guide_score("\n\n".join(assemble()), guide)
                    if sc < doc_score:
                        doc_score, improved = sc, True
                    else:
                        s["best"] = old
            if not improved:
                break
    out_paras = assemble()
    output = "\n\n".join(out_paras)
    progress("Scoring final text", 1.0)
    return {"output": output, "used_ai": use_ai, "notes": notes, "seconds": round(time.time() - t0, 1)}


def report(original, output, mode="balanced"):
    names = detectors.available()
    return {
        "before": detectors.score_all(original, names),
        "after": detectors.score_all(output, names),
        "similarity": round(meaning.similarity(original, output), 3),
        "labels": {n: detectors.DETECTOR_CLASSES[n].label for n in names},
        "judges": [] if mode == "max" else JUDGE_DETECTORS,
    }
