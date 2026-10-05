"""Whole-essay structure pass.

AI essays share a skeleton: an intro that previews every point, body paragraphs that each
open with a topic sentence and close with a summary of themselves, formulaic openers
("Firstly", "Another", "Finally"), even paragraph lengths, and a conclusion that restates it
all. This module finds those patterns and breaks them up irregularly, the way a person's
draft is irregular, without dropping any actual content."""
import random
import re

from . import meaning, rewriter

STRUCT_MIN_SIM = 0.85
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")

SUMMARY_CUES = re.compile(
    r"^(?:overall|in short|in summary|in sum|to summarize|all in all|ultimately|in the end|"
    r"thus|therefore|hence|clearly|this (?:shows|means|highlights|demonstrates|illustrates|is why|underscores)|"
    r"these (?:examples|factors|points|benefits|challenges)|by (?:\w+ing)\b|together,|"
    r"as (?:a result|such)|in this way|that(?:'s| is) why)\b", re.IGNORECASE)

# Safe to delete outright: a bare transition word/phrase followed by a comma.
OPENER_STRIP = re.compile(
    r"^(?:first(?:ly)?|second(?:ly)?|third(?:ly)?|lastly|finally|additionally|furthermore|moreover|"
    r"in addition|to begin with|to start with|also)\s*,\s*", re.IGNORECASE)
# Formulaic but load-bearing ("Another reason is..."): flagged, and the model rephrases it.
OPENER_FLAG = re.compile(r"^(?:another (?:\w+ ){0,2}(?:reason|benefit|factor|aspect|way|advantage|issue|concern)|"
                         r"one of the (?:most |main |key )?\w+)\b", re.IGNORECASE)

ROADMAP = re.compile(
    r"\b(?:this (?:essay|article|post|paper)|in this (?:essay|article|post|paper)|we will|i will|"
    r"will (?:explore|discuss|examine|look at|outline|cover))\b", re.IGNORECASE)

CONCLUSION_CUES = re.compile(r"^(?:in conclusion|to conclude|in summary|overall|ultimately|in the end|"
                             r"to sum up|all in all)\b", re.IGNORECASE)


def sentences(para):
    return [s.strip() for s in SENT_SPLIT.split(para.strip()) if s.strip()]


def _restates(sent, others):
    """True if a sentence mostly repeats the rest of its paragraph."""
    if not others:
        return False
    return meaning.similarity(sent, " ".join(others)) >= 0.78


# ------------------------------------------------------------------- analysis
def analyze(paras):
    n = len(paras)
    info = []
    for i, p in enumerate(paras):
        ss = sentences(p)
        last = ss[-1] if ss else ""
        info.append({
            "sentences": len(ss),
            "words": len(p.split()),
            "summary_closer": len(ss) >= 3 and (bool(SUMMARY_CUES.match(last)) or _restates(last, ss[:-1])),
            "formulaic_opener": bool(ss) and bool(OPENER_STRIP.match(ss[0]) or OPENER_FLAG.match(ss[0])),
            "roadmap": i == 0 and any(ROADMAP.search(s) for s in ss),
            "recap_conclusion": i == n - 1 and n >= 3 and (bool(CONCLUSION_CUES.match(p)) or
                                                          _restates(p, paras[1:-1])),
        })
    return info


# ------------------------------------------------------------ deterministic edits
def strip_summary_closer(para):
    ss = sentences(para)
    if len(ss) >= 3 and (SUMMARY_CUES.match(ss[-1]) or _restates(ss[-1], ss[:-1])):
        ss = ss[:-1]
    return " ".join(ss)


def strip_opener(para):
    m = OPENER_STRIP.match(para)
    if not m or m.end() >= len(para) - 1:
        return para
    rest = para[m.end():]
    return rest[0].upper() + rest[1:]


def strip_roadmap(para):
    ss = sentences(para)
    kept = [s for s in ss if not ROADMAP.search(s)]
    return " ".join(kept) if kept else para


# ------------------------------------------------------------------- the plan
TREATMENTS = {
    "late_point": ("Open with a concrete detail, example or consequence from this paragraph. "
                   "State the paragraph's main point partway through or near the end, not in the first sentence."),
    "implied": ("Don't state the paragraph's main point outright in a topic sentence. Let it come through "
                "the details and examples so a reader infers it."),
    "plain": "Keep a natural opening sentence, but don't make it a formal topic sentence.",
}
CONCLUSION = ("This is the last paragraph. Don't start with a summary phrase like 'In conclusion' or 'Overall'. "
              "Keep this paragraph's own points, but you may shorten any sentence that just lists earlier points again. "
              "End on one of this paragraph's existing points rather than a big-picture closing line.")
COMMON = ("Don't end the paragraph with a sentence that sums up the paragraph. "
          "Don't open with a transition word like Firstly, Another, Additionally or Finally. "
          "Keep every fact and claim in the paragraph; don't add new ones.")


def make_plan(paras, info, rng):
    """Assign each body paragraph a treatment so that main ideas surface at irregular points.
    Never the same treatment twice in a row, so the essay has no repeating rhythm."""
    plan, prev = [], None
    for i in range(len(paras)):
        if i == len(paras) - 1 and len(paras) >= 3:
            plan.append("conclusion")
            continue
        if i == 0:
            plan.append("intro")
            continue
        choices = [t for t in TREATMENTS if t != prev]
        weights = [0.45 if t == "late_point" else 0.3 if t == "implied" else 0.25 for t in choices]
        t = rng.choices(choices, weights)[0]
        plan.append(t)
        prev = t
    return plan


def reshape_lengths(paras, rng):
    """Merge one pair of short neighbours or split one long paragraph, so paragraph lengths
    aren't uniform. Body paragraphs only; intro and conclusion stay put."""
    if len(paras) < 4:
        return paras
    words = [len(p.split()) for p in paras]
    body = range(1, len(paras) - 1)
    if rng.random() < 0.5:
        pairs = [(i, words[i] + words[i + 1]) for i in body if i + 1 < len(paras) - 1]
        pairs = [p for p in pairs if p[1] <= 170]
        if pairs:
            i = min(pairs, key=lambda p: p[1])[0]
            return paras[:i] + [paras[i] + " " + paras[i + 1]] + paras[i + 2:]
    longest = max(body, key=lambda i: words[i])
    ss = sentences(paras[longest])
    if words[longest] >= 110 and len(ss) >= 5:
        cut = rng.randint(2, len(ss) - 2)
        return paras[:longest] + [" ".join(ss[:cut]), " ".join(ss[cut:])] + paras[longest + 1:]
    return paras


# -------------------------------------------------------------------- run it
def restructure(text_paras, tone="standard", model=None, seed=None, progress=lambda *a: None):
    """Returns (new paragraphs, notes). Uses the chat model to apply the per-paragraph plan."""
    rng = random.Random(seed)
    paras = list(text_paras)
    if len(paras) < 3:
        return paras, ["Too short for a structure pass"]
    info = analyze(paras)
    notes = []

    # 1. Deterministic removal of redundant scaffolding
    out = []
    for i, (p, f) in enumerate(zip(paras, info)):
        q = p
        if f["roadmap"]:
            q = strip_roadmap(q)
            notes.append(f"P{i + 1}: removed intro preview sentence")
        if f["summary_closer"] and i != len(paras) - 1:
            q = strip_summary_closer(q)
            notes.append(f"P{i + 1}: removed summary closer")
        if f["formulaic_opener"] and i > 0:
            q = strip_opener(q)
            if q != p:
                notes.append(f"P{i + 1}: removed formulaic opener")
        out.append(q)
    paras = out

    # 2. Irregular paragraph lengths
    before = len(paras)
    paras = reshape_lengths(paras, rng)
    if len(paras) != before:
        notes.append("merged two short paragraphs" if len(paras) < before else "split a long paragraph")

    # 3. Per-paragraph structural rewrite following an irregular plan
    plan = make_plan(paras, analyze(paras), rng)
    result = []
    essay = "\n\n".join(paras)
    for i, (p, t) in enumerate(zip(paras, plan)):
        progress(f"Restructuring paragraph {i + 1} of {len(paras)}", i / len(paras))
        if t == "intro" or len(p.split()) < 25:
            result.append(p)
            continue
        instr = (CONCLUSION if t == "conclusion" else TREATMENTS[t]) + " " + COMMON
        # Strict meaning check: structural edits must not add or drop content. If no attempt
        # qualifies, the paragraph keeps its original wording (the humanizer still rewrites it).
        best = None
        for attempt in range(3):
            try:
                cand = rewriter.restructure_paragraph(p, essay, instr, tone=tone, model=model,
                                                      temperature=0.7 + 0.15 * attempt)
            except Exception:
                break
            if cand and meaning.faithful(p, cand, STRUCT_MIN_SIM):
                best = cand
                break
        result.append(best or p)
        notes.append(f"P{i + 1}: {t.replace('_', ' ')}" + ("" if best else " (skipped: meaning drifted)"))
    return result, notes


def final_cleanup(paras):
    """Holistic last pass after humanizing: strip scaffolding the models may have put back."""
    out = []
    for i, p in enumerate(paras):
        if i > 0:
            p = strip_opener(p)
        if 0 < i < len(paras) - 1:
            ss = sentences(p)
            if len(ss) >= 3 and SUMMARY_CUES.match(ss[-1]):
                p = " ".join(ss[:-1])
        out.append(p)
    return out
