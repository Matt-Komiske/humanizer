"""Rule-based humanizing pass. Needs no AI model, so it runs on any computer.

Targets the surface "tells" catalogued in Wikipedia's "Signs of AI writing" and in
open-source humanizers (blader/humanizer, harshaneel/humanize): stock AI vocabulary,
filler transitions, missing contractions, em dashes and semicolons."""
import random
import re

# (pattern, replacement). Replacement "" deletes the phrase. Patterns are case-insensitive
# and matched on word boundaries; capitalization of the first letter is preserved.
PHRASES = [
    # Filler openers / hedges that add nothing
    (r"it(?:'s| is) (?:important|crucial|essential|worth) (?:to note|noting|mentioning|to mention|to remember) that ", ""),
    (r"it(?:'s| is) worth (?:noting|mentioning) that ", ""),
    (r"it should be noted that ", ""),
    (r"in today's (?:fast-paced|digital|modern|ever-changing) (?:world|age|era|landscape),? ", "today, "),
    (r"in the ever-(?:evolving|changing) (?:landscape|world|realm) of ", "in "),
    (r"at the end of the day,? ", ""),
    (r"when it comes to ", "with "),
    (r"in conclusion,? ", ""),
    (r"to sum up,? ", ""),
    (r"in summary,? ", ""),
    (r"overall,? ", ""),
    (r"ultimately,? ", "in the end, "),
    (r"(?:moreover|furthermore|additionally),? ", "also, "),
    (r"consequently,? ", "so "),
    (r"nevertheless,? ", "still, "),
    (r"nonetheless,? ", "still, "),
    (r"notably,? ", ""),
    (r"importantly,? ", ""),
    (r"thus,? ", "so "),
    (r"hence,? ", "so "),
    # Inflated vocabulary
    (r"delve(?:s)? into", "dig into"),
    (r"delving into", "digging into"),
    (r"a rich tapestry of", "a mix of"),
    (r"(?:a )?tapestry of", "a mix of"),
    (r"(?:is|stands as|serves as) a testament to", "shows"),
    (r"a testament to", "proof of"),
    (r"underscores", "shows"),
    (r"underscore", "show"),
    (r"showcasing", "showing"),
    (r"showcases", "shows"),
    (r"showcase", "show"),
    (r"pivotal", "key"),
    (r"intricacies", "details"),
    (r"intricate", "complex"),
    (r"meticulous(?:ly)?", "careful"),
    (r"leverag(?:e|ing)", "us(e|ing)"),
    (r"utiliz(?:e|es|ed|ing)", "us(e|es|ed|ing)"),
    (r"utilization", "use"),
    (r"facilitat(?:e|es|ed|ing)", "help(|s|ed|ing)"),
    (r"bolster(?:s)?", "boost"),
    (r"garner(?:s|ed)?", "get"),
    (r"streamlin(?:e|es|ed|ing)", "simplif(y|ies|ied|ying)"),
    (r"comprehensive", "thorough"),
    (r"(?:the )?realm of", "the area of"),
    (r"seamless(?:ly)?", "smooth"),
    (r"embark(?:s|ed)? on", "start"),
    (r"endeavou?rs?", "efforts"),
    (r"commendable", "good"),
    (r"plays? a (?:crucial|vital|pivotal|key|significant|important) role in", "matters for"),
    (r"serves as", "is"),
    (r"stands as", "is"),
    (r"boasts", "has"),
    (r"a (?:wide|broad|diverse) (?:range|array|variety) of", "lots of"),
    (r"a myriad of", "many"),
    (r"myriad", "many"),
    (r"a plethora of", "plenty of"),
    (r"in order to", "to"),
    (r"due to the fact that", "because"),
    (r"in light of the fact that", "since"),
    (r"resonate(?:s)? with", "appeal to"),
    (r"valuable insights", "useful ideas"),
    (r"deep dive", "close look"),
    (r"game[- ]changer", "big shift"),
    (r"navigat(?:e|ing) (?:the )?(?:complexities|challenges)", "handl(e|ing) the hard parts"),
    (r"fostering", "building"),
    (r"fosters?", "builds"),
    (r"enhanc(?:e|es|ed|ing)", "improv(e|es|ed|ing)"),
    (r"crucial", "important"),
    (r"vibrant", "lively"),
    (r"robust", "solid"),
    (r"paramount", "most important"),
    (r"(?:great|excellent) question!? ?", ""),
    (r"i hope this helps!?\.?", ""),
]

CONTRACTIONS = [
    ("do not", "don't"), ("does not", "doesn't"), ("did not", "didn't"),
    ("is not", "isn't"), ("are not", "aren't"), ("was not", "wasn't"),
    ("were not", "weren't"), ("can not", "can't"), ("cannot", "can't"),
    ("could not", "couldn't"), ("would not", "wouldn't"), ("should not", "shouldn't"),
    ("will not", "won't"), ("have not", "haven't"), ("has not", "hasn't"),
    ("it is", "it's"), ("that is", "that's"), ("there is", "there's"),
    ("they are", "they're"), ("we are", "we're"), ("you are", "you're"),
    ("i am", "I'm"),
    ("i will", "I'll"), ("we will", "we'll"), ("you will", "you'll"), ("it will", "it'll"),
    ("let us", "let's"), ("what is", "what's"), ("who is", "who's"),
]


def _expand_alt(pat, repl, matched):
    """For a pattern like 'leverag(?:e|ing)' and replacement 'us(e|ing)', pick the
    replacement option with the same index as the suffix that actually matched."""
    pm = re.search(r"\(\?:([^)]*)\)", pat)
    rm = re.search(r"\(([^)]*)\)", repl)
    p_opts, r_opts = pm.group(1).split("|"), rm.group(1).split("|")
    low = matched.lower()
    idx = max(range(len(p_opts)), key=lambda i: (low.endswith(p_opts[i]) if p_opts[i] else 0, len(p_opts[i])) if low.endswith(p_opts[i]) else (-1, 0))
    return repl[:rm.start()] + r_opts[min(idx, len(r_opts) - 1)] + repl[rm.end():]


def _match_case(src, dst):
    if not dst:
        return dst
    if src[:1].isupper():
        return dst[0].upper() + dst[1:]
    return dst


def replace_phrases(text):
    for pat, repl in PHRASES:
        rx = re.compile(r"\b" + pat, re.IGNORECASE)

        def sub(m, pat=pat, repl=repl):
            out = _expand_alt(pat, repl, m.group(0)) if "(" in repl else repl
            return _match_case(m.group(0), out)
        text = rx.sub(sub, text)
    # Deleting a sentence opener can leave a lowercase sentence start; fix it.
    text = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return text


def contract(text, rate=0.85):
    for full, short in CONTRACTIONS:
        rx = re.compile(r"\b" + full + r"\b", re.IGNORECASE)
        text = rx.sub(lambda m: _match_case(m.group(0), short) if random.random() < rate else m.group(0), text)
    return text


def punctuation(text):
    # em/en dashes used as asides -> commas or parentheses; semicolons -> periods
    text = re.sub(r"\s*[—–]\s*([^—–.!?]{1,60}?)\s*[—–]\s*", r" (\1) ", text)
    text = re.sub(r"\s*—\s*|\s+-{1,2}\s+", ", ", text)
    text = re.sub(r";\s+([a-z])", lambda m: ". " + m.group(1).upper(), text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return text


META = re.compile(r"^\s*[(\[]?\s*(?:note|notes|edit|p\.?s|i (?:tried|rewrote|have rewritten|made)|this (?:version|rewrite)|here(?:'s| is) (?:the|my|a) (?:rewritten|revised))\b.*$",
                  re.IGNORECASE | re.MULTILINE)


def tidy(text):
    text = META.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    text = re.sub(r"(\w) (n't|'s|'re|'ve|'m|'ll|'d)\b", r"\1\2", text)
    text = re.sub(r"(\ws) ' ", r"\1' ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([,.!?;:])", r"\1", text)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\(\s*", "(", text)
    text = re.sub(r"\s*\)", ")", text)
    text = re.sub(r"\bAlso, also\b", "Also", text)
    return "\n".join(line.strip() for line in text.split("\n"))


def strip_markdown(text):
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    return text


def humanize(text, seed=None):
    if seed is not None:
        random.seed(seed)
    text = strip_markdown(text)
    text = replace_phrases(text)
    text = contract(text)
    text = punctuation(text)
    return tidy(text)
