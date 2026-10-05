"""LLM rewriting through a local Ollama server (free, runs on this computer).
Uses only the standard library to talk to Ollama's HTTP API."""
import json
import os
import re
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")
CONFIG = json.load(open(_CONFIG_PATH)) if os.path.exists(_CONFIG_PATH) else {}
DEFAULT_MODEL = os.environ.get("HUMANIZER_MODEL", CONFIG.get("chat_model", "llama3.1:8b"))

TONES = {
    "standard": "a clear, natural, everyday register, like a thoughtful person writing to a colleague",
    "casual": "a relaxed, conversational register; first person and asides are fine if they fit",
    "academic": "a student or researcher writing a paper: precise and formal, but plainly worded; no first person unless the original uses it, no slang",
}

SYSTEM = """You rewrite text so it reads like a real person wrote it, while keeping EXACTLY the same meaning, facts, names, numbers and claims. Do not add new facts or opinions. Do not drop any point.

How people actually write, and what to do:
- Vary sentence length a lot. Mix some short sentences (under 7 words) with longer ones. Never write three sentences in a row of similar length.
- Use plain, specific words. Never use: delve, tapestry, testament, pivotal, crucial, robust, leverage, utilize, foster, facilitate, enhance, streamline, comprehensive, seamless, realm, landscape, underscore, showcase, notably, moreover, furthermore, additionally, ultimately, "in conclusion", "it's important to note".
- Don't open sentences with transition words. Don't stack three items in a list just for rhythm.
- Use contractions where a person would.
- No em dashes. Few semicolons. No bold, no headings, no bullet points unless the original has them.
- Don't add a summary sentence at the end. Don't hedge everything.
- Restructure sentences: change the order of clauses, merge or split sentences, choose a different but accurate angle into the point. Don't just swap synonyms.
- Write in {tone}.

Output ONLY the rewritten text. No preamble, no notes, no quotation marks around it."""

USER = """{context}Rewrite this passage:

{text}"""

_PREAMBLE = re.compile(r"^(?:here(?:'s| is)[^\n:]*:|sure[^\n]*:|rewritten[^\n]*:)\s*", re.IGNORECASE)


def available(model=None):
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=3) as r:
            names = [m["name"] for m in json.load(r).get("models", [])]
        if model is None:
            return bool(names)
        return any(n == model or n.split(":")[0] == model for n in names)
    except Exception:
        return False


def models():
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=3) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return []


def rewrite(text, tone="standard", context="", model=None, temperature=1.0, extra=""):
    model = model or DEFAULT_MODEL
    extra = ("Match the voice of these real human writing samples (style only, never copy content):\n\n"
             + "\n\n".join(style_examples(tone)) + ("\n\n" + extra if extra else ""))
    ctx = f"For context, here is the surrounding text (do not rewrite it):\n\"\"\"\n{context}\n\"\"\"\n\n" if context else ""
    system = SYSTEM.format(tone=TONES.get(tone, TONES["standard"]))
    if extra:
        system += "\n\n" + extra
    body = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": USER.format(context=ctx, text=text)},
        ],
        "options": {"temperature": temperature, "top_p": 0.95, "repeat_penalty": 1.15,
                    "num_predict": max(256, int(len(text.split()) * 3))},
    }
    req = urllib.request.Request(OLLAMA_URL + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)["message"]["content"].strip()
    out = _PREAMBLE.sub("", out).strip()
    if len(out) > 1 and out[0] == out[-1] == '"':
        out = out[1:-1].strip()
    return out


RESTRUCTURE_SYSTEM = """You are editing one paragraph of an essay so the essay's structure reads like a person's draft rather than a template. You will get the whole essay for context and one paragraph to rewrite.

Rules:
- Keep every fact, claim, name and number from the paragraph. Add nothing new.
- Follow the structural instruction exactly.
- Write in {tone}.
- Output ONLY the rewritten paragraph."""


def restructure_paragraph(para, essay, instruction, tone="standard", model=None, temperature=0.8):
    body = {
        "model": model or DEFAULT_MODEL, "stream": False,
        "messages": [
            {"role": "system", "content": RESTRUCTURE_SYSTEM.format(tone=TONES.get(tone, TONES["standard"]))},
            {"role": "user", "content": f"The whole essay, for context:\n\"\"\"\n{essay}\n\"\"\"\n\n"
                                        f"Structural instruction: {instruction}\n\nParagraph to rewrite:\n{para}"},
        ],
        "options": {"temperature": temperature, "top_p": 0.95, "repeat_penalty": 1.1,
                    "num_predict": max(256, int(len(para.split()) * 3))},
    }
    req = urllib.request.Request(OLLAMA_URL + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)["message"]["content"].strip()
    out = _PREAMBLE.sub("", out).strip().strip('"').strip()
    # drop any trailing commentary paragraph the model adds about its own edit
    parts = [p for p in re.split(r"\n\s*\n", out) if not re.match(r"^\s*[(\[]?\s*(?:note|i (?:have|'ve|kept|made|tried))\b", p, re.I)]
    return " ".join(p.strip() for p in parts).strip()


# ---------------------------------------------------------------------------- base model
# Base (non-chat) models were never tuned into the "assistant" voice that trained
# detectors key on, so their rewrites read more like people. They follow a document
# pattern instead of instructions: real human samples, then the draft, then "rewritten".
BASE_MODEL = os.environ.get("HUMANIZER_BASE_MODEL", CONFIG.get("base_model", "llama3.1:8b-text-q4_K_M"))
_EXAMPLES = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "style_examples.json")))
_STOP = ["---", "\n\n\n", "Original draft", "Draft:", "Rewritten by"]


def style_examples(tone):
    return _EXAMPLES["formal" if tone == "academic" else "casual"][:3]


def rewrite_base(text, tone="standard", model=None, temperature=1.0):
    samples = "\n\n---\n\n".join(style_examples(tone))
    prompt = (f"Below are answers people wrote in their own words.\n\n---\n\n{samples}\n\n---\n\n"
              f"Original draft (too stiff):\n{text}\n\n"
              f"Same answer rewritten by a person in their own words, keeping every point:\n")
    body = {"model": model or BASE_MODEL, "prompt": prompt, "raw": True, "stream": False,
            "options": {"temperature": temperature, "top_p": 0.95, "repeat_penalty": 1.1,
                        "num_predict": max(200, int(len(text.split()) * 2.2)), "stop": _STOP}}
    req = urllib.request.Request(OLLAMA_URL + "/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)["response"].strip()
    out = out.strip('"').strip()
    # if generation ran out of tokens mid-sentence, drop the fragment
    if out and out[-1] not in '.!?"\')':
        cut = max(out.rfind(". "), out.rfind("! "), out.rfind("? "), out.rfind(".\n"))
        if cut > len(out) * 0.6:
            out = out[:cut + 1]
    return out
