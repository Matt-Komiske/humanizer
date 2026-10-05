"""Local, free AI-text detectors. Every score is a probability-like value in [0, 1],
where higher means "more likely AI-generated"."""
import glob
import math
import os
import re
import threading

import torch
import torch.nn as nn
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          PreTrainedModel, RobertaConfig, RobertaModel)

_LOCK = threading.RLock()


def device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _half():
    """Half precision on GPUs halves memory; CPUs stay in full precision."""
    return torch.float16 if device() != "cpu" else torch.float32


def _snapshot(repo):
    """Path of a model already in the local Hugging Face cache (no network)."""
    from huggingface_hub.constants import HF_HUB_CACHE
    hits = glob.glob(os.path.join(HF_HUB_CACHE, "models--" + repo.replace("/", "--"), "snapshots", "*", "config.json"))
    if not hits:
        raise FileNotFoundError(f"{repo} not downloaded; run setup")
    return os.path.dirname(hits[0])


# --------------------------------------------------------------------------- desklib
class _DesklibModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, **_):
        h = self.model(input_ids, attention_mask=attention_mask)[0]
        m = attention_mask.unsqueeze(-1).expand(h.size()).float()
        pooled = (h * m).sum(1) / torch.clamp(m.sum(1), min=1e-9)
        return self.classifier(pooled)


class Desklib:
    """DeBERTa-v3-large classifier; self-reported top of the RAID leaderboard."""
    name = "desklib"
    label = "Desklib (DeBERTa)"
    repo = "desklib/ai-text-detector-v1.01"

    def __init__(self):
        path = _snapshot(self.repo)
        self.tok = AutoTokenizer.from_pretrained(path)
        # fp16 DeBERTa trips an assertion on Apple GPUs, so halve it only on CUDA
        self.model = _DesklibModel.from_pretrained(path).to(device(), torch.float16 if device() == "cuda" else torch.float32).eval()

    @torch.no_grad()
    def score(self, text):
        enc = self.tok(text, truncation=True, max_length=768, return_tensors="pt").to(device())
        return torch.sigmoid(self.model(**enc).float()).item()


# ----------------------------------------------------------------------- SuperAnnotate
class SuperAnnotate:
    """RoBERTa-large tuned for a low false-positive rate. Used as a held-out judge:
    the humanizer never optimizes against it, so it gives an honest check."""
    name = "superannotate"
    label = "SuperAnnotate (RoBERTa)"
    repo = "SuperAnnotate/ai-detector-low-fpr"

    def __init__(self):
        from safetensors.torch import load_file
        path = _snapshot(self.repo)
        cfg = RobertaConfig(vocab_size=50265, hidden_size=1024, num_hidden_layers=24,
                            num_attention_heads=16, intermediate_size=4096,
                            max_position_embeddings=514, type_vocab_size=1,
                            layer_norm_eps=1e-5, pad_token_id=1)
        self.roberta = RobertaModel(cfg, add_pooling_layer=False)
        self.dense = nn.Linear(1024, 1)
        sd = load_file(os.path.join(path, "model.safetensors"))
        self.roberta.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("roberta.")}, strict=False)
        self.dense.load_state_dict({"weight": sd["dense.weight"], "bias": sd["dense.bias"]})
        self.roberta.to(device(), _half()).eval()
        self.dense.to(device(), _half()).eval()
        self.tok = AutoTokenizer.from_pretrained(path)

    @torch.no_grad()
    def score(self, text):
        enc = self.tok(text, truncation=True, max_length=512, return_tensors="pt").to(device())
        h = self.roberta(**enc)[0][:, 0, :]
        return torch.sigmoid(self.dense(h).float()).item()


# --------------------------------------------------------------------------- Binoculars
class Binoculars:
    """Zero-shot detector (Hans et al. 2024): ratio of a text's perplexity to the
    cross-perplexity between a base and an instruct model. Low ratio = AI.
    Uses small Qwen2.5-0.5B models; the threshold is calibrated locally (see eval/)."""
    name = "binoculars"
    label = "Binoculars (zero-shot)"
    observer_repo = "Qwen/Qwen2.5-0.5B"
    performer_repo = "Qwen/Qwen2.5-0.5B-Instruct"
    # Raw-score threshold where the classifier flips; mapped to 0.5 probability.
    threshold = float(os.environ.get("BINOCULARS_THRESHOLD", "0.88"))
    scale = 0.05

    def __init__(self):
        dt = torch.float16 if device() != "cpu" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(_snapshot(self.observer_repo))
        self.obs = AutoModelForCausalLM.from_pretrained(_snapshot(self.observer_repo), dtype=dt).to(device()).eval()
        self.perf = AutoModelForCausalLM.from_pretrained(_snapshot(self.performer_repo), dtype=dt).to(device()).eval()

    @torch.no_grad()
    def raw(self, text):
        enc = self.tok(text, truncation=True, max_length=512, return_tensors="pt").to(device())
        ids = enc.input_ids
        if ids.shape[1] < 3:
            return None
        lo = self.obs(**enc).logits[:, :-1].float()
        lp = self.perf(**enc).logits[:, :-1].float()
        tgt = ids[:, 1:]
        ppl = nn.functional.cross_entropy(lp.transpose(1, 2), tgt).item()
        x_ppl = (torch.softmax(lo, -1) * -torch.log_softmax(lp, -1)).sum(-1).mean().item()
        return ppl / x_ppl

    def score(self, text):
        r = self.raw(text)
        if r is None:
            return 0.5
        return 1 / (1 + math.exp((r - self.threshold) / self.scale))


# --------------------------------------------------------------------------- registry
DETECTOR_CLASSES = {c.name: c for c in (Desklib, Binoculars, SuperAnnotate)}
_loaded = {}


def get(name):
    with _LOCK:
        if name not in _loaded:
            _loaded[name] = DETECTOR_CLASSES[name]()
        return _loaded[name]


def available():
    out = []
    for name, cls in DETECTOR_CLASSES.items():
        repos = [getattr(cls, a) for a in ("repo", "observer_repo", "performer_repo") if hasattr(cls, a)]
        try:
            for r in repos:
                _snapshot(r)
            out.append(name)
        except Exception:
            pass
    return out


def _windows(text, max_words=320):
    """Sentence-aligned windows that fit the detectors' 512-token limit."""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    out, cur = [], []
    for s in sents:
        if cur and len(" ".join(cur + [s]).split()) > max_words:
            out.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        out.append(" ".join(cur))
    return out


def score_text(name, text):
    """Score a whole document. Long texts are scored in windows and averaged by length,
    which is how detectors' own tools handle documents past their token limit."""
    det = get(name)
    wins = _windows(text)
    if len(wins) == 1:
        return det.score(text)
    w = [len(x.split()) for x in wins]
    return sum(det.score(x) * n for x, n in zip(wins, w)) / sum(w)


def score_all(text, names=None):
    names = names or available()
    with _LOCK:
        return {n: round(score_text(n, text), 4) for n in names}
