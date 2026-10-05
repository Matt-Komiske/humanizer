"""Download the free models this app uses. Safe to re-run: skips anything already present.
Hugging Face models go into ./models so the whole folder can be copied to another computer."""
import json
import os
import shutil
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(HERE, "models"))
sys.path.insert(0, HERE)

HF_MODELS = [
    "desklib/ai-text-detector-v1.01",
    "SuperAnnotate/ai-detector-low-fpr",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "sentence-transformers/all-MiniLM-L6-v2",
]
_cfg = json.load(open(os.path.join(HERE, "config.json"))) if os.path.exists(os.path.join(HERE, "config.json")) else {}
OLLAMA_MODELS = [_cfg.get("chat_model", "llama3.1:8b"), _cfg.get("base_model", "llama3.1:8b-text-q4_K_M")]


def hf():
    from huggingface_hub import snapshot_download
    from humanizer.detectors import _snapshot
    for repo in HF_MODELS:
        try:
            _snapshot(repo)
            print("  ok        ", repo)
            continue
        except FileNotFoundError:
            pass
        print("  downloading", repo)
        snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])


def ollama():
    exe = shutil.which("ollama")
    if not exe:
        print("\n  Ollama is not installed, so AI rewriting is off (rules-only mode still works).")
        print("  To enable it, install Ollama for free from https://ollama.com/download and run start again.\n")
        return
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as r:
            have = {m["name"] for m in json.load(r).get("models", [])}
    except Exception:
        print("  Ollama is installed but not running; skipping model download for now.")
        return
    for m in OLLAMA_MODELS:
        if m in have:
            print("  ok        ", m)
        else:
            print("  downloading", m, "(one time; 5-45 GB depending on model size)")
            subprocess.run([exe, "pull", m], check=False)


if __name__ == "__main__":
    print("Checking models…")
    hf()
    ollama()
