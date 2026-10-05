# Humanizer

A free, fully local tool that rewrites text so it reads as human-written while keeping the same meaning. It also scores the text with three open-source AI detectors so you can see the effect. Nothing is sent to the internet after setup, and nothing costs money.

## Starting it

| Computer | Do this |
|---|---|
| Mac | Double-click `start.command`. If macOS blocks it, right-click it, choose **Open**, then **Open** again. |
| Windows | Double-click `start.bat`. |
| Linux | Run `bash start.sh`. |

The app opens in your browser at http://127.0.0.1:7860. Paste text on the left, click **Humanize**, and the result appears on the right. **Save .docx** and **Save .txt** download the output.

### What the first run installs (all free)
- **Python 3.10+.** If it's missing, the launcher prints the download link (python.org). On Windows, tick "Add python.exe to PATH" during install.
- **Python libraries**, installed automatically into a private `.venv` folder.
- **Detector models** (about 5 GB), downloaded into `models/`. They're skipped if that folder was copied over.
- **Ollama** (optional but recommended). It's the free app that runs the AI rewriter. Install it from https://ollama.com/download (Mac: `brew install ollama` also works). After that, the launcher downloads the Llama 3.1 models automatically (about 10 GB).

Without Ollama, only **Quick (rules only)** mode works.

## Moving it to another computer

Run `bash package.sh`. It creates `humanizer.zip` (under 1 MB) with just the code. On the other computer, unzip it and double-click the launcher. The first run downloads the free models automatically (about 5 GB of detectors, plus the Ollama models).

Don't copy the `.venv` or `models` folders between computers. `.venv` only works on the computer that built it, and the model cache uses links that don't survive zipping.

**Hardware:** AI mode needs about 8 GB of RAM or more. It's fastest on Apple Silicon Macs or PCs with an NVIDIA GPU. On older laptops it still runs, just slowly. Quick mode runs on anything.

## Bigger models

The rewriting models are set in `config.json`. To use a stronger computer, change them and run the start script again; it downloads anything missing. You can also switch models in the app's dropdowns. Suggested settings:

| Computer | `chat_model` | `base_model` | Notes |
|---|---|---|---|
| 16 GB Mac / 8 GB GPU (default) | `llama3.1:8b` | `llama3.1:8b-text-q4_K_M` | Fast |
| 8–12 GB GPU | `llama3.1:8b` | `gemma2:9b-text-q4_K_M` | Fits on the GPU. In testing it did a little better than the Llama base model. |
| 12 GB GPU + 64 GB RAM (e.g. RTX 4070 Super) — **recommended** | `llama3.1:8b` | `gemma2:27b-text-q4_K_M` | About 17 GB, split between GPU and RAM; several times slower |
| 12 GB GPU + 64 GB RAM, time no object | `llama3.1:8b` | `llama3.1:70b-text-q4_K_M` | About 43 GB, mostly in RAM. Slow (a few words per second), so a page in Max mode can take an hour or more. |

The base model does most of the work (2 of every 3 candidates), so that's where a bigger model matters. Keeping the chat model small avoids swapping two huge models in and out of memory.

**NVIDIA PCs:** the Windows launcher installs the GPU build of PyTorch automatically. It needs a current NVIDIA driver (update it through the NVIDIA app or nvidia.com/drivers). Ollama uses the GPU on its own.

## Modes

| Mode | What it does | Speed (500 words, M2 Pro) |
|---|---|---|
| Quick | Rule-based cleanup only: AI phrases, contractions, dashes, semicolons | Instant |
| Balanced | Each paragraph is rewritten 3 ways (2 by the base model, 1 by the chat model), up to 2 rounds; the most human-scoring version that keeps the meaning wins | ~3–5 min |
| Thorough | 6 candidates per paragraph, up to 3 rounds | ~8–12 min |
| Max | 9 candidates, up to 4 rounds, steers on all three detectors (none held out) | ~15–25 min |

**Tone** controls the register: Standard, Casual, or Academic.

## How it works

1. **Whole-essay structure pass** (essays of 3+ paragraphs). AI essays share a skeleton: an intro that previews every point, body paragraphs that open with a topic sentence and close with a summary of themselves, "Firstly/Additionally/Finally" openers, even paragraph lengths, and a conclusion that recaps everything. The structure pass:
   - deletes summary closers, intro previews and bare transition openers (they only repeat content, so no meaning is lost)
   - merges two short paragraphs or splits a long one, so paragraph lengths are uneven
   - gives each body paragraph a randomly chosen treatment (state the point late, leave it implied, or open plainly). Neighbors never get the same treatment, so main ideas surface at irregular points.
   - rewrites the conclusion so it doesn't recap
2. **Paragraph rewriting** with local LLMs (Ollama). A base model (2 of every 3 candidates) and a chat model each rewrite every paragraph several times. They use plain-language rules (no stock AI vocabulary, varied sentence length, contractions, no em dashes) and real human writing samples as a voice guide.
3. **Rule pass.** About 80 phrase rewrites ("delve into" → "dig into"), contractions, and punctuation fixes.
4. **Faithfulness checks.** A candidate is rejected if its meaning similarity to the source falls below the threshold, or if it adds a number or name that wasn't in the source.
5. **Whole-essay selection.** Each paragraph's version is chosen by how the assembled essay scores, not how that paragraph scores alone, because detectors judge the whole document. Then a final pass strips any summary closers or transition openers the models put back.

## The detectors (all local, free)

| Detector | Type | Role |
|---|---|---|
| Desklib (`desklib/ai-text-detector-v1.01`) | DeBERTa-v3-large classifier; self-reported top of the RAID benchmark | **Held out.** It is never used for steering, so it is the honest check. |
| Binoculars (Qwen2.5-0.5B base + instruct) | Zero-shot perplexity ratio | Guides the rewrite |
| SuperAnnotate (`SuperAnnotate/ai-detector-low-fpr`) | RoBERTa-large classifier | Guides the rewrite |

Calibration on real human vs AI text is in `eval/calibration_result.json`. Test-set results are in `eval/results_*.json`.

**Limits.** Commercial detectors (GPTZero, Turnitin, Originality.ai, etc.) use their own models and aren't available for free automated testing. You can paste output into their free web pages to spot-check. No humanizer beats every detector every time. Detectors also flag some genuinely human writing as AI.

## Measured results (M2 Pro, 16 GB, default 8B models)

Test texts were essays, a cover letter, a blog post and similar pieces written by Llama 3.1 8B and Claude. Scores are the probability of AI (above 0.5 = flagged).

| Detector | Before | After, Balanced mode |
|---|---|---|
| Binoculars (zero-shot perplexity, the same family of signal GPTZero-style tools use) | 0.41–0.85 | 0.03–0.16. **Beaten in every run.** |
| SuperAnnotate (trained RoBERTa-large classifier) | 1.00 | 0.04 / 0.81 / 1.00 / 1.00. **Beaten only sometimes.** |
| Desklib (trained DeBERTa-large classifier, strongest of the three) | 1.00 | 0.94–1.00. **Never beaten.** |

Meaning similarity stayed between 0.84 and 0.94. Rules-only (Quick) mode moves none of the classifiers.

Per-paragraph candidate tests:
- Llama 8B base: 12 of 18 candidates kept the meaning. 2 passed SuperAnnotate, 0 passed desklib (best 0.78).
- Gemma 2 9B base: 15 of 18 kept the meaning. 4 passed SuperAnnotate, 0 passed desklib (best 0.82).

A stronger or different base model helps somewhat. No setup tested here got text past desklib while keeping the meaning. 27B and 70B models were not tested; they don't fit in 16 GB.

## Re-running the evaluation

```
.venv/bin/python eval/build_data.py      # rebuild calibration/test sets (needs Ollama)
.venv/bin/python eval/calibrate.py       # detector accuracy on human vs AI text
.venv/bin/python eval/run_eval.py balanced
```

## Data and credits

- `eval/calibration.json`, `eval/testset.json` and `humanizer/style_examples.json` contain text derived from the HC3 dataset ([Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)): Guo et al., 2023, "How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection Dimensions" (arXiv:2301.07597). HC3 is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), and these derived data files are shared under the same license.
- The detector and helper models (`desklib/ai-text-detector-v1.01`, `SuperAnnotate/ai-detector-low-fpr`, `Qwen/Qwen2.5-0.5B`, `sentence-transformers/all-MiniLM-L6-v2`) are not included in this repository. They are downloaded at runtime from Hugging Face and are covered by their own licenses.
