"""Local web app: http://127.0.0.1:7860"""
import io
import os
import threading
import uuid
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(HERE, "models"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from flask import Flask, jsonify, request, send_file, send_from_directory  # noqa: E402

from humanizer import detectors, pipeline, rewriter  # noqa: E402

app = Flask(__name__, static_folder=os.path.join(HERE, "static"))
JOBS = {}


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/status")
def status():
    return jsonify({
        "ollama": rewriter.available(),
        "models": rewriter.models(),
        "default_model": rewriter.DEFAULT_MODEL,
        "default_base_model": rewriter.BASE_MODEL,
        "detectors": [{"name": n, "label": detectors.DETECTOR_CLASSES[n].label,
                       "judge": n in pipeline.JUDGE_DETECTORS} for n in detectors.available()],
    })


def _run(job_id, text, mode, tone, model, base_model):
    job = JOBS[job_id]

    def progress(msg, frac):
        job.update(message=msg, progress=frac)
    try:
        res = pipeline.humanize(text, mode=mode, tone=tone, model=model or None, base_model=base_model or None,
                                progress=progress, cancel=lambda: job.get("cancel"))
        job["message"] = "Scoring before and after"
        rep = pipeline.report(text, res["output"], mode) if detectors.available() else None
        job.update(status="done", output=res["output"], report=rep, used_ai=res["used_ai"], notes=res.get("notes", []),
                   seconds=res["seconds"], progress=1.0, message="Done")
    except Exception as e:
        job.update(status="error", message=str(e))


@app.post("/api/humanize")
def humanize():
    d = request.get_json(force=True)
    text = (d.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text"}), 400
    job_id = uuid.uuid4().hex[:10]
    JOBS[job_id] = {"status": "running", "message": "Starting", "progress": 0.0}
    threading.Thread(target=_run, daemon=True,
                     args=(job_id, text, d.get("mode", "balanced"), d.get("tone", "standard"),
                           d.get("model"), d.get("base_model"))).start()
    return jsonify({"job": job_id})


@app.get("/api/job/<job_id>")
def job(job_id):
    j = JOBS.get(job_id)
    if not j:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify({k: v for k, v in j.items() if k != "cancel"})


@app.post("/api/job/<job_id>/cancel")
def cancel(job_id):
    if job_id in JOBS:
        JOBS[job_id]["cancel"] = True
    return jsonify({"ok": True})


@app.post("/api/detect")
def detect():
    text = (request.get_json(force=True).get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text"}), 400
    return jsonify({"scores": detectors.score_all(text)})


@app.post("/api/export")
def export():
    d = request.get_json(force=True)
    text, fmt = d.get("text") or "", d.get("format", "docx")
    if fmt == "txt":
        return send_file(io.BytesIO(text.encode()), as_attachment=True,
                         download_name="humanized.txt", mimetype="text/plain")
    from docx import Document
    doc = Document()
    for para in text.split("\n\n"):
        doc.add_paragraph(para.strip())
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="humanized.docx",
                     mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document")


def _warm():
    for n in detectors.available():
        try:
            detectors.get(n)
        except Exception as e:
            print("Could not load detector", n, e)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    threading.Thread(target=_warm, daemon=True).start()
    if os.environ.get("NO_BROWSER") != "1":
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    print(f"\n  Humanizer running at http://127.0.0.1:{port}  (Ctrl+C to stop)\n")
    app.run(host="127.0.0.1", port=port, threaded=True)
