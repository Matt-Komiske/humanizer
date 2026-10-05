"""Run the humanizer on every text in eval/testset.json and report detector scores.
Usage: python eval/run_eval.py [quick|balanced|thorough|max] [indices, e.g. 0,2,4] [nostructure]"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "models"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, ROOT)
from humanizer import pipeline  # noqa: E402


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "balanced"
    tests = json.load(open(os.path.join(HERE, "testset.json")))
    if len(sys.argv) > 2 and sys.argv[2] != "all":
        tests = [tests[int(i)] for i in sys.argv[2].split(",")]
    restructure = "nostructure" not in sys.argv
    tag = mode + ("" if restructure else "_nostructure")
    from humanizer import detectors
    rows = []
    for i, t in enumerate(tests):
        t0 = time.time()
        res = pipeline.humanize(t["text"], mode=mode, restructure=restructure)
        rep = pipeline.report(t["text"], res["output"])
        # per-paragraph scores, to compare "each paragraph alone" with "the whole essay"
        paras = [p for p in res["output"].split("\n\n") if len(p.split()) >= 25]
        per_para = {n: round(sum(detectors.score_all(p, [n])[n] for p in paras) / max(1, len(paras)), 3)
                    for n in rep["after"]}
        rows.append({"source": t["source"], "words": len(t["text"].split()), "seconds": round(time.time() - t0, 1),
                     **rep, "para_avg_after": per_para, "notes": res.get("notes"), "output": res["output"]})
        b, a = rep["before"], rep["after"]
        print(f"[{i + 1}/{len(tests)}] {t['source']:12} {rows[-1]['words']:4}w {rows[-1]['seconds']:6.1f}s  sim {rep['similarity']:.2f}  " +
              "  ".join(f"{k} {b[k]:.2f}->{a[k]:.2f} (paras {per_para[k]:.2f})" for k in a), flush=True)
    names = list(rows[0]["after"])
    summary = {"mode": tag, "n": len(rows),
               "avg_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 1),
               "avg_similarity": round(sum(r["similarity"] for r in rows) / len(rows), 3)}
    for n in names:
        summary[n] = {
            "avg_before": round(sum(r["before"][n] for r in rows) / len(rows), 3),
            "avg_after": round(sum(r["after"][n] for r in rows) / len(rows), 3),
            "pass_before": sum(r["before"][n] < 0.5 for r in rows),
            "pass_after": sum(r["after"][n] < 0.5 for r in rows),
        }
    print(json.dumps(summary, indent=1))
    json.dump({"summary": summary, "rows": rows}, open(os.path.join(HERE, f"results_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
