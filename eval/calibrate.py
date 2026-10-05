"""Measure each detector on known human vs AI text and pick a Binoculars threshold
(raw score at ~5% false-positive rate on human text). Writes eval/calibration_result.json."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from humanizer import detectors  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def auroc(pos, neg):
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    data = json.load(open(os.path.join(HERE, "calibration.json")))
    res = {}
    for name in ["desklib", "superannotate", "binoculars"]:
        det = detectors.get(name)
        f = (lambda t: -det.raw(t)) if name == "binoculars" else det.score  # higher = more AI
        h = [f(t) for t in data["human"]]
        a = [f(t) for t in data["ai"]]
        r = {"auroc": round(auroc(a, h), 3)}
        if name == "binoculars":
            hs = sorted(h)
            thr = -hs[int(0.95 * len(hs))]  # 5% of human texts fall on the AI side
            r["threshold"] = round(thr, 4)
            r["fpr"] = round(sum(x > -thr for x in h) / len(h), 3)
            r["tpr"] = round(sum(x > -thr for x in a) / len(a), 3)
        else:
            r["fpr@0.5"] = round(sum(x > 0.5 for x in h) / len(h), 3)
            r["tpr@0.5"] = round(sum(x > 0.5 for x in a) / len(a), 3)
        res[name] = r
        print(name, r, flush=True)
    json.dump(res, open(os.path.join(HERE, "calibration_result.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
