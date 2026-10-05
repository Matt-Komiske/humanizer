"""Build calibration + test sets.
calibration.json: real human answers and ChatGPT answers from HC3 (Hello-SimpleAI/HC3),
                  plus fresh Llama 3.1 essays.
testset.json:     AI-written texts the humanizer is evaluated on (none used for tuning)."""
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from humanizer import rewriter  # noqa: E402
import urllib.request  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
random.seed(7)


def detok(t):
    t = re.sub(r" ([.,!?;:%)])", r"\1", t)
    t = re.sub(r"([($]) ", r"\1", t)
    t = re.sub(r" (n't|'s|'re|'ve|'m|'ll|'d)\b", r"\1", t)
    t = re.sub(r"\" (.*?) \"", r'"\1"', t)
    t = re.sub(r" - ", "-", t)
    return t.strip()


def hc3(name, n_h, n_ai):
    rows = [json.loads(l) for l in open(os.path.join(HERE, "data", name + ".jsonl"))]
    random.shuffle(rows)
    hum, ai = [], []
    for r in rows:
        for h in r["human_answers"][:1]:
            h = detok(h)
            if 80 <= len(h.split()) <= 260 and len(hum) < n_h and "http" not in h:
                hum.append(h)
        for a in r["chatgpt_answers"][:1]:
            a = detok(a)
            if 80 <= len(a.split()) <= 260 and len(ai) < n_ai:
                ai.append(a)
    return hum, ai


def llama(prompt):
    body = {"model": rewriter.DEFAULT_MODEL, "stream": False,
            "messages": [{"role": "user", "content": prompt}], "options": {"temperature": 0.7}}
    req = urllib.request.Request(rewriter.OLLAMA_URL + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["message"]["content"].strip()


CAL_TOPICS = ["why cities have traffic jams", "how vaccines train the immune system", "the pros and cons of remote work",
              "why the sky is blue", "how credit scores work", "the history of the printing press",
              "why sleep matters for memory", "how solar panels make electricity", "what causes inflation",
              "why bees are important for farming"]
TEST_PROMPTS = [
    "Write a 250-word persuasive essay arguing that schools should start later in the morning.",
    "Write a 300-word blog post about the benefits of learning to cook at home.",
    "Write a 250-word explanation of how photosynthesis works for a high school audience.",
    "Write a 300-word essay on the impact of social media on teenage mental health.",
    "Write a 250-word cover letter for an entry-level marketing coordinator position.",
    "Write a 300-word reflection on what the Industrial Revolution changed about daily life.",
    "Write a 250-word product description for a reusable stainless steel water bottle.",
    "Write a 300-word essay about whether college should be free.",
    "Write a 250-word summary of the causes of World War I.",
    "Write a 300-word article on why people should adopt shelter pets.",
]

# Written by Claude (also AI text) to cover a different model family.
CLAUDE_SAMPLES = [
    """Climate change represents one of the most pressing challenges facing humanity today. Rising global temperatures, driven primarily by the burning of fossil fuels, are causing more frequent and severe weather events, from devastating hurricanes to prolonged droughts. These changes have far-reaching consequences for agriculture, water supplies, and human health.

Addressing this crisis requires a multifaceted approach. Governments must implement policies that incentivize renewable energy adoption and reduce carbon emissions. At the same time, individuals can contribute by making more sustainable choices in their daily lives, such as reducing energy consumption and supporting environmentally responsible businesses.

While the challenges are significant, there is reason for optimism. Technological advancements in solar and wind energy have dramatically reduced costs, making clean energy more accessible than ever before. By working together, we can build a more sustainable future for generations to come.""",
    """Effective time management is essential for academic success. Students who plan their schedules carefully are better able to balance coursework, extracurricular activities, and personal responsibilities. One useful strategy is to break large assignments into smaller, manageable tasks, each with its own deadline. This approach reduces procrastination and makes daunting projects feel more achievable.

Another key practice is prioritization. Not all tasks carry equal weight, and students should focus their energy on assignments that have the greatest impact on their grades and learning. Tools such as planners, digital calendars, and to-do lists can help students stay organized and keep track of upcoming deadlines.

Finally, it is important to remember that rest is a crucial component of productivity. Students who get adequate sleep and take regular breaks are more focused and retain information more effectively. By developing strong time management habits, students set themselves up for success both in school and beyond.""",
]


def main():
    hum, gpt = [], []
    for name, nh, na in (("reddit_eli5", 25, 10), ("open_qa", 15, 10)):
        h, a = hc3(name, nh, na)
        hum += h
        gpt += a
    print("human", len(hum), "chatgpt", len(gpt))
    lla = []
    for t in CAL_TOPICS:
        lla.append(llama(f"Write two paragraphs (about 180 words total) explaining {t}. Plain prose, no headings."))
        print("cal llama", len(lla))
    json.dump({"human": hum, "ai": gpt + lla}, open(os.path.join(HERE, "calibration.json"), "w"), indent=1)

    tests = []
    for p in TEST_PROMPTS:
        tests.append({"source": "llama3.1:8b", "prompt": p,
                      "text": llama(p + " Plain prose, no title or headings, no placeholders like [Name].")})
        print("test", len(tests))
    for s in CLAUDE_SAMPLES:
        tests.append({"source": "claude", "prompt": "", "text": s})
    json.dump(tests, open(os.path.join(HERE, "testset.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
