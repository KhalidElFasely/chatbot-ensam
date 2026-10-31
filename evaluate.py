"""Step 5 - Evaluate the chatbot on eval/questions.jsonl and write eval/results.md.

Metrics (good slides for the presentation):
  * retrieval hit rate : the expected page is among the chunks retrieved
  * citation rate      : the answer cites at least one source, and the expected page is cited
  * refusal rate       : out-of-scope questions are refused (no invented answer)
  * latency            : seconds per answer

Usage:  python evaluate.py            (calls the Claude API: ~12 questions, a few cents)
        python evaluate.py --retrieval-only   (no API calls)
"""
import json
import statistics
import sys

import config
from rag import Retriever, SchoolAssistant


def main(retrieval_only=False):
    with open(config.ROOT / "eval" / "questions.jsonl", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f]
    retriever = Retriever()
    bot = None if retrieval_only else SchoolAssistant(retriever)
    rows = []

    for case in cases:
        hits = retriever.search(case["question"])
        row = {"question": case["question"], "in_scope": case["in_scope"]}
        if case["in_scope"]:
            row["retrieved"] = any(case["expect_url"] in h.chunk["url"] for h in hits)
        if bot:
            ans = bot.ask(case["question"])
            row["latency"] = ans.latency_s
            row["answer"] = ans.text
            if case["in_scope"]:
                row["cited"] = any(case["expect_url"] in s["url"] for s in ans.sources)
            else:
                row["refused"] = not ans.sources
        rows.append(row)
        print(f"- {case['question']}  ->  " + ", ".join(f"{k}={v}" for k, v in row.items()
                                                          if k in ("retrieved", "cited", "refused")))

    def rate(key):
        vals = [r[key] for r in rows if key in r]
        return f"{sum(vals)}/{len(vals)}" if vals else "n/a"

    lines = ["# Evaluation results", "",
             f"- Retrieval hit rate: **{rate('retrieved')}**"]
    if bot:
        lat = [r["latency"] for r in rows]
        lines += [f"- Correct page cited: **{rate('cited')}**",
                  f"- Out-of-scope refused: **{rate('refused')}**",
                  f"- Latency: median **{statistics.median(lat):.1f}s**, max {max(lat):.1f}s",
                  "", "## Answers", ""]
        for r in rows:
            lines += [f"### {r['question']}", "", r["answer"], ""]
    out = config.ROOT / "eval" / "results.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:7]))
    print("Report written to", out)


if __name__ == "__main__":
    main(retrieval_only="--retrieval-only" in sys.argv)
