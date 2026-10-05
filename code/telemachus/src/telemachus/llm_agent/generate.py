"""Offline outcome table: run every information source once per question.

Each question gets the outcome of ANSWER, THINK, TOOL and VOTE (greedy
decoding, fixed sampling seeds for VOTE). Gate policies are later evaluated
on this fixed table, so all methods see identical model outputs (as in
cascade studies such as FrugalGPT). EXPERT returns the gold answer.

Usage (GPU recommended):
  python -m telemachus.llm_agent.generate --split test --output results/llm/test_table.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{split}.jsonl"
SYSTEM = "You are a careful math assistant."
NUM = re.compile(r"-?\d[\d,]*\.?\d*")

PROMPTS = {
    "ANSWER": "{q}\nGive only the final numerical answer, with no explanation.",
    "THINK": "{q}\nSolve the problem step by step, concisely. End with: The answer is <number>.",
    "TOOL": ("{q}\nWrite a short Python program that computes the answer and stores it in a variable named "
             "answer. Use only arithmetic and the math module. Reply with the code in one ```python block."),
}


def load_gsm8k(split: str, cache: Path) -> List[Dict]:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"gsm8k_{split}.jsonl"
    if not path.exists():
        urllib.request.urlretrieve(GSM8K_URL.format(split=split), path)
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        item = json.loads(line)
        gold = parse_number(item["answer"].split("####")[-1])
        rows.append({"id": f"{split}-{i}", "question": item["question"], "gold": gold})
    return rows


def parse_number(text: str) -> Optional[float]:
    nums = NUM.findall(text.replace("$", ""))
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def parse_think(text: str) -> Optional[float]:
    m = re.search(r"answer is\s*:?\s*\$?(-?\d[\d,]*\.?\d*)", text, re.IGNORECASE)
    return parse_number(m.group(1)) if m else parse_number(text)


def parse_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def same(a: Optional[float], b: Optional[float]) -> bool:
    return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)


class Generator:
    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "float16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=getattr(torch, dtype)).to(device).eval()
        self.device = device

    def _chat(self, user: str, prefill: str = "") -> str:
        return self.tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                            tokenize=False, add_generation_prompt=True) + prefill

    def generate(self, prompts: List[str], max_new_tokens: int, sample: bool = False, seed: int = 0, prefill: str = ""):
        torch = self.torch
        from transformers import LogitsProcessor, LogitsProcessorList

        steps = []

        class MaxLogProb(LogitsProcessor):
            # With greedy decoding the chosen token is the argmax, so its
            # log-probability is the per-step max of log_softmax. Storing one
            # float per sequence and step avoids keeping full-vocabulary scores.
            def __call__(self, input_ids, scores):
                steps.append(torch.log_softmax(scores.float(), -1).max(-1).values)
                return scores

        enc = self.tok([self._chat(p, prefill) for p in prompts], return_tensors="pt", padding=True).to(self.device)
        if sample:
            torch.manual_seed(seed)
        with torch.no_grad():
            seqs = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=sample,
                                       temperature=.7 if sample else None, top_p=.95 if sample else None,
                                       top_k=None, pad_token_id=self.tok.pad_token_id,
                                       logits_processor=LogitsProcessorList([] if sample else [MaxLogProb()]))
        gen = seqs[:, enc["input_ids"].shape[1]:]
        mask = gen != self.tok.pad_token_id
        # Sampled sequences (VOTE) have no stored log-probabilities; agreement is used instead.
        chosen = torch.stack(steps, 1)[:, :gen.shape[1]] if steps else torch.zeros(gen.shape, device=gen.device)
        results = []
        for i in range(len(prompts)):
            n = int(mask[i].sum())
            lp = chosen[i][mask[i]]
            results.append({"text": self.tok.decode(gen[i], skip_special_tokens=True), "tokens": n,
                            "mean_logprob": float(lp.mean()) if n else 0., "min_logprob": float(lp.min()) if n else 0.})
        return results


def build_table(rows: List[Dict], gen: Generator, batch: int, vote_k: int, output: Path):
    from .sandbox import run_program
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists():
        done = {json.loads(l)["id"] for l in output.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [r for r in rows if r["id"] not in done]
    t0 = time.time()
    with output.open("a", encoding="utf-8") as f:
        for b in range(0, len(todo), batch):
            chunk = todo[b:b + batch]
            qs = [r["question"] for r in chunk]
            # The answer is pre-filled so the model completes only the number.
            ans = gen.generate([PROMPTS["ANSWER"].format(q=q) for q in qs], 12, prefill="The final answer is ")
            think = gen.generate([PROMPTS["THINK"].format(q=q) for q in qs], 384)
            tool = gen.generate([PROMPTS["TOOL"].format(q=q) for q in qs], 256)
            votes = [gen.generate([PROMPTS["THINK"].format(q=q) for q in qs], 384, sample=True, seed=1000 + k)
                     for k in range(vote_k)]
            for i, r in enumerate(chunk):
                a_val = parse_number(ans[i]["text"].split("\n")[0])
                t_val = parse_think(think[i]["text"])
                code = parse_code(tool[i]["text"])
                p_val = run_program(code)
                v_vals = [parse_think(v[i]["text"]) for v in votes]
                counts: Dict[float, int] = {}
                for v in v_vals:
                    if v is not None:
                        key = next((k for k in counts if same(k, v)), v)
                        counts[key] = counts.get(key, 0) + 1
                v_best = max(counts, key=counts.get) if counts else None
                vote = {"VOTE": {"value": v_best, "correct": same(v_best, r["gold"]),
                                 "agreement": (counts[v_best] / vote_k) if v_best is not None else 0.,
                                 "tokens": sum(v[i]["tokens"] for v in votes),
                                 "mean_logprob": sum(v[i]["mean_logprob"] for v in votes) / vote_k}} if vote_k else {}
                rec = {
                    "id": r["id"], "gold": r["gold"], **vote,
                    "ANSWER": {"value": a_val, "correct": same(a_val, r["gold"]), **{k: ans[i][k] for k in ("tokens", "mean_logprob", "min_logprob")}},
                    "THINK": {"value": t_val, "correct": same(t_val, r["gold"]), **{k: think[i][k] for k in ("tokens", "mean_logprob", "min_logprob")}},
                    "TOOL": {"value": p_val, "correct": same(p_val, r["gold"]), "executed": p_val is not None,
                             **{k: tool[i][k] for k in ("tokens", "mean_logprob", "min_logprob")}},
                }
                f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{min(b + batch, len(todo))}/{len(todo)} questions, {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "test"), required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N questions after a fixed shuffle (0 = all)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--vote-k", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache", type=Path, default=Path(".cache/gsm8k"))
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    rows = load_gsm8k(a.split, a.cache)
    if a.limit:
        import random
        random.Random(2026).shuffle(rows)
        rows = rows[:a.limit]
    build_table(rows, Generator(a.model, a.device), a.batch, a.vote_k, a.output)


if __name__ == "__main__":
    main()
