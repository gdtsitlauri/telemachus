"""Visual mission grounding benchmark with certified escalation to a human.

Procedurally generated top-down scenes contain 2-3 landing pads (colour x
shape) plus distractors, under noise, low light and occlusion. The mission
names one pad ("the yellow triangle"); the pinned VLM must answer left /
middle / right. The answer probability gates escalation: below theta the
operator is asked (expensive, always correct). Learn-then-Test chooses theta
so that, with probability >= 1 - delta, P(wrong mission goal) <= alpha.

Usage:
  python -m telemachus.vlm_benchmark --n 600 --output results/vlm_benchmark
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .calibration import learn_then_test

COLORS = {"red": (220, 30, 30), "blue": (30, 70, 220), "green": (30, 160, 60), "yellow": (235, 200, 20)}
SHAPES = ("square", "circle", "triangle")
SIDES3 = ("left", "middle", "right")


def make_scene(rng: random.Random, size=(360, 180)):
    """Return image, instruction, expected side, difficulty metadata."""
    n = rng.choice((2, 3))
    sides = ("left", "right") if n == 2 else SIDES3
    pads = []
    while len(pads) < n:
        c, s = rng.choice(list(COLORS)), rng.choice(SHAPES)
        if (c, s) not in pads:
            pads.append((c, s))
    target = rng.randrange(n)
    tc, ts = pads[target]
    # Name the shape only when the colour alone is ambiguous.
    ambiguous = sum(c == tc for c, _ in pads) > 1
    desc = f"{tc} {ts}" if ambiguous or rng.random() < .5 else f"{tc} landing pad"
    brightness = rng.uniform(.35, 1.)
    noise = rng.uniform(0., .25)
    occlusion = rng.random() < .3
    img = Image.new("RGB", size, (236, 236, 230))
    d = ImageDraw.Draw(img)
    w = size[0] // n
    for i, (c, s) in enumerate(pads):
        cx, cy, r = w * i + w // 2 + rng.randint(-12, 12), size[1] // 2 + rng.randint(-15, 15), rng.randint(28, 40)
        box = (cx - r, cy - r, cx + r, cy + r)
        if s == "square":
            d.rectangle(box, fill=COLORS[c], outline=(40, 40, 40), width=3)
        elif s == "circle":
            d.ellipse(box, fill=COLORS[c], outline=(40, 40, 40), width=3)
        else:
            d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=COLORS[c], outline=(40, 40, 40))
    for _ in range(rng.randint(0, 3)):  # small grey distractor blocks
        x, y = rng.randint(0, size[0] - 20), rng.choice((rng.randint(0, 20), rng.randint(size[1] - 30, size[1] - 10)))
        d.rectangle((x, y, x + 18, y + 12), fill=(120, 120, 120))
    if occlusion:
        x = rng.randint(0, size[0] - 60)
        d.rectangle((x, 0, x + rng.randint(15, 40), size[1]), fill=(90, 90, 90))
    arr = np.asarray(img, float) * brightness
    arr += np.random.default_rng(rng.randrange(1 << 30)).normal(0, 255 * noise, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.2)))
    meta = {"n_pads": n, "pads": pads, "target": target, "brightness": brightness, "noise": noise,
            "occlusion": occlusion, "ambiguous_colour": ambiguous}
    return img, f"Identify the {desc}", sides[target], sides, meta


class ScoredResolver:
    """Pinned SmolVLM2 with the probability of its chosen side."""

    def __init__(self, cache_dir=None, device="cpu"):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from .vlm import MODEL_ID, MODEL_REVISION
        self.torch, self.device = torch, device
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, cache_dir=cache_dir)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, cache_dir=cache_dir,
            dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
        self.model_id, self.revision = MODEL_ID, MODEL_REVISION

    def resolve(self, image, instruction, sides):
        torch = self.torch
        options = " or ".join(sides)
        q = f"{instruction}. Is it on the {options} of the image? Answer with one word: {options}."
        msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": q}]}]
        prompt = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(images=[image], text=prompt, return_tensors="pt").to(self.device)
        if self.device == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1].float()
        logp = torch.log_softmax(logits, -1)
        # Score each option by its first-token probability (with and without a leading space).
        scores = {}
        for s in sides:
            ids = {self.processor.tokenizer.encode(v, add_special_tokens=False)[0] for v in (s, " " + s, s.capitalize(), " " + s.capitalize())}
            scores[s] = float(torch.logsumexp(logp[list(ids)], 0))
        z = np.logaddexp.reduce(list(scores.values()))
        probs = {s: float(np.exp(v - z)) for s, v in scores.items()}
        side = max(probs, key=probs.get)
        return {"side": side, "probability": probs[side], "option_probabilities": probs}


def run_benchmark(n: int, output: Path, seed: int = 2026, cache_dir=None, device="cpu"):
    output.mkdir(parents=True, exist_ok=True)
    table_path = output / "table.jsonl"
    done = {json.loads(l)["i"] for l in table_path.read_text().splitlines()} if table_path.exists() else set()
    resolver = ScoredResolver(cache_dir, device)
    rng = random.Random(seed)
    t0 = time.time()
    with table_path.open("a") as f:
        for i in range(n):
            img, instr, expected, sides, meta = make_scene(rng)
            if i in done:
                continue
            r = resolver.resolve(img, instr, sides)
            rec = {"i": i, "instruction": instr, "expected": expected, "side": r["side"],
                   "probability": r["probability"], "correct": r["side"] == expected, "meta": meta}
            f.write(json.dumps(rec) + "\n")
            if i < 12:
                img.save(output / f"scene_{i:03d}.png")
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{n} scenes, {time.time() - t0:.0f}s", flush=True)
    return [json.loads(l) for l in table_path.read_text().splitlines()]


def certify(table: List[Dict], alpha=.05, delta=.1, n_cal=None, expert_cost=1.):
    rows = sorted(table, key=lambda r: r["i"])
    n_cal = n_cal or len(rows) // 2
    cal, test = rows[:n_cal], rows[n_cal:]
    grid = [1.01] + [round(x, 3) for x in np.linspace(.99, 0., 100)]  # most -> least escalation

    def run(data, theta):
        ask = [r["probability"] < theta for r in data]
        wrong = [int(not a and not r["correct"]) for a, r in zip(ask, data)]
        return {"n": len(data), "wrong": int(sum(wrong)), "error": float(np.mean(wrong)),
                "expert_rate": float(np.mean(ask))}
    counts = []
    for th in grid:
        r = run(cal, th)
        counts.append(({"theta": th}, r["n"], r["wrong"]))
        sel, res = learn_then_test(counts, alpha, delta)
        if not res[-1].rejected:
            break
    sel, _ = learn_then_test(counts, alpha, delta)
    theta = None if sel is None else grid[sel]
    report = {"alpha": alpha, "delta": delta, "n_calibration": len(cal), "n_test": len(test),
              "vlm_accuracy_test": float(np.mean([r["correct"] for r in test])),
              "selected_theta": theta, "test": run(test, theta) if theta is not None else None,
              "by_n_pads": {k: float(np.mean([r["correct"] for r in test if r["meta"]["n_pads"] == k]))
                            for k in (2, 3) if any(r["meta"]["n_pads"] == k for r in test)}}
    if report["test"]:
        report["test"]["within_alpha"] = report["test"]["error"] <= alpha
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--alpha", type=float, default=.05)
    ap.add_argument("--delta", type=float, default=.1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cache-dir", type=Path, default=Path(".cache/models"))
    ap.add_argument("--output", type=Path, default=Path("results/vlm_benchmark"))
    a = ap.parse_args()
    table = run_benchmark(a.n, a.output, cache_dir=a.cache_dir, device=a.device)
    report = certify(table, a.alpha, a.delta)
    (a.output / "certified_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
