#!/usr/bin/env python3
"""Reproducibility check: rerun parts of each study from code/ and compare with results/.

  1. navigation  code hashes equal the frozen v6 protocol hashes; 3 certified test
                 episodes re-simulated and compared with the stored per-episode rows
  2. language    certified LLM cascade (Qwen2.5-3B tables) re-evaluated
  3. multi-agent held-out seed 5 of the guaranteed online study re-run end to end
Usage (from this folder): python analysis/repro_check.py   -> writes results/repro_check.json
"""
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
TEL, PRO, R = ROOT / "code/telemachus", ROOT / "code/prometheus", ROOT / "results"
out = {}


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


# 1. navigation -----------------------------------------------------------------
sys.path.insert(0, str(TEL / "src"))
from telemachus.provenance import code_manifest  # noqa: E402
from telemachus.config import ControllerConfig  # noqa: E402
from telemachus import experiments as ex  # noqa: E402
from telemachus.simulation import scenarios  # noqa: E402
from dataclasses import replace  # noqa: E402

proto = json.loads((R / "1_navigation_certified/main/protocol.json").read_text())
manifest = code_manifest(TEL)
frozen = proto.get("code_manifest") or proto.get("source_hashes")
same = {k: manifest.get(k) == v for k, v in frozen.items()} if isinstance(frozen, dict) else {}
out["navigation_code_hashes_match_frozen_protocol"] = {"n_files": len(same), "all_match": all(same.values()),
                                                       "mismatched": [k for k, v in same.items() if not v]}
rows = list(csv.DictReader(open(R / "1_navigation_certified/main/test_episodes.csv")))
cfg = ControllerConfig.load(TEL / "configs/v6_main.json")
ex._worker_init(str(TEL), cfg.to_dict())
sc = {s.name: s for s in scenarios("certified")}
checks = []
for row in [r for r in rows if r["method"] == "telemachus"][:3]:
    c = replace(cfg, safety_margin=.04).to_dict()
    f = ex.flatten_result(ex._worker_episode((str(TEL), c, int(row["seed"]), sc[row["scenario"]], "telemachus")))
    checks.append({"seed": row["seed"], "scenario": row["scenario"], "stored": [row["outcome"], row["cost"], row["steps"]],
                   "rerun": [f["outcome"], f["cost"], f["steps"]],
                   "match": row["outcome"] == f["outcome"] and close(row["cost"], f["cost"], 1e-6) and int(row["steps"]) == f["steps"]})
out["navigation_episodes"] = checks
print("navigation:", out["navigation_code_hashes_match_frozen_protocol"]["all_match"], [c["match"] for c in checks], flush=True)

# 2. language -------------------------------------------------------------------
from telemachus.llm_agent import cascade  # noqa: E402
d = R / "5_language_agent/qwen2.5-3b"
train, test = cascade.load_table(d / "train_table.jsonl"), cascade.load_table(d / "test_table.jsonl")
stored = json.loads((d / "certified_summary.json").read_text())
r = cascade.evaluate(train, test, alpha=.3, delta=.1, n_cal=600)
llm = {}
for g in ("learned", "confidence", "agreement"):
    a, b = r["gates"][g]["test"], stored["alpha_0.3"]["gates"][g]["test"]
    llm[g] = {"stored": [b["error"], b["cost"]], "rerun": [a["error"], a["cost"]],
              "match": close(a["error"], b["error"]) and close(a["cost"], b["cost"], 1e-6)}
out["language_alpha_0.3"] = llm
print("language:", {g: v["match"] for g, v in llm.items()}, flush=True)

# 3. multi-agent ----------------------------------------------------------------
sys.path.insert(0, str(PRO / "src"))
import torch  # noqa: E402
import yaml  # noqa: E402
torch.set_num_threads(1)
from prometheus.long_horizon import LongHorizonSpec  # noqa: E402
from prometheus.long_train import default_long_methods, run_long_method  # noqa: E402
from prometheus.hp_online import run_hp_study  # noqa: E402  (grid solver; equals bisection on seed 5)
p = yaml.safe_load((R / "4_multiagent_online_shift/high_probability/hp_shift_frozen.yaml").read_text())
spec = LongHorizonSpec()
mcfg = [c for c in default_long_methods(5) if c.name == p["method"]][0]
_, _, model = run_long_method(spec, mcfg, return_model=True)
res = run_hp_study(model, spec, mcfg.comm_top_k, p["guarantee"]["alpha"], p["guarantee"]["delta"], 5,
                   p["fallback"]["declared_p0"], p["online_update"]["eta"], p["deployment"]["chunks"],
                   p["shift"]["slip_prob"], p["shift"]["cue_noise"])
st = json.loads((R / "4_multiagent_online_shift/high_probability/seeds/seed_05.json").read_text())
out["multiagent_seed_5"] = {
    "stored": [st["theta_nominal"], st["shift_online_hp"]["failure_rate"], st["auc_nominal"]["learned"]],
    "rerun": [res["theta_nominal"], res["shift_online_hp"]["failure_rate"], res["auc_nominal"]["learned"]],
    "match": st["theta_nominal"] == res["theta_nominal"] and close(st["shift_online_hp"]["failure_rate"], res["shift_online_hp"]["failure_rate"], 1e-9)}
print("multi-agent:", out["multiagent_seed_5"]["match"], flush=True)

(R / "repro_check.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
