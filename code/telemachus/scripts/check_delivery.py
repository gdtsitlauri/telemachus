from pathlib import Path
import json
import numpy as np
from telemachus.config import ControllerConfig
from telemachus.experiments import verify_protocol,summarize,flatten_result
from telemachus.provenance import write_json,file_hash,canonical_hash


def audit(root):
    root=Path(root);reports={}
    for folder in ("holdout","holdout_budget30","stress"):
        p=root/"results"/folder
        protocol=json.loads((p/"protocol.json").read_text());verify_protocol(root,protocol)
        cfg=ControllerConfig.from_dict(protocol["config"])
        episodes=[json.loads(f.read_text()) for f in sorted((p/"episodes").glob("*.json"))]
        expected=len(protocol["seeds"])*len(protocol["scenarios"])*len(protocol["methods"])
        assert len(episodes)==expected
        assert (p/"run_manifest.json").exists()
        keys=set();worlds={}
        for r in episodes:
            k=(r["scenario"]["name"],r["seed"],r["method"])
            assert k not in keys;keys.add(k)
            assert r["protocol_hash"]==protocol["protocol_hash"]
            assert r["config_hash"]==canonical_hash(protocol["config"])
            assert r["run_code_hash"]==canonical_hash(protocol["source_hashes"])
            pair=k[:2];worlds.setdefault(pair,set()).add(r["world_hash"])
            resources=r["resources"]
            for metric,limit in (("cost","max_cost"),("mc_samples","max_mc_samples"),
                                 ("logical_latency_ms","max_logical_latency_ms"),
                                 ("observations","max_observations"),("queries","max_queries"),("probes","max_probes")):
                assert resources[metric]<=getattr(cfg.budget,limit)+1e-8
            for t in r["trace"]:
                if r["method"]!="nominal" and t["route"] in ("ACT","PROBE"):
                    assert t["selected"] is not None
                    assert t["risk_ucb"]<=cfg.risk_threshold+1e-10
                    assert t["support_lcb"]>=cfg.stability_threshold-1e-10
                    if r["method"]!="legacy_v4":
                        assert t["risk_duration_seconds"]==cfg.dt*cfg.risk_horizon
        assert all(len(v)==1 for v in worlds.values())
        assert keys=={(s["name"],seed,method) for s in protocol["scenarios"] for seed in protocol["seeds"] for method in protocol["methods"]}
        recomputed=summarize([flatten_result(r) for r in episodes])
        saved=json.loads((p/"summary.json").read_text())
        assert recomputed["methods"]==saved["methods"]
        assert recomputed["paired_comparisons"]==saved["paired_comparisons"]
        reports[folder]={"episodes":len(episodes),"paired_worlds":len(worlds),"budgets_verified":True,
                         "selected_action_bounds_verified":True,"summary_recomputed_from_raw_episodes":True,
                         "source_hashes_match_frozen_protocol":True}
    checkpoint=json.loads((root/"artifacts/resource_policy.json").read_text())
    dataset=root/"artifacts/resource_training_data_v2.json"
    assert file_hash(dataset)==checkpoint["data_sha256"]
    groups=[set(s) for s in checkpoint["splits"].values()]
    assert all(not groups[i]&groups[j] for i in range(3) for j in range(i+1,3))
    for folder in ("holdout","holdout_budget30","stress"):
        seeds=set(json.loads((root/"results"/folder/"protocol.json").read_text())["seeds"])
        assert not seeds&set.union(*groups)
    reports["policy"]={"training_data_hash_verified":True,"train_calibration_test_and_final_seeds_disjoint":True}
    write_json(root/"results/delivery_audit.json",reports)
    return reports


if __name__=="__main__":print(json.dumps(audit(Path(__file__).resolve().parents[1]),indent=2))
