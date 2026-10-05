from __future__ import annotations

from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
import math
import multiprocessing
from pathlib import Path
import csv
import json
import time
import numpy as np
from scipy.stats import binomtest
from .config import ControllerConfig
from .controller import Controller, METHODS
from .simulation import NavigationWorld, SensorActuatorClient, scenarios
from .vision import NeuralVision
from .provenance import write_json, code_manifest, environment_info, canonical_hash, file_hash, utc_now


# Seeds used during v5 development (frozen in every v5 protocol).
DEVELOPMENT_SEEDS_V5=(19,100,101,102,103,104,105,110,111,200,201,202,203,*range(1000,1016),*range(2000,2040),
                      *range(2300,2308),*range(5000,5032),*range(8000,8008))
# v6 additionally used 2400-2419 (safety pilots). 12000-19999 covers every v5
# holdout/stress range (16000-16031, 18000-18007) and earlier candidates.
# Certified calibration/test seeds must avoid all of them.
DEVELOPMENT_SEEDS=(*DEVELOPMENT_SEEDS_V5,*range(2400,2420),*range(12000,20000))


def load_resource_policy(root,cfg):
    if cfg.resource_policy_mode=="heuristic":return None
    from .resource_policy import ResourcePolicy
    if not cfg.learned_resource_policy:raise ValueError("A learned policy path is required")
    return ResourcePolicy(Path(root)/cfg.learned_resource_policy)


def run_episode(seed, scenario, cfg, method, vision,goal=None,*,root=None,resource_policy=None):
    world=NavigationWorld(seed,scenario,cfg,goal=goal)
    if method=="legacy_v4":
        from .legacy_v4.controller import Controller as LegacyController
        from .legacy_v4.config import ControllerConfig as LegacyConfig
        from .legacy_v4.config import BudgetConfig as LegacyBudget
        keys={f.name for f in fields(LegacyConfig)}
        budget_keys={f.name for f in fields(LegacyBudget)}
        values={k:v for k,v in cfg.to_dict().items() if k in keys}
        values["budget"]={k:v for k,v in values["budget"].items() if k in budget_keys}
        legacy_cfg=LegacyConfig.from_dict(values)
        agent=LegacyController(SensorActuatorClient(world,vision),vision,legacy_cfg,"telemachus",seed=seed)
    else:
        if cfg.resource_policy_mode!="heuristic" and resource_policy is None:
            if root is None:raise ValueError("Pass project root to load the declared policy")
            resource_policy=load_resource_policy(root,cfg)
        agent=Controller(SensorActuatorClient(world,vision),vision,cfg,method,seed=seed,resource_policy=resource_policy)
    result=agent.run()
    result["method"]=method
    result.update(world.truth_record())
    return result


def freeze_protocol(root, path, cfg, seeds, methods, split="holdout"):
    root=Path(root)
    seeds,methods=list(seeds),list(methods)
    if not seeds or not methods or len(set(seeds))!=len(seeds) or len(set(methods))!=len(methods):
        raise ValueError("Use nonempty unique seeds and methods")
    manifest={"version":5,"created":utc_now(),"split":split,"seeds":list(seeds),
              "methods":list(methods),"scenarios":[s.__dict__ for s in scenarios(split)],
              "config":cfg.to_dict(),"source_hashes":code_manifest(root),
              "vision_sha256":file_hash(root/cfg.vision_artifact),"environment":environment_info(),
              "endpoints":["success","collision","deferred","cost","wall_seconds"],
              "resource_policy_sha256":None if cfg.resource_policy_mode=="heuristic" else file_hash(root/cfg.learned_resource_policy),
              "primary_comparisons":[m for m in ("legacy_v4","confidence_gate","budget_voi","no_learning") if m in methods],
              "development_seeds":list(DEVELOPMENT_SEEDS_V5),
              "certificate_scope":{"executed_control_interval_seconds":cfg.dt,"v5_risk_interval_seconds":cfg.dt*cfg.risk_horizon,"v5_utility_horizon_seconds":cfg.dt*cfg.horizon,
                                   "v4_risk_interval_seconds":cfg.dt*cfg.horizon,"conditioning":"internal belief distribution, not model correctness or whole trajectory"},
              "statistical_unit":"paired seed and scenario; scenario strata retained",
              "cost_unit":"declared simulation resource proxy, not Joules",
              "limitations":["synthetic RGB","simulated physics and radio","no trajectory or real-world safety guarantee",
                             "v4/v5 comparisons also change filtering, control constraints, and integration density; common-core v5 baselines isolate acquisition",
                             "resource proxies do not represent actual CPU seconds or energy; wall time is reported separately"]}
    if split in ("holdout","stress") and set(seeds)&set(manifest["development_seeds"]):
        raise ValueError("Final evaluation seeds must be unseen during development")
    manifest["protocol_hash"]=canonical_hash(manifest)
    write_json(path,manifest)
    return manifest


def verify_protocol(root, protocol):
    if code_manifest(root)!=protocol["source_hashes"]:
        raise ValueError("Source/config changed after protocol freeze; create a NEW protocol rather than reuse holdout")
    if file_hash(Path(root)/protocol["config"]["vision_artifact"])!=protocol["vision_sha256"]:
        raise ValueError("Vision artifact changed after protocol freeze")
    if protocol.get("resource_policy_sha256") is not None and file_hash(Path(root)/protocol["config"]["learned_resource_policy"])!=protocol["resource_policy_sha256"]:
        raise ValueError("Resource policy changed after protocol freeze")
    body={k:v for k,v in protocol.items() if k!="protocol_hash"}
    if canonical_hash(body)!=protocol["protocol_hash"]:
        raise ValueError("Protocol was modified")


_WORKER_VISION=None
_WORKER_POLICY=None


def _worker_init(root,cfg_dict):
    global _WORKER_VISION,_WORKER_POLICY
    cfg=ControllerConfig.from_dict(cfg_dict)
    _WORKER_VISION=NeuralVision(Path(root)/cfg.vision_artifact,threads=1)
    _WORKER_POLICY=load_resource_policy(root,cfg)


def _worker_episode(task):
    root,cfg_dict,seed,scenario,method=task
    return run_episode(seed,scenario,ControllerConfig.from_dict(cfg_dict),method,_WORKER_VISION,root=root,resource_policy=_WORKER_POLICY)


def run_suite(root, output, cfg, seeds, methods=METHODS, split="development", *, protocol_path=None,workers=1):
    root,output=Path(root),Path(output); output.mkdir(parents=True,exist_ok=True)
    seeds,methods=list(seeds),list(methods)
    if not seeds or not methods or len(set(seeds))!=len(seeds) or len(set(methods))!=len(methods):
        raise ValueError("Use nonempty unique seeds and methods")
    manifest=None
    if split in ("holdout","stress"):
        if protocol_path is None:
            raise ValueError("Holdout requires a frozen protocol")
        manifest=json.loads(Path(protocol_path).read_text())
        verify_protocol(root,manifest)
        if cfg.to_dict()!=manifest["config"] or list(seeds)!=manifest["seeds"] or list(methods)!=manifest["methods"] or split!=manifest["split"]:
            raise ValueError("Run does not match frozen protocol")
    vision=NeuralVision(root/cfg.vision_artifact,threads=1)
    policy=load_resource_policy(root,cfg)
    sources=code_manifest(root);run_code_hash=canonical_hash(sources)
    write_json(output/"run_manifest.json",{"created":utc_now(),"source_hashes":sources,
                "config":cfg.to_dict(),"seeds":list(seeds),"methods":list(methods),"split":split,
                "vision_sha256":file_hash(root/cfg.vision_artifact),"environment":environment_info(),"workers":workers,
                "resource_policy_sha256":None if policy is None else file_hash(root/cfg.learned_resource_policy),
                "protocol_hash":None if manifest is None else manifest["protocol_hash"]})
    rows=[]; started=time.perf_counter()
    # Every task is keyed independently. Results are consumed in declared
    # order; worker count changes timing, not the simulator or policy RNG.
    executor=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn"),initializer=_worker_init,initargs=(str(root),cfg.to_dict())) if workers>1 else None
    try:
      for scenario in scenarios(split):
        tasks=[(str(root),cfg.to_dict(),seed,scenario,method) for seed in seeds for method in methods
               if not (output/"episodes"/f"{scenario.name}_{seed}_{method}.json").exists()]
        completed=iter(executor.map(_worker_episode,tasks)) if executor else iter(run_episode(t[2],t[3],cfg,t[4],vision,root=root,resource_policy=policy) for t in tasks)
        for seed in seeds:
            for method in methods:
                filename=output/"episodes"/f"{scenario.name}_{seed}_{method}.json"
                if filename.exists():
                    result=json.loads(filename.read_text())
                    if result.get("config_hash")!=canonical_hash(cfg.to_dict()):
                        raise ValueError("Existing result has a different config")
                    if result.get("run_code_hash")!=run_code_hash:
                        raise ValueError("Existing result belongs to a different source version; use a new output folder")
                    if manifest is not None and result.get("protocol_hash")!=manifest["protocol_hash"]:
                        raise ValueError("Existing result belongs to a different protocol")
                else:
                    result=next(completed)
                    result["config_hash"]=canonical_hash(cfg.to_dict())
                    result["run_code_hash"]=run_code_hash
                    result["split"]=split
                    result["protocol_hash"]=None if manifest is None else manifest["protocol_hash"]
                    write_json(filename,result)
                rows.append(flatten_result(result))
        print(f"{split}: {scenario.name} finished ({len(rows)} episodes)",flush=True)
    finally:
        if executor:executor.shutdown()
    fields=list(rows[0])
    with (output/"episodes.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    report=summarize(rows)
    report.update({"split":split,"n_episodes":len(rows),"suite_wall_seconds":time.perf_counter()-started,
                   "protocol_hash":None if manifest is None else manifest["protocol_hash"],
                   "created":utc_now(),"budgets":cfg.budget.__dict__})
    write_json(output/"summary.json",report)
    return report


def flatten_result(r):
    return {"method":r["method"],"scenario":r["scenario"]["name"],"seed":r["seed"],
            "outcome":r["outcome"],"reason":r["reason"],"steps":r["steps"],
            "success":int(r["outcome"]=="success"),"collision":int(r["outcome"]=="collision"),
            "deferred":int(r["outcome"]=="deferred"),"wall_seconds":r["wall_seconds"],
            **r["resources"],"world_hash":r["world_hash"]}


def wilson(k,n,alpha=.05):
    from scipy.stats import norm
    z=norm.ppf(1-alpha/2);p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(max(0,center-half)),float(min(1,center+half))]


def paired_bootstrap(a,b,seed=912,n_boot=4000):
    # Rows are paired by (scenario,seed); repeated policy RNG seeds are not
    # treated as independent data. Resampling seeds preserves scenario clusters.
    keys=sorted(set(a)&set(b)); seeds=sorted({key[1] for key in keys})
    diffs=np.array([np.mean([a[k]-b[k] for k in keys if k[1]==s]) for s in seeds])
    rng=np.random.default_rng(seed)
    boot=np.mean(diffs[rng.integers(0,len(diffs),(n_boot,len(diffs)))],axis=1)
    return {"difference":float(np.mean(diffs)),"ci95":np.quantile(boot,[.025,.975]).tolist(),"n_seed_clusters":len(seeds)}


def summarize(rows):
    methods=sorted({r["method"] for r in rows}); aggregated={}; by_scenario={}
    for method in methods:
        sample=[r for r in rows if r["method"]==method];n=len(sample)
        aggregated[method]={"n":n}
        for key in ("success","collision","deferred"):
            total=sum(r[key] for r in sample)
            aggregated[method][key]=total/n
            aggregated[method][key+"_ci95_wilson"]=wilson(total,n)
        for key in ("cost","steps","wall_seconds","mc_samples","observations","queries","probes","logical_latency_ms","expert_queries"):
            aggregated[method][key+"_mean"]=math.fsum(r.get(key,0) for r in sample)/n
        successes=sum(r["success"] for r in sample)
        aggregated[method]["cost_per_success"]=math.fsum(r["cost"] for r in sample)/successes if successes else None
        by_scenario[method]={}
        for scenario in sorted({r["scenario"] for r in sample}):
            s=[r for r in sample if r["scenario"]==scenario]
            by_scenario[method][scenario]={k:math.fsum(r[k] for r in s)/len(s) for k in ("success","collision","deferred","cost")}
    comparisons={}
    if "telemachus" in methods:
        for baseline in methods:
            if baseline=="telemachus":continue
            comparisons[baseline]={}
            for endpoint in ("success","collision","cost"):
                a={(r["scenario"],r["seed"]):r[endpoint] for r in rows if r["method"]=="telemachus"}
                b={(r["scenario"],r["seed"]):r[endpoint] for r in rows if r["method"]==baseline}
                comparisons[baseline][endpoint]=paired_bootstrap(a,b)
            common={(r["scenario"],r["seed"]) for r in rows if r["method"]=="telemachus" and r["success"]}&{(r["scenario"],r["seed"]) for r in rows if r["method"]==baseline and r["success"]}
            if common:
                a={(r["scenario"],r["seed"]):r["cost"] for r in rows if r["method"]=="telemachus" and (r["scenario"],r["seed"]) in common}
                b={(r["scenario"],r["seed"]):r["cost"] for r in rows if r["method"]==baseline and (r["scenario"],r["seed"]) in common}
                comparisons[baseline]["cost_on_joint_success"]={**paired_bootstrap(a,b),"n_paired_episodes":len(common),"scope":"Selected subset where both methods succeed; not an unconditional efficiency estimand."}
    return {"methods":aggregated,"by_scenario":by_scenario,"paired_comparisons":comparisons,
            "interpretation":"Wilson intervals describe episode proportions; paired CI resamples seed clusters across scenarios. Exploratory, no SOTA or safety claim."}
