from __future__ import annotations

from dataclasses import replace
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


def run_episode(seed, scenario, cfg, method, vision,goal=None):
    world=NavigationWorld(seed,scenario,cfg,goal=goal)
    agent=Controller(SensorActuatorClient(world,vision),vision,cfg,method,seed=seed)
    result=agent.run()
    result.update(world.truth_record())
    return result


def freeze_protocol(root, path, cfg, seeds, methods, split="holdout"):
    root=Path(root)
    seeds,methods=list(seeds),list(methods)
    if not seeds or not methods or len(set(seeds))!=len(seeds) or len(set(methods))!=len(methods):
        raise ValueError("Use nonempty unique seeds and methods")
    manifest={"version":4,"created":utc_now(),"split":split,"seeds":list(seeds),
              "methods":list(methods),"scenarios":[s.__dict__ for s in scenarios(split)],
              "config":cfg.to_dict(),"source_hashes":code_manifest(root),
              "vision_sha256":file_hash(root/cfg.vision_artifact),"environment":environment_info(),
              "endpoints":["success","collision","deferred","cost","wall_seconds"],
              "primary_comparisons":["confidence_gate","budget_voi"],
              "statistical_unit":"paired seed and scenario; scenario strata retained",
              "cost_unit":"declared simulation resource proxy, not Joules",
              "limitations":["synthetic RGB","simulated physics and radio","no trajectory or real-world safety guarantee"]}
    manifest["protocol_hash"]=canonical_hash(manifest)
    write_json(path,manifest)
    return manifest


def verify_protocol(root, protocol):
    if code_manifest(root)!=protocol["source_hashes"]:
        raise ValueError("Source/config changed after protocol freeze; create a NEW protocol rather than reuse holdout")
    if file_hash(Path(root)/protocol["config"]["vision_artifact"])!=protocol["vision_sha256"]:
        raise ValueError("Vision artifact changed after protocol freeze")
    body={k:v for k,v in protocol.items() if k!="protocol_hash"}
    if canonical_hash(body)!=protocol["protocol_hash"]:
        raise ValueError("Protocol was modified")


def run_suite(root, output, cfg, seeds, methods=METHODS, split="development", *, protocol_path=None):
    root,output=Path(root),Path(output); output.mkdir(parents=True,exist_ok=True)
    seeds,methods=list(seeds),list(methods)
    if not seeds or not methods or len(set(seeds))!=len(seeds) or len(set(methods))!=len(methods):
        raise ValueError("Use nonempty unique seeds and methods")
    manifest=None
    if split=="holdout":
        if protocol_path is None:
            raise ValueError("Holdout requires a frozen protocol")
        manifest=json.loads(Path(protocol_path).read_text())
        verify_protocol(root,manifest)
        if cfg.to_dict()!=manifest["config"] or list(seeds)!=manifest["seeds"] or list(methods)!=manifest["methods"]:
            raise ValueError("Run does not match frozen protocol")
    vision=NeuralVision(root/cfg.vision_artifact)
    sources=code_manifest(root);run_code_hash=canonical_hash(sources)
    write_json(output/"run_manifest.json",{"created":utc_now(),"source_hashes":sources,
                "config":cfg.to_dict(),"seeds":list(seeds),"methods":list(methods),"split":split,
                "vision_sha256":file_hash(root/cfg.vision_artifact),"environment":environment_info(),
                "protocol_hash":None if manifest is None else manifest["protocol_hash"]})
    rows=[]; started=time.perf_counter()
    # Stable append-only episode files permit recovery without resampling.
    for scenario in scenarios(split):
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
                    result=run_episode(seed,scenario,cfg,method,vision)
                    result["config_hash"]=canonical_hash(cfg.to_dict())
                    result["run_code_hash"]=run_code_hash
                    result["split"]=split
                    result["protocol_hash"]=None if manifest is None else manifest["protocol_hash"]
                    write_json(filename,result)
                rows.append(flatten_result(result))
        print(f"{split}: {scenario.name} finished ({len(rows)} episodes)",flush=True)
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
        for key in ("cost","steps","wall_seconds","mc_samples","observations","queries","probes","logical_latency_ms"):
            aggregated[method][key+"_mean"]=float(np.mean([r[key] for r in sample]))
        by_scenario[method]={}
        for scenario in sorted({r["scenario"] for r in sample}):
            s=[r for r in sample if r["scenario"]==scenario]
            by_scenario[method][scenario]={k:float(np.mean([r[k] for r in s])) for k in ("success","collision","deferred","cost")}
    comparisons={}
    if "telemachus" in methods:
        for baseline in methods:
            if baseline=="telemachus":continue
            comparisons[baseline]={}
            for endpoint in ("success","collision","cost"):
                a={(r["scenario"],r["seed"]):r[endpoint] for r in rows if r["method"]=="telemachus"}
                b={(r["scenario"],r["seed"]):r[endpoint] for r in rows if r["method"]==baseline}
                comparisons[baseline][endpoint]=paired_bootstrap(a,b)
    return {"methods":aggregated,"by_scenario":by_scenario,"paired_comparisons":comparisons,
            "interpretation":"Wilson intervals describe episode proportions; paired CI resamples seed clusters across scenarios. Exploratory, no SOTA or safety claim."}
