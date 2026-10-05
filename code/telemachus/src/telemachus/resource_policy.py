from __future__ import annotations

import copy
import json
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np
from .belief import FEATURE_NAMES
from .budget import BudgetExceeded
from .provenance import write_json,keyed_rng,file_hash,utc_now,code_manifest


ACTIONS=("ACT","SIMULATE","OBSERVE","ASK","PROBE")
POLICY_FEATURES=tuple(FEATURE_NAMES)+("mc_fraction","empirical_min_risk","empirical_max_support",
    "risk_ucb_min","support_lcb_max","certified_action","selected_speed","score_margin",
    "distance_to_goal","path_length","remaining_cost_fraction","mission_slack","queries_failed",
    "ticks_since_focus","ticks_since_probe","gain_updates","stagnation","look_fraction",
    "last_simulation","last_observation","last_query","epistemic_attempts",
    "empirical_joint_acceptable_fraction","best_joint_margin","queries_this_tick","observations_this_tick",
    "safety_margin")
# v3 adds the calibrated safety margin as an input so one policy serves every
# setting that episode-level safety calibration may select.
POLICY_SCHEMA="resource-counterfactual-v3"


def policy_features(agent,decision):
    s,c=decision["stats"],decision["cert"];idx=decision["action"]
    ordered=np.sort(s.mean_scores)
    remaining=agent.cfg.budget.max_cost-agent.ledger.cost
    extra=[s.n_samples/agent.cfg.mc_looks[-1],float(np.min(s.risk_counts/s.n_samples)),
       float(np.max(s.support_counts/s.n_samples)),float(np.min(c.risk_ucb)),float(np.max(c.support_lcb)),
       float(idx is not None),0. if idx is None else decision["controls"][idx,0],float(ordered[-1]-ordered[-2]),
       float(np.linalg.norm(agent.pose.mean[:2]-agent.client.goal)),decision["path_length"],
       remaining/agent.cfg.budget.max_cost,(remaining-agent.mission_reserve(decision))/agent.cfg.budget.max_cost,
       min(agent.failed_queries,5),min(agent.tick-agent.last_focused_tick,100)/100,
       min(agent.tick-agent.last_probed_tick,100)/100,min(agent.dynamics.n_updates,50)/50,
       min(agent.stagnation,20)/20,decision["look"]/max(1,len(agent.cfg.mc_looks)-1),
       int(decision["last_resource"]=="SIMULATE"),int(decision["last_resource"]=="OBSERVE"),
       int(decision["last_resource"]=="ASK"),decision["resource_attempts"],
       float(np.mean((s.risk_counts/s.n_samples<agent.cfg.risk_threshold)&(s.support_counts/s.n_samples>=agent.cfg.stability_threshold))),
       float(np.max(np.minimum(agent.cfg.risk_threshold-s.risk_counts/s.n_samples,s.support_counts/s.n_samples-agent.cfg.stability_threshold))),
       decision["resource_history"].count("ASK"),decision["resource_history"].count("OBSERVE"),
       agent.cfg.safety_margin]
    x=np.concatenate([decision["features"],np.asarray(extra,float)])
    if len(x)!=len(POLICY_FEATURES) or not np.isfinite(x).all():raise ValueError("Invalid observable policy features")
    return x


class ResourcePolicy:
    """Learned value of acquiring evidence, with a hard motion certificate.

    Trees predict measured counterfactual acquisition outcomes. Scenario names,
    fault parameters, true poses and obstacle maps are absent from runtime
    features. JSON trees avoid loading executable pickle models.
    """
    def __init__(self,checkpoint):
        data=json.loads(Path(checkpoint).read_text())
        if data["schema"]==POLICY_SCHEMA and tuple(data["features"])==POLICY_FEATURES:
            self.n_features=len(POLICY_FEATURES)
        elif data["schema"]=="resource-counterfactual-v2" and tuple(data["features"])==POLICY_FEATURES[:-1]:
            # Delivered v5 checkpoint: identical features without safety_margin.
            self.n_features=len(POLICY_FEATURES)-1
        else:
            raise ValueError("Policy checkpoint schema mismatch")
        self.schema=data["schema"]
        self.models=data["models"];self.blind_models=data["blind_models"]
        self.calibrators=data["calibrators"];self.blind_calibrators=data["blind_calibrators"]

    @staticmethod
    def predict_trees(model,x):
        output=[]
        for tree in model["trees"]:
            node=0
            while tree["left"][node]>=0:
                f=tree["feature"][node]
                node=tree["left"][node] if x[f]<=tree["threshold"][node] else tree["right"][node]
            output.append(tree["values"][node])
        return np.mean(output,axis=0)

    def predict(self,x,blind=False):
        x=np.asarray(x,float).copy()
        if x.shape!=(len(POLICY_FEATURES),) or not np.isfinite(x).all():raise ValueError("Invalid policy input")
        x=x[:self.n_features]
        if blind:x[:len(FEATURE_NAMES)]=0.
        models=self.blind_models if blind else self.models
        calibrators=self.blind_calibrators if blind else self.calibrators
        predicted={}
        for name in ACTIONS:
            p=np.asarray(self.predict_trees(models[name],x),float)
            cal=calibrators[name]
            p[0]=float(np.interp(p[0],cal["x"],cal["y"]))
            if len(p)>=5:
                # v3: multi-step lookahead targets, separately calibrated.
                calk=cal.get("lookahead",cal)
                p[3]=float(np.interp(p[3],calk["x"],calk["y"]))
                resolve_k,cost_k=p[3],p[4]
            else:
                resolve_k,cost_k=p[0],p[2]
            predicted[name]={"resolution_probability":float(np.clip(p[0],0,1)),
                             "gain_uncertainty_reduction":float(np.clip(p[1],0,1)),
                             "cost":float(max(0,p[2])),
                             "lookahead_resolution_probability":float(np.clip(resolve_k,0,1)),
                             "lookahead_cost":float(max(0,cost_k))}
        return predicted

    def choose(self,agent,decision,x,blind=False,reserve=True):
        predictions=self.predict(x,blind)
        decision["resource_predictions"]=predictions
        decision["policy_features"]=x.tolist()
        if decision["action"] is not None:
            # Accurate probes are bought only when expected future reduction
            # in model uncertainty is worth their cost and leaves a reserve.
            probe=predictions["PROBE"]
            steps=decision["path_length"]/(.8*max(.35,agent.dynamics.mean)*agent.cfg.dt)
            benefit=probe["gain_uncertainty_reduction"]*steps*.055
            if (agent.dynamics.std>.11 and agent.tick-agent.last_probed_tick>=18
                    and probe["resolution_probability"]>=.8 and benefit>agent.cfg.budget.probe_cost
                    and agent.resource_feasible("PROBE",decision,reserve=reserve)):
                return "PROBE"
            # A learned warning can request evidence, never waive a failed
            # certificate. Acquisition is optional when budget reserve is low.
            if predictions["ACT"]["resolution_probability"]<.93 and agent.tick-agent.last_focused_tick>=2:
                if agent.resource_feasible("OBSERVE",decision,reserve=reserve):return "OBSERVE"
            return "ACT"
        values={}
        for name in ("SIMULATE","OBSERVE","ASK"):
            if not agent.resource_feasible(name,decision):continue
            if name=="ASK" and agent.failed_queries>=1:continue
            if name=="ASK" and "ASK" in decision["resource_history"]:continue
            p=predictions[name]
            charge=agent.resource_cost(name,decision)
            cost=charge.cost+(0. if name=="SIMULATE" else agent.cfg.mc_looks[0]*agent.cfg.budget.sample_cost)
            if self.schema==POLICY_SCHEMA:
                # Value of information as resolution probability per unit of
                # expected cost until a certified action exists (v3 targets).
                cost=max(cost,p["lookahead_cost"])
                score=(p["lookahead_resolution_probability"]+.02)/(cost+.05)
            else:
                score=(p["resolution_probability"]+.02)/(cost+.05)
            if reserve and name!="SIMULATE" and not agent.resource_feasible(name,decision,reserve=True):score*=.55
            if name==decision["last_resource"] and name!="SIMULATE":score*=.45
            values[name]=score
        return max(values,key=values.get) if values else "DEFER"


def counterfactual_outcomes(agent,decision):
    """Training-only interventions on copies; observed data are the inputs.

    Physical contact is an offline reward label after applying an action to a
    copied world, not an input or an online oracle. Original episode RNG, radio,
    state and ledger are untouched.

    Targets per action (schema v3):
      [0] resolved after this action alone and the certified move was safe
      [1] relative reduction of dynamics-gain uncertainty
      [2] cost of this action
      [3] resolved and safe within LOOKAHEAD further heuristic acquisitions
      [4] total cost until resolution (or until the lookahead gives up)
    """
    results={}
    for action in ACTIONS:
        if action=="ACT" and decision["action"] is None:continue
        if action!="ACT" and not agent.resource_feasible(action,decision):continue
        cloned=copy.deepcopy(agent,memo={id(agent.vision):agent.vision})
        cloned.dataset_callback=None
        d=copy.deepcopy(decision)
        initial_cost=cloned.ledger.cost;initial_std=cloned.dynamics.std
        gain=lambda:max(0,1-cloned.dynamics.std/max(initial_std,.001))
        try:
            if action in ("OBSERVE","ASK","SIMULATE"):
                d=cloned.acquire(action,d)
            elif action=="PROBE":
                cloned._move(d,probe=True)
                if cloned.reason!="running":
                    ok=float(cloned.outcome=="success");spent=cloned.ledger.cost-initial_cost
                    results[action]=[ok,gain(),spent,ok,spent]
                    continue
                cloned.begin_tick();d=cloned.initial_compute(cloned.prepare_decision())
            one_step_cost=cloned.ledger.cost-initial_cost
            resolved_now=d["action"] is not None
            for _ in range(LOOKAHEAD if not resolved_now else 0):
                route=cloned.choose_resource(d)
                if route not in ("SIMULATE","OBSERVE","ASK"):break
                d=cloned.acquire(route,d)
                if d["action"] is not None:break
            cost_k=cloned.ledger.cost-initial_cost
            safe=0.
            if d["action"] is not None:
                motion=cloned._move(d)
                safe=float(not motion.collision)
            results[action]=[safe if resolved_now else 0.,gain(),one_step_cost,safe,cost_k]
        except BudgetExceeded:
            spent=cloned.ledger.cost-initial_cost
            results[action]=[0.,0.,spent,0.,spent]
    return results


# Further acquisitions explored by the counterfactual lookahead target.
LOOKAHEAD=2


_COLLECT_VISION=None


def _collection_init(root,cfg_dict):
    global _COLLECT_VISION
    from .vision import NeuralVision
    _COLLECT_VISION=NeuralVision(Path(root)/cfg_dict["vision_artifact"],threads=1)


def _collection_episode(task):
    from .simulation import NavigationWorld,SensorActuatorClient,scenarios
    from .controller import Controller
    from .config import ControllerConfig
    seed,scenario,cfg_dict,max_states=task;cfg=ControllerConfig.from_dict(cfg_dict)
    rows=[];world=NavigationWorld(seed,scenario,cfg)
    agent=Controller(SensorActuatorClient(world,_COLLECT_VISION),_COLLECT_VISION,cfg,"no_learning",seed=seed)
    def callback(a,d):
        if len(rows)>=max_states:return
        rng=keyed_rng(seed,a.decisions,803)
        if d["action"] is not None and rng.random()>.35:return
        x=policy_features(a,d);targets=counterfactual_outcomes(a,d)
        rows.append({"seed":seed,"scenario":scenario.name,"tick":a.tick,"decision":a.decisions,
                     "features":x.tolist(),"targets":targets})
    agent.dataset_callback=callback;result=agent.run()
    return rows,{"seed":seed,"scenario":scenario.__dict__,"outcome":result["outcome"],"states":len(rows),
                 "safety_margin":cfg.safety_margin}


TRAINING_MARGINS=(0.,.02,.04,.06,.08)


def training_scenarios(seed,randomize=True):
    from .simulation import scenarios
    from dataclasses import replace
    base=scenarios()
    if not randomize:return base
    rng=keyed_rng(seed,0,804);out=[]
    for scenario in base:
        changes={"camera_style":int(rng.integers(0,2)),"layout_family":int(rng.integers(0,2))}
        if scenario.name in ("visual","compound"):
            changes.update(camera_noise=float(rng.uniform(.12,.34)),camera_brightness=float(rng.uniform(.25,.65)),camera_occlusion=float(rng.uniform(.10,.4)))
        if scenario.name in ("localization","compound"):changes["localization_sigma"]=float(rng.uniform(.10,.30))
        if scenario.name in ("dynamics","compound"):
            changes.update(speed_gain=float(rng.uniform(.45,.88)),onset=int(rng.integers(10,24)))
        if scenario.name in ("radio","compound"):changes["channel_snr_db"]=float(rng.uniform(-4.,3.))
        out.append(replace(scenario,**changes))
    return out


def collect_dataset(root,cfg,seeds,output,split="development",max_states_per_episode=64,workers=1,randomize=True):
    if split!="development":raise ValueError("Policy collection must use development worlds")
    root=Path(root);seeds=list(seeds);rows=[];episodes=[];wall=time.perf_counter()
    sources=code_manifest(root);cfg_dict=cfg.to_dict()
    tasks=[]
    for seed in seeds:
        for i,s in enumerate(training_scenarios(seed,randomize)):
            task_cfg=dict(cfg_dict)
            if randomize:
                # Safety margins that episode-level calibration may select.
                task_cfg["safety_margin"]=float(keyed_rng(seed,i,805).choice(TRAINING_MARGINS))
            tasks.append((seed,s,task_cfg,max_states_per_episode))
    executor=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn"),initializer=_collection_init,initargs=(str(root),cfg_dict)) if workers>1 else None
    if executor:completed=executor.map(_collection_episode,tasks)
    else:
        _collection_init(root,cfg_dict);completed=map(_collection_episode,tasks)
    try:
        for i,(r,e) in enumerate(completed):
            rows.extend(r);episodes.append(e)
            if (i+1)%24==0:print(f"counterfactual collection: {i+1}/{len(tasks)} episodes, {len(rows)} states",flush=True)
    finally:
        if executor:executor.shutdown()
    report={"schema":POLICY_SCHEMA,"features":list(POLICY_FEATURES),"rows":rows,
            "config":cfg_dict,"source_hashes":sources,"vision_sha256":file_hash(root/cfg.vision_artifact),
            "domain_randomized":randomize,"certified_state_sampling_probability":.35,"max_states_per_episode":max_states_per_episode,
            "episodes":episodes,"seeds":list(seeds),"wall_seconds":time.perf_counter()-wall,"created":utc_now(),
            "truth_usage":"Only offline physical contact/arrival reward labels; no truth variables in policy features."}
    write_json(output,report)
    return report


def export_forest(model):
    trees=[]
    for estimator in model.estimators_:
        t=estimator.tree_
        trees.append({"left":t.children_left.tolist(),"right":t.children_right.tolist(),
                      "feature":t.feature.tolist(),"threshold":t.threshold.tolist(),
                      "values":t.value[:,:,0].tolist()})
    return {"trees":trees}


def train_policy(dataset,output,train_seeds,calibration_seeds,test_seeds):
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss,roc_auc_score
    dataset=Path(dataset);data=json.loads(dataset.read_text());sets=[set(s) for s in (train_seeds,calibration_seeds,test_seeds)]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)):raise ValueError("Seed groups must be disjoint")
    models,blind_models,calibrators,blind_calibrators={}, {}, {}, {};metrics={}
    for blind in (False,True):
        for action in ACTIONS:
            split=[]
            for seeds in sets:
                rows=[r for r in data["rows"] if r["seed"] in seeds and action in r["targets"]]
                x=np.array([r["features"] for r in rows]);y=np.array([r["targets"][action] for r in rows])
                if not len(x):raise ValueError(f"No {action} samples in split")
                if blind:x[:,:len(FEATURE_NAMES)]=0.
                split.append((x,y))
            model=ExtraTreesRegressor(n_estimators=48,max_depth=10,min_samples_leaf=8,
                                      max_features=.8,random_state=831,n_jobs=2)
            model.fit(*split[0])
            cal_prediction=model.predict(split[1][0])[:,0]
            if len(np.unique(split[1][1][:,0]))>1:
                cal=IsotonicRegression(y_min=.01,y_max=.99,out_of_bounds="clip").fit(cal_prediction,split[1][1][:,0])
                knots={"x":cal.X_thresholds_.tolist(),"y":cal.y_thresholds_.tolist()}
            else:
                knots={"x":[0.,1.],"y":[float(split[1][1][:,0].mean())]*2}
            pred=np.interp(model.predict(split[2][0])[:,0],knots["x"],knots["y"])
            key=("blind_" if blind else "")+action
            metrics[key]={"n_train":len(split[0][0]),"n_calibration":len(split[1][0]),"n_test":len(split[2][0]),
                "test_positive_rate":float(split[2][1][:,0].mean()),"brier":float(brier_score_loss(split[2][1][:,0],pred))}
            if len(np.unique(split[2][1][:,0]))>1:metrics[key]["auc"]=float(roc_auc_score(split[2][1][:,0],pred))
            if split[0][1].shape[1]>=5:
                # Separate isotonic calibration of the lookahead resolution target.
                cal_k=model.predict(split[1][0])[:,3];yk=split[1][1][:,3]
                if len(np.unique(yk))>1:
                    ck=IsotonicRegression(y_min=.01,y_max=.99,out_of_bounds="clip").fit(cal_k,yk)
                    knots["lookahead"]={"x":ck.X_thresholds_.tolist(),"y":ck.y_thresholds_.tolist()}
                else:
                    knots["lookahead"]={"x":[0.,1.],"y":[float(yk.mean())]*2}
                pk=np.interp(model.predict(split[2][0])[:,3],knots["lookahead"]["x"],knots["lookahead"]["y"])
                yt=split[2][1][:,3]
                metrics[key]["lookahead_positive_rate"]=float(yt.mean())
                metrics[key]["lookahead_brier"]=float(brier_score_loss(yt,pk))
                if len(np.unique(yt))>1:metrics[key]["lookahead_auc"]=float(roc_auc_score(yt,pk))
            (blind_models if blind else models)[action]=export_forest(model)
            (blind_calibrators if blind else calibrators)[action]=knots
    checkpoint={"schema":POLICY_SCHEMA,"features":list(POLICY_FEATURES),"actions":list(ACTIONS),
                "models":models,"blind_models":blind_models,"calibrators":calibrators,"blind_calibrators":blind_calibrators,
                "data_sha256":file_hash(dataset),"splits":{"train":sorted(sets[0]),"calibration":sorted(sets[1]),"test":sorted(sets[2])}}
    write_json(output,checkpoint)
    report={"created":utc_now(),"metrics":metrics,"data_sha256":file_hash(dataset),"splits":checkpoint["splits"],
            "model":"ExtraTrees multi-output acquisition resolution/cost/gain predictions with held-seed isotonic calibration",
            "limitations":"Calibration is on synthetic development interventions; no real-world or OOD reliability guarantee."}
    # v5 kept the fixed name resource_policy_training.json for resource_policy.json.
    name="resource_policy_training.json" if Path(output).name=="resource_policy.json" else Path(output).stem+"_training.json"
    write_json(Path(output).with_name(name),report)
    return report
