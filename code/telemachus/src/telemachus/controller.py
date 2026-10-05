from __future__ import annotations

from collections import Counter
import time
import numpy as np
from .config import ControllerConfig
from .budget import ResourceLedger,BudgetExceeded
from .physics import PoseFilter,DynamicsIdentifier,TrajectoryIdentifier
from .belief import MapBelief,ObservableDiagnosis,observed_features,FEATURE_NAMES
from .radio import ChannelBelief
from .simulation import NavigationWorld
from .planning import astar,path_waypoint,candidate_controls,sample_worlds,choose_certified
from .provenance import keyed_rng


METHODS=("legacy_v4","nominal","confidence_gate","budget_voi","telemachus",
         "no_learning","no_reserve","no_source","no_model_update","fixed_compute","no_expert")
# Methods that may escalate to the (budgeted) expert. Prior-art baselines and
# the no_expert ablation never do, which isolates the value of asking for help.
EXPERT_METHODS=("telemachus","no_learning","no_reserve","no_source","no_model_update","fixed_compute")
LEARNED_METHODS=("telemachus","no_reserve","no_source","no_model_update","fixed_compute","no_expert")


class Controller:
    """Measured-state navigation with a separate resource acquisition policy.

    Resource prediction never authorizes motion. Only the simultaneous Monte
    Carlo certificate permits an ACT or physical PROBE. Collision risk covers
    the declared forecast interval, including the actual executed control.
    This is not a full-trajectory or real-world safety certificate.
    """
    def __init__(self,client,vision,config:ControllerConfig,method="telemachus",seed=0,resource_policy=None):
        if method not in METHODS or method=="legacy_v4":raise ValueError("Invalid v5 method")
        self.client,self.vision,self.cfg,self.method,self.seed=client,vision,config,method,int(seed)
        self.ledger=ResourceLedger(config.budget)
        self.map=MapBelief(int(round(config.world_size/config.cell_size)),config.cell_size,config.pose_georeferencing)
        self.dynamics=TrajectoryIdentifier() if config.trajectory_identification else DynamicsIdentifier()
        self.channel,self.diagnosis=ChannelBelief(),ObservableDiagnosis()
        self.resource_policy=resource_policy
        if config.resource_policy_mode in ("learned","blind") and method in LEARNED_METHODS and resource_policy is None:
            raise ValueError("A learned policy checkpoint is required; no silent heuristic fallback")
        self.pose=None;self.tick=0;self.decisions=0
        self.outcome,self.reason="deferred","running"
        self.trace,self.feature_rows=[],[]
        self.last_probed_tick,self.last_focused_tick=-100,-100
        self.failed_queries=0;self.previous_gps=None;self.previous_control=None;self.previous_heading=None
        self.last_distance=None;self.stagnation=0
        self.dataset_callback=None
        self.latest_measured_pose=None

    @property
    def adaptive_model(self):return self.cfg.adapt_dynamics and self.method!="no_model_update"

    def _update_reading(self,reading):
        mean=np.mean(reading.gps_burst,axis=0)
        self.latest_measured_pose=mean.copy()
        if self.pose is None:self.pose=PoseFilter(mean)
        self.pose.update(reading.gps_burst,focused=reading.focus)
        self.map.update(reading,self.vision,self.pose.mean)
        if reading.focus:self.last_focused_tick=reading.tick
        return mean

    def _sense(self,focused=False):
        name="OBSERVE" if focused else "PASSIVE"
        self.ledger.spend(name,self.ledger.charge_for(name))
        reading=self.client.sense(focused=focused);self._update_reading(reading)
        return reading

    def _ask(self):
        nbytes,latency=self.client.ask_reservation()
        self.ledger.spend("ASK",self.ledger.charge_for("ASK",radio_latency_ms=latency,n_bytes=nbytes))
        packet=self.client.ask();self.ledger.reconcile_radio(packet.bytes_sent,packet.logical_latency_ms)
        self.channel.update(packet);self.failed_queries=0 if packet.delivered else self.failed_queries+1
        evidence=NavigationWorld.decode_packet(packet,self.tick)
        if evidence is not None:
            self.map.update_remote(*evidence,self.pose.mean);self.last_focused_tick=self.tick
        return packet.delivered

    def _expert(self):
        self.ledger.spend("EXPERT",self.ledger.charge_for("EXPERT"))
        burst,centers,probabilities,variance=self.client.ask_expert()
        self.pose.update(burst,focused=True)
        self.latest_measured_pose=np.mean(burst,axis=0)
        self.map.update_remote(centers,probabilities,variance,self.pose.mean)
        self.last_focused_tick=self.tick

    @property
    def expert_enabled(self):
        return self.method in EXPERT_METHODS and self.cfg.budget.max_expert_queries>0

    def _identify(self,before,after,heading,control,weight=1.,variance=.002):
        if not self.adaptive_model:return
        v=control[0]*np.sinc(control[1]*self.cfg.dt/(2*np.pi))
        kwargs={"measurement_variance":variance} if isinstance(self.dynamics,TrajectoryIdentifier) else {}
        self.dynamics.update(before[:2],after[:2],heading+control[1]*self.cfg.dt/2,v,self.cfg.dt,weight=weight,**kwargs)

    def begin_tick(self):
        reading=self._sense();mean=np.mean(reading.gps_burst,axis=0)
        if self.previous_control is not None:
            variance=float(np.mean(np.var(reading.gps_burst[:,:2],axis=0,ddof=1))/len(reading.gps_burst))
            self._identify(self.previous_gps,mean,self.previous_heading,self.previous_control,variance=variance)
            self.previous_control=None
        distance=float(np.linalg.norm(self.pose.mean[:2]-self.client.goal))
        if self.last_distance is not None and distance>=self.last_distance-.015:self.stagnation+=1
        else:self.stagnation=0
        self.last_distance=distance
        return reading

    def prepare_decision(self):
        self.decisions+=1
        features=observed_features(self.map,self.pose,self.dynamics,self.channel)
        scores=self.diagnosis.predict(features)
        self.feature_rows.append({"tick":self.tick,"features":features.tolist(),"scores":scores})
        path=astar(self.map.probability,self.pose.mean[:2],self.client.goal,self.cfg.cell_size)
        waypoint=path_waypoint(path,self.client.goal,self.cfg.cell_size) if path else self.client.goal.copy()
        path_length=sum(float(np.hypot(*(np.subtract(b,a))))*self.cfg.cell_size for a,b in zip(path,path[1:])) if path else float(np.linalg.norm(self.pose.mean[:2]-self.client.goal))*1.4
        return {"features":features,"scores":scores,"path_length":path_length,"waypoint":waypoint,
                "controls":candidate_controls(self.pose.mean,waypoint,self.cfg.turn_before_move),"stats":None,"cert":None,"action":None,
                "look":-1,"batches":0,"resource_attempts":0,"last_resource":None,"resource_history":[]}

    def _sample(self,decision,n):
        self.ledger.spend("SIMULATE",self.ledger.charge_for("SIMULATE",n_samples=n))
        stats=sample_worlds(self.map.probability,self.pose.mean,self.pose.cov,decision["controls"],decision["waypoint"],
             self.dynamics.mean if self.adaptive_model else 1.,self.dynamics.std if self.adaptive_model else .08,
             keyed_rng(self.seed,self.decisions,301,decision["batches"]),n,self.cfg.dt,self.cfg.horizon,self.cfg.cell_size,
             self.cfg.risk_horizon,self.cfg.collision_substeps,self.cfg.stability_regret,.12+self.cfg.safety_margin)
        decision["batches"]+=1
        decision["stats"]=stats if decision["stats"] is None else decision["stats"].merge(stats)
        decision["action"],decision["cert"]=choose_certified(decision["stats"],self.cfg)
        return decision

    def initial_compute(self,decision):
        start=1 if self.method=="fixed_compute" and len(self.cfg.mc_looks)>1 else 0
        self._sample(decision,self.cfg.mc_looks[start]);decision["look"]=start
        return decision

    def resource_cost(self,name,decision):
        if name=="SIMULATE":
            look=decision["look"]+1
            if look>=len(self.cfg.mc_looks):return None
            return self.ledger.charge_for(name,n_samples=self.cfg.mc_looks[look]-decision["stats"].n_samples)
        if name=="ASK":
            nbytes,latency=self.client.ask_reservation()
            return self.ledger.charge_for(name,radio_latency_ms=latency,n_bytes=nbytes)
        return self.ledger.charge_for(name)

    def mission_reserve(self,decision):
        speed=.8*max(.35,self.dynamics.mean-1.25*self.dynamics.std)
        expected_steps=decision["path_length"]/max(speed*self.cfg.dt,.04)
        minimum=self.cfg.budget.passive_cost+self.cfg.mc_looks[0]*self.cfg.budget.sample_cost
        return self.cfg.reserve_fraction*expected_steps*minimum

    def resource_feasible(self,name,decision,reserve=False):
        charge=self.resource_cost(name,decision)
        if charge is None or not self.ledger.can_afford(charge):return False
        if name=="PROBE" and (decision["action"] is None or decision["controls"][decision["action"],0]<.2):return False
        if reserve and name!="SIMULATE":
            return self.cfg.budget.max_cost-self.ledger.cost-charge.cost-self.cfg.mc_looks[0]*self.cfg.budget.sample_cost>=self.mission_reserve(decision)
        return True

    def choose_resource(self,decision):
        route=self._choose_resource(decision)
        # Escalation: ask the expert only as a last resort, i.e. when the
        # cheap sources are exhausted for this decision and the agent would
        # otherwise stop. The expert never authorizes motion: the certificate
        # is recomputed from the updated belief.
        if (self.expert_enabled and decision["action"] is None and "EXPERT" not in decision["resource_history"]
                and (route=="DEFER" or (route in ("OBSERVE","ASK") and decision["resource_attempts"]>=self.cfg.max_epistemic_streak-1))
                and self.resource_feasible("EXPERT",decision)):
            return "EXPERT"
        return route

    def _choose_resource(self,decision):
        action,scores=decision["action"],decision["scores"]
        if self.method=="confidence_gate":
            if action is not None:return "ACT"
            if self.resource_feasible("SIMULATE",decision):return "SIMULATE"
            return "OBSERVE" if self.resource_feasible("OBSERVE",decision) else "DEFER"
        if self.method=="budget_voi" or (self.method=="no_source" and self.resource_policy is None):
            if action is not None:return "ACT"
            if self.resource_feasible("SIMULATE",decision):return "SIMULATE"
            q=self.cfg.budget.query_cost/max(self.channel.delivery_probability,.05)
            name="ASK" if q<self.cfg.budget.observe_cost and self.failed_queries<1 and "ASK" not in decision["resource_history"] else "OBSERVE"
            return name if self.resource_feasible(name,decision) else "DEFER"
        if self.resource_policy is not None and self.method!="no_learning":
            from .resource_policy import policy_features
            return self.resource_policy.choose(self,decision,policy_features(self,decision),
                      blind=self.method=="no_source" or self.cfg.resource_policy_mode=="blind",reserve=self.method!="no_reserve")
        if action is not None:
            if (self.dynamics.std>.12 and self.pose.noise_evidence<.01 and self.dynamics.n_updates>=5
                    and self.tick-self.last_probed_tick>=18 and self.resource_feasible("PROBE",decision,reserve=True)):
                return "PROBE"
            return "ACT"
        stats=decision["stats"]
        plausible=(stats.risk_counts/stats.n_samples<self.cfg.risk_threshold)&(stats.support_counts/stats.n_samples>=self.cfg.stability_threshold)
        if np.any(plausible) and self.resource_feasible("SIMULATE",decision):return "SIMULATE"
        source="ASK" if scores["vision"]>scores["localization"] and self.failed_queries==0 else "OBSERVE"
        if source=="ASK" and "ASK" in decision["resource_history"]:source="OBSERVE"
        if decision["last_resource"]==source:source="OBSERVE"
        if self.resource_feasible(source,decision):return source
        other="OBSERVE" if source=="ASK" else "ASK"
        return other if self.resource_feasible(other,decision) else "DEFER"

    def acquire(self,name,decision):
        if name=="SIMULATE":
            look=decision["look"]+1
            self._sample(decision,self.cfg.mc_looks[look]-decision["stats"].n_samples)
            decision["look"]=look;decision["last_resource"]=name
            return decision
        if name=="OBSERVE":self._sense(focused=True)
        elif name=="ASK":self._ask()
        elif name=="EXPERT":self._expert()
        else:raise ValueError("Invalid acquisition")
        attempts=decision["resource_attempts"]+1
        new=self.prepare_decision();new["resource_attempts"]=attempts;new["last_resource"]=name
        new["resource_history"]=decision["resource_history"]+[name]
        return self.initial_compute(new)

    def recovery_action(self,decision):
        """Best action whose collision-risk bound alone is certified, or None.

        Near-optimality (support) is a progress criterion, not a safety one.
        Recovery keeps the same simultaneous risk bound and only gives up the
        support requirement, so the per-decision safety statement is unchanged.
        """
        if not self.cfg.safe_recovery or decision["stats"] is None:return None
        safe=decision["cert"].risk_ucb<=self.cfg.risk_threshold
        if not np.any(safe):return None
        return int(np.argmax(np.where(safe,decision["stats"].mean_scores,-np.inf)))

    def _move(self,decision,probe=False,recovery=False):
        action=decision["action"]
        if action is None:raise ValueError("Uncertified action")
        if recovery:
            if probe or decision["cert"].risk_ucb[action]>self.cfg.risk_threshold:
                raise ValueError("Recovery must satisfy the certified risk bound")
        elif not decision["cert"].acceptable(self.cfg.stability_threshold,self.cfg.risk_threshold)[action]:
            raise ValueError("Resource policy attempted to bypass the certificate")
        control=decision["controls"][action]
        if probe:
            self.ledger.spend("PROBE",self.ledger.charge_for("PROBE"))
            pre=self.client.sense(focused=True);before=np.mean(pre.gps_burst,axis=0)
            motion=self.client.act(control)
            post=self.client.sense(focused=True);after=np.mean(post.gps_burst,axis=0)
            self._identify(before,after,before[2],control,weight=8.,variance=.0001)
            self.pose.predict(control,self.cfg.dt,self.dynamics.mean if self.adaptive_model else 1.,self.dynamics.std)
            self._update_reading(post)
            self.last_probed_tick=self.tick;self.previous_control=None
        else:
            self.previous_gps=self.latest_measured_pose.copy()
            self.previous_heading=float(self.pose.mean[2]);self.previous_control=control.copy()
            motion=self.client.act(control)
            self.pose.predict(control,self.cfg.dt,self.dynamics.mean if self.adaptive_model else 1.,self.dynamics.std if self.adaptive_model else .08)
        self.tick=motion.tick
        if motion.collision:self.outcome,self.reason="collision","physical_contact"
        elif motion.reached_goal:self.outcome,self.reason="success","physical_arrival"
        return motion

    def _log(self,decision,route):
        cert,action=decision["cert"],decision["action"]
        row={"tick":self.tick,"decision":self.decisions,"route":route,"selected":action,
             "n_samples":decision["stats"].n_samples,"pose":self.pose.mean.tolist(),
             "gain_estimate":self.dynamics.mean,"gain_std":self.dynamics.std,
             "scores":decision["scores"],"cost":self.ledger.cost,"mission_reserve":self.mission_reserve(decision),
             "risk_duration_seconds":self.cfg.dt*self.cfg.risk_horizon}
        if action is not None:row.update(support_lcb=float(cert.support_lcb[action]),risk_ucb=float(cert.risk_ucb[action]))
        else:row.update(risk_ucb_all=cert.risk_ucb.tolist(),support_lcb_all=cert.support_lcb.tolist())
        row["selected_control"]=None if action is None else decision["controls"][action].tolist()
        row["resource_history"]=decision["resource_history"]
        if "resource_predictions" in decision:
            row["resource_predictions"]=decision["resource_predictions"]
            row["policy_features"]=decision["policy_features"]
        self.trace.append(row)

    def step(self):
        if self.reason!="running":return
        self.begin_tick();decision=self.prepare_decision()
        if self.method=="nominal":
            self.ledger.spend("SIMULATE",self.ledger.charge_for("SIMULATE",n_samples=1))
            point=sample_worlds((self.map.probability>.5).astype(float),self.pose.mean,np.zeros((3,3)),decision["controls"],
                decision["waypoint"],self.dynamics.mean if self.adaptive_model else 1.,0.,keyed_rng(self.seed,self.decisions,302),1,
                self.cfg.dt,self.cfg.horizon,self.cfg.cell_size,self.cfg.risk_horizon,self.cfg.collision_substeps,self.cfg.stability_regret)
            idx=int(np.argmax(point.mean_scores));control=decision["controls"][idx]
            self.previous_gps=self.latest_measured_pose.copy();self.previous_heading=float(self.pose.mean[2]);self.previous_control=control.copy()
            motion=self.client.act(control);self.pose.predict(control,self.cfg.dt,self.dynamics.mean,self.dynamics.std);self.tick=motion.tick
            self.trace.append({"tick":self.tick-1,"route":"ACT","selected":idx,"n_samples":1,"cost":self.ledger.cost})
            if motion.collision:self.outcome,self.reason="collision","physical_contact"
            elif motion.reached_goal:self.outcome,self.reason="success","physical_arrival"
            return
        decision=self.initial_compute(decision)
        while self.reason=="running":
            if self.dataset_callback is not None:self.dataset_callback(self,decision)
            route=self.choose_resource(decision)
            stop=route=="DEFER" or (decision["resource_attempts"]>=self.cfg.max_epistemic_streak and route not in ("SIMULATE","EXPERT","ACT","PROBE"))
            if stop and self.recovery_action(decision) is not None:
                decision["action"]=self.recovery_action(decision);route="RECOVER"
            self._log(decision,route)
            if route=="RECOVER":
                self._move(decision,recovery=True);return
            if route=="DEFER":self.reason="resource_policy_defer";return
            if route in ("ACT","PROBE"):
                self._move(decision,probe=route=="PROBE");return
            if decision["resource_attempts"]>=self.cfg.max_epistemic_streak and route not in ("SIMULATE","EXPERT"):
                self.reason="unresolved_uncertainty";return
            decision=self.acquire(route,decision)

    def result(self,wall_seconds=0.):
        return {"method":self.method,"seed":self.seed,"outcome":self.outcome,"reason":self.reason,
            "steps":self.tick,"decisions":self.decisions,"resources":self.ledger.summary(),
            "action_counts":dict(Counter(e["action"] for e in self.ledger.entries)),"wall_seconds":wall_seconds,
            "final_gain_estimate":self.dynamics.mean,"trace":self.trace,
            "feature_names":list(FEATURE_NAMES),"feature_rows":self.feature_rows,
            "certificate_scope":"simultaneous action/declared-look bounds for the configured risk forecast, including the executed control, under the internal distribution; no model-correctness or trajectory guarantee"}

    def run(self):
        wall=time.perf_counter()
        try:
            while self.reason=="running" and self.tick<self.cfg.max_steps:self.step()
            if self.reason=="running":self.reason="max_steps"
        except BudgetExceeded as exc:self.reason="budget_"+str(exc)
        return self.result(time.perf_counter()-wall)
