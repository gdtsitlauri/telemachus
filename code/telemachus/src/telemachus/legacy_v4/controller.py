from __future__ import annotations

from collections import Counter
import time
import numpy as np
from .config import ControllerConfig
from .budget import ResourceLedger, BudgetExceeded
from .physics import PoseFilter, DynamicsIdentifier
from .belief import MapBelief, ObservableDiagnosis, observed_features, FEATURE_NAMES
from .radio import ChannelBelief
from .simulation import NavigationWorld
from .planning import astar, path_waypoint, candidate_controls, sample_worlds, choose_certified
from .provenance import keyed_rng


METHODS = ("nominal", "confidence_gate", "budget_voi", "telemachus",
           "no_source", "no_model_update", "fixed_compute")


class Controller:
    def __init__(self, client, vision, config: ControllerConfig, method="telemachus", seed=0):
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        self.client, self.vision, self.cfg, self.method, self.seed = client, vision, config, method, seed
        self.ledger = ResourceLedger(config.budget)
        self.map = MapBelief(int(round(config.world_size/config.cell_size)), config.cell_size)
        self.dynamics, self.channel, self.diagnosis = DynamicsIdentifier(), ChannelBelief(), ObservableDiagnosis()
        self.pose = None
        self.decisions, self.trace, self.feature_rows = 0, [], []
        self.last_probed_tick = -100
        self.last_focused_tick = -100

    def _sense(self, focused=False):
        name = "OBSERVE" if focused else "PASSIVE"
        self.ledger.spend(name, self.ledger.charge_for(name))
        reading = self.client.sense(focused=focused)
        if self.pose is None:
            self.pose = PoseFilter(np.mean(reading.gps_burst,axis=0))
        self.pose.update(reading.gps_burst, focused=focused)
        self.map.update(reading,self.vision,self.pose.mean)
        if focused:
            self.last_focused_tick = reading.tick
        return reading

    def _ask(self, tick):
        nbytes, latency = self.client.ask_reservation()
        self.ledger.spend("ASK",self.ledger.charge_for("ASK",radio_latency_ms=latency,n_bytes=nbytes))
        packet = self.client.ask()
        self.ledger.reconcile_radio(packet.bytes_sent,packet.logical_latency_ms)
        self.channel.update(packet)
        evidence = NavigationWorld.decode_packet(packet,tick)
        if evidence is not None:
            self.map.update_remote(*evidence, self.pose.mean)
            self.last_focused_tick = tick
        return packet.delivered

    def _sample(self, controls, waypoint, n, decision, batch):
        self.ledger.spend("SIMULATE",self.ledger.charge_for("SIMULATE",n_samples=n))
        adapt = self.cfg.adapt_dynamics and self.method != "no_model_update"
        return sample_worlds(self.map.probability,self.pose.mean,self.pose.cov,controls,waypoint,
                             self.dynamics.mean if adapt else 1., self.dynamics.std if adapt else .08,
                             keyed_rng(self.seed,decision,301,batch),n,self.cfg.dt,self.cfg.horizon,self.cfg.cell_size)

    def _route(self, scores, tick, has_action):
        source_aware = self.method in ("telemachus","no_model_update","fixed_compute")
        if source_aware:
            if (scores["model"] > self.cfg.source_threshold and tick-self.last_probed_tick >= 12
                    and has_action and self.pose.noise_evidence<.009 and self.dynamics.n_updates>=6
                    and self.dynamics.residual_ewma>.03):
                if self.ledger.can_afford(self.ledger.charge_for("PROBE")):
                    return "PROBE"
            if has_action:
                return None
            if scores["localization"] >= scores["vision"]:
                return "OBSERVE"
            if scores["vision"] > self.cfg.source_threshold:
                if self.channel.delivery_probability >= self.cfg.channel_min_delivery and self.channel.snr_db > 0:
                    return "ASK"
                return "OBSERVE"
            return "OBSERVE"
        if has_action:
            return None
        if self.method in ("budget_voi","no_source"):
            # Source-blind value/cost proxy: global map entropy, delivery and
            # resource prices. Same action set and limits as TELEMACHUS.
            q_cost = self.cfg.budget.query_cost/max(self.channel.delivery_probability,.05)
            return "ASK" if q_cost < self.cfg.budget.observe_cost and self.channel.snr_db > 0 else "OBSERVE"
        return "OBSERVE"

    def run(self):
        wall = time.perf_counter(); outcome, reason = "deferred", "max_steps"
        tick = 0; last_control = None; before = None
        try:
            while tick < self.cfg.max_steps:
                reading = self._sense()
                if before is not None and last_control is not None:
                    # Identification uses measured displacement, including the
                    # commanded turn's midpoint, never an environment gain.
                    heading = before[2] + last_control[1]*self.cfg.dt/2
                    v = last_control[0]*np.sinc(last_control[1]*self.cfg.dt/(2*np.pi))
                    self.dynamics.update(before[:2],np.mean(reading.gps_burst[:,:2],axis=0),
                                         heading,v,self.cfg.dt)
                    before = None; last_control = None
                for epistemic in range(self.cfg.max_epistemic_streak+1):
                    self.decisions += 1
                    features = observed_features(self.map,self.pose,self.dynamics,self.channel)
                    scores = self.diagnosis.predict(features)
                    self.feature_rows.append({"tick":tick,"features":features.tolist(),"scores":scores})
                    path = astar(self.map.probability,self.pose.mean[:2],self.client.goal,self.cfg.cell_size)
                    if not path:
                        waypoint = self.client.goal
                    else:
                        waypoint = path_waypoint(path,self.client.goal,self.cfg.cell_size)
                    controls = candidate_controls(self.pose.mean,waypoint)
                    cert = None; stats = None
                    if self.method == "nominal":
                        # Point-estimate receding-horizon control, same neural
                        # map and EKF, no uncertainty gate or extra observations.
                        # Same collision-aware downstream feedback and utility
                        # as every gated method, evaluated at one deterministic
                        # map/pose/gain rather than uncertainty samples.
                        self.ledger.spend("SIMULATE",self.ledger.charge_for("SIMULATE",n_samples=1))
                        point=sample_worlds((self.map.probability>.5).astype(float),self.pose.mean,
                              np.zeros((3,3)),controls,waypoint,self.dynamics.mean if self.cfg.adapt_dynamics else 1.,0.,
                              keyed_rng(self.seed,self.decisions,302),1,self.cfg.dt,self.cfg.horizon,self.cfg.cell_size)
                        action=int(np.argmax(point.mean_scores))
                        route = None
                    else:
                        start_look = 1 if self.method == "fixed_compute" and len(self.cfg.mc_looks)>1 else 0
                        action = None
                        for look in range(start_look,len(self.cfg.mc_looks)):
                            n = self.cfg.mc_looks[look]-(0 if stats is None else stats.n_samples)
                            if not self.ledger.can_afford(self.ledger.charge_for("SIMULATE",n_samples=n)):
                                break
                            more = self._sample(controls,waypoint,n,self.decisions,look)
                            stats = more if stats is None else stats.merge(more)
                            action,cert = choose_certified(stats,self.cfg)
                            if action is not None:
                                break
                            # MC computation cannot repair evidence already
                            # suggesting a bad world/model. Re-observe instead.
                            if self.method in ("telemachus","no_model_update","fixed_compute") and max(scores["vision"],scores["localization"])>.75 and tick-self.last_focused_tick>=1:
                                break
                        route = self._route(scores,tick,action is not None)
                    row = {"tick":tick,"decision":self.decisions,"route":route or "ACT",
                           "selected":action,"n_samples":0 if stats is None else stats.n_samples,
                           "pose":self.pose.mean.tolist(),"gain_estimate":self.dynamics.mean,
                           "scores":scores,"cost":self.ledger.cost}
                    if cert is not None and action is not None:
                        row.update({"support_lcb":float(cert.support_lcb[action]),
                                    "risk_ucb":float(cert.risk_ucb[action])})
                    elif cert is not None:
                        row.update({"risk_ucb_all":cert.risk_ucb.tolist(),"support_lcb_all":cert.support_lcb.tolist()})
                    self.trace.append(row)
                    if route in ("OBSERVE","ASK"):
                        if epistemic >= self.cfg.max_epistemic_streak:
                            reason="unresolved_uncertainty"; break
                        if route == "OBSERVE":
                            self._sense(focused=True)
                        else:
                            nbytes, latency = self.client.ask_reservation()
                            charge=self.ledger.charge_for("ASK",radio_latency_ms=latency,n_bytes=nbytes)
                            if self.ledger.can_afford(charge):
                                self._ask(tick)
                            else:
                                self._sense(focused=True)
                        continue
                    if action is None:
                        reason="no_certified_action"; break
                    control = controls[action]
                    if route == "PROBE":
                        # Probe executes an already certified action, with
                        # accurate before/after bursts. It can physically fail.
                        self.ledger.spend("PROBE",self.ledger.charge_for("PROBE"))
                        pre=self.client.sense(focused=True)
                        premean=np.mean(pre.gps_burst,axis=0)
                        motion=self.client.act(control)
                        post=self.client.sense(focused=True)
                        postmean=np.mean(post.gps_burst,axis=0)
                        self.dynamics.update(premean[:2],postmean[:2],premean[2]+control[1]*self.cfg.dt/2,
                                             control[0]*np.sinc(control[1]*self.cfg.dt/(2*np.pi)),self.cfg.dt,weight=8.)
                        self.pose.predict(control,self.cfg.dt,self.dynamics.mean,self.dynamics.std)
                        self.pose.update(post.gps_burst,focused=True)
                        self.map.update(post,self.vision,self.pose.mean)
                        self.last_probed_tick=tick
                        self.last_focused_tick=tick+1
                    else:
                        # If extra observations changed the estimate, use the
                        # updated observable filter as the pre-action position.
                        before=self.pose.mean.copy()
                        motion=self.client.act(control)
                        last_control=control.copy()
                        adapt=self.cfg.adapt_dynamics and self.method!="no_model_update"
                        self.pose.predict(control,self.cfg.dt,self.dynamics.mean if adapt else 1.,
                                          self.dynamics.std if adapt else .08)
                    tick=motion.tick
                    if motion.collision:
                        outcome,reason="collision","physical_contact"
                    elif motion.reached_goal:
                        outcome,reason="success","physical_arrival"
                    break
                else:
                    reason="unresolved_uncertainty"
                if outcome in ("success","collision") or reason not in ("max_steps",):
                    break
            if tick>=self.cfg.max_steps and outcome=="deferred":
                reason="max_steps"
        except BudgetExceeded as exc:
            reason="budget_"+str(exc)
        counts=Counter(e["action"] for e in self.ledger.entries)
        return {"method":self.method,"seed":self.seed,"outcome":outcome,"reason":reason,
                "steps":tick,"decisions":self.decisions,"resources":self.ledger.summary(),
                "action_counts":dict(counts),"wall_seconds":time.perf_counter()-wall,
                "final_gain_estimate":self.dynamics.mean,"trace":self.trace,
                "feature_names":list(FEATURE_NAMES),"feature_rows":self.feature_rows,
                "certificate_scope":"selected action and predeclared looks under fixed internal distribution; not model correctness or full trajectory safety"}
