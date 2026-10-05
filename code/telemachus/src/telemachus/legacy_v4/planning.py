from __future__ import annotations

from dataclasses import dataclass
import heapq
import numpy as np
from .physics import unicycle_step, wrap_angle
from .certificate import certify_all_actions


def astar(probability, start_xy, goal_xy, cell_size=0.5):
    p = np.asarray(probability)
    n = p.shape[0]
    start = tuple(np.clip(np.floor(np.asarray(start_xy)[::-1] / cell_size).astype(int), 0, n - 1))
    goal = tuple(np.clip(np.floor(np.asarray(goal_xy)[::-1] / cell_size).astype(int), 0, n - 1))
    def heuristic(a):
        return float(np.hypot(a[0] - goal[0], a[1] - goal[1]))
    q = [(heuristic(start), 0.0, start)]
    costs = {start: 0.0}; parents = {start: None}
    while q:
        _, cost, cur = heapq.heappop(q)
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur); cur = parents[cur]
            return path[::-1]
        if cost > costs[cur] + 1e-10:
            continue
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nxt = (cur[0] + dr, cur[1] + dc)
            if min(nxt) < 0 or max(nxt) >= n or (p[nxt] > 0.48 and nxt != goal):
                continue
            if dr and dc and (p[cur[0] + dr, cur[1]] > 0.35 or p[cur[0], cur[1] + dc] > 0.35):
                continue
            step = np.hypot(dr, dc) + 4.0 * p[nxt]
            total = cost + step
            if total < costs.get(nxt, float("inf")):
                costs[nxt] = total; parents[nxt] = cur
                heapq.heappush(q, (total + heuristic(nxt), total, nxt))
    return []


def path_waypoint(path, goal, cell_size=0.5):
    if len(path) < 2:
        return np.asarray(goal, float)
    index = 1
    if len(path) > 2:
        a = np.subtract(path[1], path[0]); b = np.subtract(path[2], path[1])
        if np.array_equal(a, b):
            index = 2
    r, c = path[index]
    return np.array([(c + 0.5) * cell_size, (r + 0.5) * cell_size])


def candidate_controls(pose, waypoint):
    target_angle = np.arctan2(waypoint[1] - pose[1], waypoint[0] - pose[0])
    error = float(wrap_angle(target_angle - pose[2]))
    w = float(np.clip(2.5 * error, -1.8, 1.8))
    # Rotate-in-place is a physical action, not a free epistemic action.
    rows = [[0.0, w], [0.25, w], [0.50, w], [0.80, w]]
    rows.extend([[0.50, float(np.clip(w + d, -1.8, 1.8))] for d in [-0.50, 0.50]])
    rows.extend([[0.80, float(np.clip(w + d, -1.8, 1.8))] for d in [-0.35, 0.35]])
    return np.array(rows)


def sampled_collision(states, worlds, cell_size=0.5, radius=0.12):
    """Rectangle collision, including neighboring cells and the robot radius."""
    states = np.asarray(states)
    k, a = states.shape[:2]
    n = worlds.shape[-1]; length = n * cell_size
    xx, yy = states[..., 0], states[..., 1]
    hit = (xx < radius) | (yy < radius) | (xx > length - radius) | (yy > length - radius)
    rows = np.floor(yy / cell_size).astype(int)
    cols = np.floor(xx / cell_size).astype(int)
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            rr, cc = rows + dr, cols + dc
            valid = (rr >= 0) & (rr < n) & (cc >= 0) & (cc < n)
            occ = worlds[np.arange(k)[:, None], np.clip(rr, 0, n - 1), np.clip(cc, 0, n - 1)]
            xlo, ylo = cc * cell_size, rr * cell_size
            dx = np.maximum(np.maximum(xlo - xx, xx - (xlo + cell_size)), 0)
            dy = np.maximum(np.maximum(ylo - yy, yy - (ylo + cell_size)), 0)
            hit |= valid & occ & (dx * dx + dy * dy <= radius * radius)
    return hit


@dataclass
class WorldStatistics:
    support_counts: np.ndarray
    risk_counts: np.ndarray
    score_sums: np.ndarray
    n_samples: int

    @property
    def mean_scores(self):
        return self.score_sums / self.n_samples

    def merge(self, other):
        return WorldStatistics(self.support_counts + other.support_counts,
                               self.risk_counts + other.risk_counts,
                               self.score_sums + other.score_sums, self.n_samples + other.n_samples)


def sample_worlds(probability, pose, pose_cov, controls, waypoint, gain_mean, gain_std,
                  rng, n_samples, dt=0.4, horizon=3, cell_size=0.5):
    k = n_samples; n = probability.shape[0]; a = len(controls)
    worlds = rng.random((k, n, n)) < probability[None, :, :]
    poses = rng.multivariate_normal(pose, pose_cov + np.eye(3) * 1e-9, k)
    # The physical position currently occupied by the robot is observed free.
    rr = np.clip(np.floor(poses[:, 1] / cell_size).astype(int), 0, n - 1)
    cc = np.clip(np.floor(poses[:, 0] / cell_size).astype(int), 0, n - 1)
    worlds[np.arange(k), rr, cc] = False
    gains = np.clip(rng.normal(gain_mean, gain_std, k), 0.3, 1.5)
    states = np.repeat(poses[:, None, :], a, axis=1)
    risk = np.zeros((k, a), bool)
    # Substeps prevent geometric tunneling in the internal model as well.
    for tick in range(horizon):
        if tick == 0:
            future_controls = np.broadcast_to(controls[None,:,:], (k,a,2))
        else:
            # Frozen downstream feedback policy. Holding a turn command for
            # the entire horizon falsely predicts arcs into known obstacles.
            delta=waypoint[None,None,:]-states[...,:2]
            error=wrap_angle(np.arctan2(delta[...,1],delta[...,0])-states[...,2])
            turn=np.clip(2.5*error,-1.8,1.8)
            speed=np.broadcast_to(controls[None,:,0],(k,a))*np.clip(np.cos(error),.15,1.)
            speed=np.minimum(speed,np.linalg.norm(delta,axis=-1)/dt)
            future_controls=np.stack([speed,turn],axis=-1)
        for _ in range(2):
            states=unicycle_step(states,future_controls,dt/2,gains[:,None])
            risk |= sampled_collision(states,worlds,cell_size)
    distance = np.linalg.norm(states[..., :2] - waypoint[None, None, :], axis=-1)
    scores = -distance - 0.04 * np.abs(controls[None, :, 1]) - 4.0 * risk
    # Stability means near-optimal downstream utility, not arbitrary tie-breaking.
    support = scores >= np.max(scores, axis=1, keepdims=True) - 0.07
    return WorldStatistics(support.sum(0), risk.sum(0), scores.sum(0), k)


def choose_certified(stats, cfg):
    cert = certify_all_actions(stats.support_counts, stats.risk_counts, stats.n_samples,
                               cfg.mc_delta, len(cfg.mc_looks))
    valid = cert.acceptable(cfg.stability_threshold, cfg.risk_threshold)
    if not np.any(valid):
        return None, cert
    score = np.where(valid, stats.mean_scores, -np.inf)
    return int(np.argmax(score)), cert
