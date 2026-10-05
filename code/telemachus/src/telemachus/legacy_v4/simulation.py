from __future__ import annotations

from dataclasses import dataclass
import json
import numpy as np
from .physics import unicycle_step, wrap_angle
from .planning import sampled_collision, astar
from .provenance import keyed_rng, canonical_hash
from .vision import render_tiles
from .radio import RadioLink


@dataclass(frozen=True)
class Scenario:
    name: str = "nominal"
    camera_noise: float = 0.035
    camera_brightness: float = 1.0
    camera_occlusion: float = 0.0
    localization_sigma: float = 0.045
    speed_gain: float = 1.0
    channel_snr_db: float = 8.0
    onset: int = 0
    layout_family: int = 0
    camera_style: int = 0


def scenarios(split="development"):
    if split not in ("development", "holdout"):
        raise ValueError("Unknown split")
    h = split == "holdout"
    common = {"layout_family": 2 if h else 0, "camera_style": 2 if h else 0}
    return [Scenario("nominal", **common),
            Scenario("visual", camera_noise=.24 if h else .20, camera_brightness=.30 if h else .38,
                     camera_occlusion=.30, **common),
            Scenario("localization", localization_sigma=.24 if h else .18, **common),
            Scenario("dynamics", speed_gain=.53 if h else .62, onset=16, **common),
            Scenario("radio", channel_snr_db=-3.0 if h else -1.5, **common),
            Scenario("compound", camera_noise=.19, camera_brightness=.42, localization_sigma=.16,
                     speed_gain=.62, channel_snr_db=-2.0, onset=12, **common)]


@dataclass(frozen=True)
class SensorReading:
    gps_burst: np.ndarray
    tiles: np.ndarray
    tile_centers: np.ndarray
    tick: int
    focus: bool


@dataclass(frozen=True)
class MotionResult:
    collision: bool
    reached_goal: bool
    tick: int


class NavigationWorld:
    """Hidden simulator state. Never passed to the decision policy.

    SensorReading has no scenario, obstacle labels, true pose or dynamics gain.
    MotionResult carries only externally detectable contact/arrival events.
    """
    def __init__(self, seed, scenario, cfg, goal=None):
        self.seed, self.scenario, self.cfg = int(seed), scenario, cfg
        self.n = int(round(cfg.world_size / cfg.cell_size))
        if self.n < 16:
            raise ValueError("World needs at least 16 cells per dimension")
        self.grid = self._layout()
        self.pose = np.array([.75, .75, np.pi / 4])
        self.goal = np.array([cfg.world_size - .75, cfg.world_size - .75]) if goal is None else np.asarray(goal,float)
        if self.goal.shape!=(2,) or not np.isfinite(self.goal).all() or min(self.goal)<.25 or max(self.goal)>cfg.world_size-.25:
            raise ValueError("Invalid mission goal")
        gr,gc=np.floor(self.goal[::-1]/cfg.cell_size).astype(int)
        if self.grid[gr,gc]:
            raise ValueError("Mission goal lies inside an obstacle")
        self.tick = 0
        self.sense_index, self.packet_index = 0, 0
        self.collided = False
        self.trajectory = [self.pose.tolist()]
        self.radio = RadioLink(cfg.channel_backend)
        self.world_hash = canonical_hash({"seed": seed, "grid": self.grid.astype(int).tolist(),
                                          "scenario": scenario.__dict__,"goal":self.goal.tolist()})

    def _layout(self):
        rng = keyed_rng(self.seed, self.scenario.layout_family, 201)
        g = np.zeros((self.n, self.n), dtype=bool)
        for _ in range(18):
            r, c = rng.integers(2, self.n - 3, 2)
            height, width = rng.integers(1, 4, 2)
            g[r:r + height, c:c + width] = True
        if self.scenario.layout_family>=2:
            # Unseen structural family: staggered walls, subsequently crossed
            # by a protected bent route. This is more than changing the seed.
            g[2:self.n-2,self.n//3]=True
            g[2:self.n-2,2*self.n//3]=True
        # Feasible wide routes, with random bends. Protected cells are never
        # given to the agent: feasibility belongs to world construction only.
        k = int(rng.integers(7, self.n - 7))
        points = [(1, 1), (k, 1), (k, self.n - 2), (self.n - 2, self.n - 2)]
        if self.scenario.layout_family % 2 == 0:
            points = [(c, r) for r, c in points]
        if self.scenario.layout_family>=2:
            points=[(1,1),(k,1),(k,self.n-5),(self.n-2,self.n-5),(self.n-2,self.n-2)]
        for (r0, c0), (r1, c1) in zip(points, points[1:]):
            for t in np.linspace(0, 1, max(abs(r1-r0), abs(c1-c0)) + 1):
                r, c = int(round(r0 + t*(r1-r0))), int(round(c0 + t*(c1-c0)))
                g[max(0,r-1):min(self.n,r+2), max(0,c-1):min(self.n,c+2)] = False
        g[:4, :4] = False; g[-4:, -4:] = False
        return g

    def _active(self):
        return self.tick >= self.scenario.onset

    def sense(self, focused=False, remote=False):
        self.sense_index += 1
        rng = keyed_rng(self.seed, self.tick, 202, self.sense_index)
        sigma = self.scenario.localization_sigma if self._active() else .045
        if focused or remote:
            sigma = .018
        burst = np.repeat(self.pose[None], 6 if focused else 3, axis=0)
        burst += rng.normal(0, [sigma, sigma, sigma*.65], burst.shape)
        burst[:, 2] = wrap_angle(burst[:, 2])
        center = np.floor(self.pose[:2][::-1] / self.cfg.cell_size).astype(int)
        radius = 4 if not focused else 5
        rows = np.arange(max(0,center[0]-radius), min(self.n,center[0]+radius+1))
        cols = np.arange(max(0,center[1]-radius), min(self.n,center[1]+radius+1))
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        coords = np.column_stack([(cc.ravel()+.5)*self.cfg.cell_size,
                                  (rr.ravel()+.5)*self.cfg.cell_size])
        # Calibrated overhead camera has measured georeferencing. A common
        # GPS error shifts the whole frame; the agent never receives indices.
        coords += np.mean(burst[:, :2], axis=0) - self.pose[:2]
        noisy = self._active() and not (focused or remote)
        tiles = render_tiles(self.grid[rr, cc].ravel(), rng,
                             noise=self.scenario.camera_noise if noisy else .025,
                             brightness=self.scenario.camera_brightness if noisy else 1.,
                             occlusion=self.scenario.camera_occlusion if noisy else 0.,
                             style=self.scenario.camera_style)
        return SensorReading(burst, tiles, coords, self.tick, focused or remote)

    def remote_packet(self, vision):
        self.packet_index += 1
        reading = self.sense(remote=True)
        # A peer runs the same learned perception, then sends compact soft
        # evidence and measured georeferencing. No simulator label is sent.
        probabilities, variance, _ = vision.predict(reading.tiles)
        values = np.column_stack([reading.tile_centers, probabilities, variance]).astype("<f2")
        payload = len(values).to_bytes(2, "little") + values.tobytes()
        rng = keyed_rng(self.seed, self.tick, 203, self.packet_index)
        snr = self.scenario.channel_snr_db if self._active() else 8.0
        result = self.radio.transmit(payload, snr, rng)
        return result

    @staticmethod
    def decode_packet(result, tick):
        if not result.delivered:
            return None
        count = int.from_bytes(result.payload[:2], "little")
        values = np.frombuffer(result.payload[2:], "<f2").reshape(count, 4).astype(float)
        return values[:, :2], values[:, 2], values[:, 3]

    def move(self, control):
        u = np.asarray(control, float)
        if u.shape != (2,) or not np.isfinite(u).all() or u[0] < 0 or u[0] > .8 or abs(u[1]) > 1.8:
            raise ValueError("Control outside physical actuator limits")
        gain = self.scenario.speed_gain if self._active() else 1.
        for _ in range(8):
            candidate = unicycle_step(self.pose, u, self.cfg.dt/8, gain)
            if sampled_collision(candidate[None,None], self.grid[None], self.cfg.cell_size)[0,0]:
                self.collided = True
                break
            self.pose = candidate
        self.tick += 1; self.trajectory.append(self.pose.tolist())
        reached = bool(np.linalg.norm(self.pose[:2]-self.goal) <= self.cfg.goal_tolerance)
        return MotionResult(self.collided, reached, self.tick)

    def truth_record(self):
        # Evaluation only; the policy never calls this method.
        return {"scenario": self.scenario.__dict__, "world_hash": self.world_hash,
                "trajectory": self.trajectory, "grid": self.grid.astype(int).tolist(),
                "true_final_pose": self.pose.tolist(), "goal": self.goal.tolist()}


class SensorActuatorClient:
    """Restricted observable interface, explicitly excluding world truth."""
    def __init__(self, world, vision):
        self.__world = world
        self.__vision = vision
        self.goal = world.goal.copy()

    def sense(self, focused=False):
        return self.__world.sense(focused=focused)

    def ask(self):
        return self.__world.remote_packet(self.__vision)

    def ask_reservation(self):
        # Reserve the largest possible local camera packet before transmission.
        return RadioLink.reservation(2 + 9*9*8)

    def act(self, control):
        return self.__world.move(control)
