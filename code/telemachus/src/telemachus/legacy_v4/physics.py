from __future__ import annotations

import numpy as np


def wrap_angle(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def unicycle_step(state, control, dt, speed_gain=1.0, turn_gain=1.0):
    """Vectorized exact constant-input integration, including the w -> 0 limit."""
    x = np.asarray(state, dtype=float)
    u = np.asarray(control, dtype=float)
    v = np.asarray(speed_gain) * u[..., 0]
    w = np.asarray(turn_gain) * u[..., 1]
    theta = x[..., 2]
    half = w * dt * 0.5
    travel = v * dt * np.sinc(half / np.pi)
    return np.stack([x[..., 0] + travel * np.cos(theta + half),
                     x[..., 1] + travel * np.sin(theta + half),
                     wrap_angle(theta + w * dt)], axis=-1)


def rk4_step(state, control, dt, speed_gain=1.0, turn_gain=1.0):
    x = np.asarray(state, dtype=float)
    u = np.asarray(control, dtype=float)
    def f(z):
        return np.array([speed_gain * u[0] * np.cos(z[2]), speed_gain * u[0] * np.sin(z[2]), turn_gain * u[1]])
    a = f(x); b = f(x + dt * a / 2); c = f(x + dt * b / 2); d = f(x + dt * c)
    y = x + dt * (a + 2 * b + 2 * c + d) / 6
    y[2] = wrap_angle(y[2])
    return y


class PoseFilter:
    """EKF with sensor-noise estimation from observable innovation/burst residuals."""
    def __init__(self, measurement):
        self.mean = np.asarray(measurement, dtype=float).copy()
        self.cov = np.diag([0.004, 0.004, 0.003])
        self.measurement_var = np.array([0.004, 0.004, 0.003])
        self.innovation_energy = 0.0
        self.noise_evidence = 0.0
        self.last_innovation = np.zeros(3)

    def predict(self, control, dt, gain=1.0, gain_std=0.06):
        theta = self.mean[2]
        v = control[0] * gain
        F = np.eye(3)
        F[0, 2] = -v * dt * np.sin(theta)
        F[1, 2] = v * dt * np.cos(theta)
        self.mean = unicycle_step(self.mean, control, dt, gain)
        q = 0.00025 + (abs(control[0]) * dt * gain_std) ** 2
        self.cov = F @ self.cov @ F.T + np.diag([q, q, 0.00035])

    def update(self, measurements, focused=False):
        z = np.asarray(measurements, dtype=float)
        if z.ndim == 1:
            z = z[None, :]
        if len(z) > 1:
            # Burst samples are collected at the same physical pose.
            tmp = z.copy()
            tmp[:, 2] = self.mean[2] + wrap_angle(tmp[:, 2] - self.mean[2])
            variance = np.maximum(np.var(tmp, axis=0, ddof=1), [0.00005, 0.00005, 0.00005])
            if not focused:
                self.measurement_var = 0.75 * self.measurement_var + 0.25 * variance
            self.noise_evidence = 0.8 * self.noise_evidence + 0.2 * float(np.mean(variance[:2]))
            R = np.diag(variance / len(z))
            zz = np.mean(tmp, axis=0)
        else:
            zz = z[0]
            R = np.diag(self.measurement_var * (0.20 if focused else 1.0))
        residual = zz - self.mean
        residual[2] = wrap_angle(residual[2])
        S = self.cov + R
        energy = float(residual @ np.linalg.solve(S, residual))
        self.innovation_energy = 0.8 * self.innovation_energy + 0.2 * min(energy, 50.0)
        self.last_innovation = residual.copy()
        K = np.linalg.solve(S.T, self.cov.T).T
        self.mean = self.mean + K @ residual
        self.mean[2] = wrap_angle(self.mean[2])
        I = np.eye(3)
        self.cov = (I - K) @ self.cov @ (I - K).T + K @ R @ K.T
        self.cov = 0.5 * (self.cov + self.cov.T)
        return zz


class DynamicsIdentifier:
    def __init__(self):
        self.samples: list[tuple[float, float]] = []
        self.mean = 1.0
        self.std = 0.08
        self.residual_ewma = 0.0
        self.bias_ewma = 0.0
        self.n_updates = 0

    def update(self, before_xy, after_xy, heading, v, dt, weight=1.0):
        commanded = float(v * dt)
        if commanded < 0.07:
            return
        observed = float(np.dot(np.asarray(after_xy) - np.asarray(before_xy), [np.cos(heading), np.sin(heading)]))
        if not np.isfinite(observed):
            return
        ratio = float(np.clip(observed / commanded, -0.2, 2.5))
        residual = observed - self.mean * commanded
        self.residual_ewma = 0.82 * self.residual_ewma + 0.18 * abs(residual)
        self.bias_ewma = 0.82 * self.bias_ewma + 0.18 * residual
        self.samples.append((ratio, max(0.1, float(weight))))
        self.samples = self.samples[-18:]
        values = np.array([x for x, _ in self.samples])
        weights = np.array([w for _, w in self.samples])
        center = float(np.median(values))
        clipped = np.clip(values, center - 0.5, center + 0.5)
        self.mean = float(np.clip(np.average(clipped, weights=weights), 0.35, 1.45))
        spread = float(np.sqrt(np.average((clipped - self.mean) ** 2, weights=weights)))
        self.std = float(np.clip(spread / np.sqrt(max(len(values), 1)) + 0.025, 0.025, 0.30))
        self.n_updates += 1
