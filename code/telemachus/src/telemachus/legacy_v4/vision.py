from __future__ import annotations

from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from torch import nn
from scipy.optimize import minimize_scalar
from .provenance import keyed_rng, write_json, canonical_hash, utc_now


TILE_SIZE = 12


def render_tiles(occupied, rng, noise=0.035, brightness=1.0, occlusion=0.0, style=0):
    """Synthetic calibrated overhead-camera tiles. Truth stays inside rendering.

    The task is deliberately synthetic and is not a photorealistic camera claim.
    The predictor receives only the resulting RGB arrays.
    """
    occupied = np.asarray(occupied, dtype=bool).reshape(-1)
    n = len(occupied)
    a = np.empty((n, 3, TILE_SIZE, TILE_SIZE), dtype=np.float32)
    yy, xx = np.mgrid[:TILE_SIZE, :TILE_SIZE]
    for i, occ in enumerate(occupied):
        bg = np.array([0.77, 0.81, 0.83]) if style != 2 else np.array([0.81, 0.77, 0.71])
        a[i] = bg[:, None, None]
        # Floor texture is present in both classes, with independent nuisance data.
        a[i] += rng.normal(0, 0.016, a[i].shape)
        if occ:
            if rng.random() < 0.45:
                mask = (xx - 5.5) ** 2 + (yy - 5.5) ** 2 < rng.uniform(17, 25)
            else:
                pad = int(rng.integers(1, 3))
                mask = (xx >= pad) & (xx < TILE_SIZE - pad) & (yy >= pad) & (yy < TILE_SIZE - pad)
            color = np.array([0.21, 0.31, 0.43]) if style != 2 else np.array([0.34, 0.27, 0.22])
            a[i, :, mask] = color[:, None].T
        elif rng.random() < 0.25:
            # Thin floor markings prevent a simple mean-intensity oracle.
            col = int(rng.integers(TILE_SIZE))
            a[i, :, :, col] -= 0.09
        if occlusion > 0 and rng.random() < occlusion:
            start = int(rng.integers(0, TILE_SIZE - 3))
            a[i, :, start:start + 4, :] = 0.59 + rng.normal(0, noise, (3, 4, TILE_SIZE))
    # Low light, fog, and sensor noise are measurements, not supplied to the agent.
    a = 0.5 + (a - 0.5) * brightness
    a += rng.normal(0, noise, a.shape).astype(np.float32)
    return np.clip(a, 0, 1).astype(np.float32)


class TileCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(8, 16, 3, padding=1), nn.GELU(),
                                 nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten(),
                                 nn.Linear(144, 24), nn.GELU(), nn.Linear(24, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _dataset(seed, n):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    groups = np.arange(n) // 64
    x = np.empty((n, 3, TILE_SIZE, TILE_SIZE), np.float32)
    for group in np.unique(groups):
        ids = np.flatnonzero(groups == group)
        x[ids] = render_tiles(y[ids], rng, noise=float(rng.uniform(0.015, 0.15)),
                              brightness=float(rng.uniform(0.55, 1.15)),
                              occlusion=float(rng.uniform(0, 0.12)), style=int(rng.integers(2)))
    return x, y.astype(np.float32)


def expected_calibration_error(prob, labels, bins=10):
    p = np.asarray(prob); y = np.asarray(labels)
    ece = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        hi = lo + 1 / bins
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if np.any(mask):
            ece += np.mean(mask) * abs(np.mean(p[mask]) - np.mean(y[mask]))
    return float(ece)


def train_vision(output, *, seeds=(11, 23, 37), n_train=4608, epochs=10, threads=2):
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    x, y = _dataset(7101, n_train)
    xc, yc = _dataset(7102, 1536)
    xt, yt = _dataset(7103, 1536)
    X, Y = torch.from_numpy(x), torch.from_numpy(y)
    states, temperatures, training = [], [], []
    start = time.perf_counter()
    test_logits = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = TileCNN()
        opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0005)
        order_rng = np.random.default_rng(seed)
        logs = []
        for ep in range(epochs):
            values = []
            model.train()
            for ids in np.array_split(order_rng.permutation(len(X)), max(1, len(X) // 128)):
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.binary_cross_entropy_with_logits(model(X[ids]), Y[ids])
                loss.backward(); opt.step()
                values.append(float(loss.detach()))
            logs.append({"epoch": ep + 1, "loss": float(np.mean(values))})
        model.eval()
        with torch.inference_mode():
            logits_c = model(torch.from_numpy(xc)).numpy()
            logits_t = model(torch.from_numpy(xt)).numpy()
        def nll(log_temp):
            z = logits_c / np.exp(log_temp)
            return float(np.mean(np.logaddexp(0, z) - yc * z))
        fit = minimize_scalar(nll, bounds=(-1.5, 2.0), method="bounded")
        temp = float(np.exp(fit.x))
        temperatures.append(temp)
        states.append(model.state_dict())
        training.append({"seed": seed, "temperature": temp, "epochs": logs})
        test_logits.append(logits_t / temp)
    mean_prob = np.mean([1 / (1 + np.exp(-np.clip(z, -40, 40))) for z in test_logits], axis=0)
    p = Path(output); p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dicts": states, "temperatures": temperatures,
                "architecture": "TileCNN-v1", "seeds": list(seeds)}, p)
    report = {"timestamp": utc_now(), "data_seeds": {"train": 7101, "calibration": 7102, "test": 7103},
              "n_train": len(x), "n_calibration": len(xc), "n_test": len(xt),
              "synthetic_camera": True, "pretrained": False,
              "training": training, "test_accuracy": float(np.mean((mean_prob >= 0.5) == yt)),
              "test_brier": float(np.mean((mean_prob - yt) ** 2)),
              "test_ece": expected_calibration_error(mean_prob, yt),
              "fit_wall_seconds": time.perf_counter() - start,
              "train_data_hash": canonical_hash({"seed": 7101, "n": n_train, "version": 1})}
    write_json(p.with_suffix(".json"), report)
    np.savez_compressed(p.with_name("vision_test_predictions.npz"), y=yt, p=mean_prob)
    return report


class NeuralVision:
    def __init__(self, checkpoint, threads=2):
        torch.set_num_threads(threads)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["architecture"] != "TileCNN-v1":
            raise ValueError("Unknown vision architecture")
        self.models = []
        for state in saved["state_dicts"]:
            model = TileCNN(); model.load_state_dict(state); model.eval()
            self.models.append(model)
        self.temperatures = saved["temperatures"]

    def predict(self, tiles):
        x = np.asarray(tiles, dtype=np.float32)
        if x.ndim != 4 or x.shape[1:] != (3, TILE_SIZE, TILE_SIZE):
            raise ValueError("Expected N x 3 x 12 x 12 RGB tiles")
        if not np.isfinite(x).all() or np.min(x) < 0 or np.max(x) > 1:
            raise ValueError("Invalid image input")
        with torch.inference_mode():
            tensor = torch.from_numpy(np.ascontiguousarray(x))
            probs = np.array([torch.sigmoid(model(tensor) / t).numpy()
                              for model, t in zip(self.models, self.temperatures)])
        mean = np.clip(np.mean(probs, axis=0), 0.001, 0.999)
        variance = np.var(probs, axis=0)
        entropy = -(mean * np.log(mean) + (1 - mean) * np.log(1 - mean)) / np.log(2)
        return mean, variance, entropy


def camera_quality_features(tiles):
    x = np.asarray(tiles)
    hf = np.mean(np.abs(np.diff(x, axis=-1)))
    return {"image_high_frequency": float(hf), "image_contrast": float(np.std(x)),
            "image_brightness": float(np.mean(x))}


def tiles_to_image(tiles, side):
    x = np.asarray(tiles).reshape(side, side, 3, TILE_SIZE, TILE_SIZE)
    mosaic = x.transpose(0, 3, 1, 4, 2).reshape(side * TILE_SIZE, side * TILE_SIZE, 3)
    return Image.fromarray(np.uint8(np.clip(mosaic, 0, 1) * 255))
