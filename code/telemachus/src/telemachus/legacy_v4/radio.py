from __future__ import annotations

from dataclasses import dataclass
import struct
import zlib
import numpy as np


@dataclass(frozen=True)
class PacketResult:
    delivered: bool
    payload: bytes | None
    pilot_snr_db: float
    bytes_sent: int
    logical_latency_ms: float
    backend: str


class RadioLink:
    """BPSK, repetition-3, pilot estimation and CRC through a noisy PHY.

    The simulator owns the actual SNR. Only a received pilot estimate and ACK
    reach the controller. Sionna's real AWGN block is used when requested.
    This is a simulated link, not a claim about deployed 6G equipment.
    """
    def __init__(self, backend="analytic"):
        if backend not in ("analytic", "sionna"):
            raise ValueError("Unknown channel backend")
        self.backend = backend
        self.block = None
        if backend == "sionna":
            from sionna.phy.channel import AWGN
            self.block = AWGN(device="cpu")

    @staticmethod
    def reservation(n_bytes):
        # Include the pilot and CRC and charge even a failed packet.
        bits = (n_bytes + 4) * 8 * 3 + 128
        return (bits + 7) // 8, 12.0 + bits / 32.0

    def transmit(self, payload, snr_db, rng):
        crc = struct.pack("<I", zlib.crc32(payload) & 0xffffffff)
        bits = np.unpackbits(np.frombuffer(payload + crc, dtype=np.uint8))
        symbols = np.concatenate([np.ones(128), 1.0 - 2.0 * np.repeat(bits, 3)])
        # N0 is the complex noise variance; real BPSK noise has N0/2.
        n0 = float(10.0 ** (-snr_db / 10.0))
        if self.block is None:
            received = symbols + rng.normal(0, np.sqrt(n0 / 2), len(symbols))
        else:
            import torch
            # The block has an independent generator, reset per packet from
            # the world's keyed RNG. Policy Monte Carlo cannot advance it.
            self.block.torch_rng.manual_seed(int(rng.integers(0, 2**31 - 1)))
            x = torch.tensor(symbols, dtype=torch.complex64)
            with torch.inference_mode():
                received = self.block(x, n0).real.numpy()
        pilot = received[:128]
        noise_est = max(float(np.var(pilot, ddof=1)) * 2, 1e-8)
        estimate = float(10 * np.log10(max(float(np.mean(pilot)) ** 2, 1e-8) / noise_est))
        decoded = (received[128:].reshape(-1, 3).sum(axis=1) < 0).astype(np.uint8)
        raw = np.packbits(decoded).tobytes()
        body, check = raw[:-4], raw[-4:]
        ok = check == struct.pack("<I", zlib.crc32(body) & 0xffffffff)
        nbytes, latency = self.reservation(len(payload))
        return PacketResult(ok, body if ok else None, estimate, nbytes, latency, self.backend)


class ChannelBelief:
    def __init__(self):
        self.successes, self.failures = 5.0, 1.0
        self.snr_db = 8.0
        self.n_packets = 0

    def update(self, result):
        self.successes += int(result.delivered)
        self.failures += int(not result.delivered)
        self.snr_db = 0.5 * self.snr_db + 0.5 * result.pilot_snr_db
        self.n_packets += 1

    @property
    def delivery_probability(self):
        return self.successes / (self.successes + self.failures)
