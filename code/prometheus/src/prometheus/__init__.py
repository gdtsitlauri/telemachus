"""PROMETHEUS: resource-aware lifelong collective learning research prototype."""
from .benchmark import BenchmarkSpec, ContinualCoordinationBandit
from .train import MethodConfig, run_method
from .long_horizon import LongHorizonSpec, ContinualSignalGrid
from .long_train import LongMethodConfig, run_long_method

__version__ = "2.0.0"
__all__ = [
    "BenchmarkSpec", "ContinualCoordinationBandit", "MethodConfig", "run_method",
    "LongHorizonSpec", "ContinualSignalGrid", "LongMethodConfig", "run_long_method",
]
