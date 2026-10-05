from pathlib import Path
from telemachus.config import ControllerConfig
from telemachus.controller import METHODS
from telemachus.experiments import run_suite

if __name__=="__main__":
    root=Path(__file__).resolve().parents[1]
    report=run_suite(root,root/"results"/"development",ControllerConfig.load(root/"configs/main.json"),range(2300,2308),METHODS,workers=4)
    for name,r in report["methods"].items():
        print(name,{k:r[k] for k in ("success","collision","deferred","cost_mean")})
