from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from .config import ControllerConfig
from .controller import METHODS
from .provenance import write_json


def build_parser():
    parser=argparse.ArgumentParser(description="TELEMACHUS v4: observable uncertainty and resource-aware continuous navigation")
    parser.add_argument("--root",type=Path,default=Path.cwd())
    commands=parser.add_subparsers(dest="command",required=True)
    p=commands.add_parser("train-vision");p.add_argument("--epochs",type=int,default=10)
    p=commands.add_parser("episode");p.add_argument("--seed",type=int,default=100)
    p.add_argument("--scenario",default="nominal",choices=("nominal","visual","localization","dynamics","radio","compound"))
    p.add_argument("--method",choices=METHODS,default="telemachus");p.add_argument("--config",type=Path)
    p.add_argument("--output",type=Path,default=Path("results/demo.json"))
    for name in ("suite","freeze"):
        p=commands.add_parser(name);p.add_argument("--config",type=Path)
        p.add_argument("--seed-start",type=int,default=100 if name=="suite" else 1000)
        p.add_argument("--n-seeds",type=int,default=4 if name=="suite" else 16)
        p.add_argument("--methods",nargs="+",choices=METHODS,default=list(METHODS))
        p.add_argument("--output",type=Path,default=Path("results/development" if name=="suite" else "results/protocol.json"))
        if name=="suite":
            p.add_argument("--split",choices=("development","holdout"),default="development")
            p.add_argument("--protocol",type=Path)
    p=commands.add_parser("vlm-test");p.add_argument("--cache-dir",type=Path,default=Path(".cache/models"))
    p.add_argument("--local-only",action="store_true")
    p=commands.add_parser("serve");p.add_argument("--port",type=int,default=8000)
    p.add_argument("--model-cache",type=Path,default=Path(".cache/models"))
    p=commands.add_parser("plot");p.add_argument("--results",type=Path,default=Path("results/holdout"))
    return parser


def main(argv=None):
    args=build_parser().parse_args(argv);root=args.root.resolve()
    resolve=lambda p:p if p.is_absolute() else root/p
    cfg=ControllerConfig.load(resolve(args.config)) if getattr(args,"config",None) else ControllerConfig()
    if args.command=="train-vision":
        from .vision import train_vision
        result=train_vision(root/cfg.vision_artifact,epochs=args.epochs)
    elif args.command=="episode":
        from .vision import NeuralVision
        from .simulation import scenarios
        from .experiments import run_episode
        v=NeuralVision(root/cfg.vision_artifact)
        s=next(s for s in scenarios() if s.name==args.scenario)
        result=run_episode(args.seed,s,cfg,args.method,v);write_json(resolve(args.output),result)
        result={k:v for k,v in result.items() if k not in ("trace","feature_rows","grid","trajectory")}
    elif args.command=="freeze":
        from .experiments import freeze_protocol
        result=freeze_protocol(root,resolve(args.output),cfg,range(args.seed_start,args.seed_start+args.n_seeds),args.methods)
        result={"protocol_hash":result["protocol_hash"],"path":str(resolve(args.output))}
    elif args.command=="suite":
        from .experiments import run_suite
        result=run_suite(root,resolve(args.output),cfg,range(args.seed_start,args.seed_start+args.n_seeds),args.methods,args.split,
                         protocol_path=resolve(args.protocol) if args.protocol else None)
        result={"n_episodes":result["n_episodes"],"methods":result["methods"]}
    elif args.command=="vlm-test":
        from .vlm import run_vlm_validation
        result=run_vlm_validation(root,resolve(args.cache_dir),args.local_only)
    elif args.command=="serve":
        from .service import create_app
        import uvicorn
        uvicorn.run(create_app(root,resolve(args.model_cache)),host="127.0.0.1",port=args.port);return
    elif args.command=="plot":
        from .visuals import create_plots
        result=create_plots(resolve(args.results))
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":main()
