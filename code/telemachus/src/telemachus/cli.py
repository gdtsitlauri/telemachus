from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from .config import ControllerConfig
from .controller import METHODS
from .provenance import write_json


def build_parser():
    parser=argparse.ArgumentParser(description="TELEMACHUS v5: learned evidence acquisition with certified continuous control")
    parser.add_argument("--root",type=Path,default=Path.cwd())
    commands=parser.add_subparsers(dest="command",required=True)
    p=commands.add_parser("train-vision");p.add_argument("--epochs",type=int,default=10)
    p=commands.add_parser("episode");p.add_argument("--seed",type=int,default=100)
    p.add_argument("--scenario",default="nominal",choices=("nominal","visual","localization","dynamics","radio","compound"))
    p.add_argument("--method",choices=METHODS,default="telemachus");p.add_argument("--config",type=Path)
    p.add_argument("--output",type=Path,default=Path("results/demo.json"))
    for name in ("suite","freeze"):
        p=commands.add_parser(name);p.add_argument("--config",type=Path)
        p.add_argument("--seed-start",type=int,default=100 if name=="suite" else 12000)
        p.add_argument("--n-seeds",type=int,default=4 if name=="suite" else 16)
        p.add_argument("--methods",nargs="+",choices=METHODS,default=list(METHODS))
        p.add_argument("--output",type=Path,default=Path("results/development" if name=="suite" else "results/protocol.json"))
        p.add_argument("--split",choices=("development","holdout","stress"),default="development" if name=="suite" else "holdout")
        if name=="suite":
            p.add_argument("--protocol",type=Path)
            p.add_argument("--workers",type=int,default=1)
    p=commands.add_parser("collect-policy");p.add_argument("--seed-start",type=int,default=2000)
    p.add_argument("--n-seeds",type=int,default=40);p.add_argument("--max-states",type=int,default=64)
    p.add_argument("--workers",type=int,default=1)
    p.add_argument("--output",type=Path,default=Path("artifacts/resource_training_data.json"))
    p=commands.add_parser("train-policy");p.add_argument("--dataset",type=Path,default=Path("artifacts/resource_training_data.json"))
    p.add_argument("--output",type=Path,default=Path("artifacts/resource_policy.json"))
    p=commands.add_parser("vlm-test");p.add_argument("--cache-dir",type=Path,default=Path(".cache/models"))
    p.add_argument("--local-only",action="store_true")
    p=commands.add_parser("serve");p.add_argument("--port",type=int,default=8000)
    p.add_argument("--model-cache",type=Path,default=Path(".cache/models"))
    p=commands.add_parser("plot");p.add_argument("--results",type=Path,default=Path("results/holdout"))
    # v6 certified protocol: freeze -> calibrate (Learn-then-Test) -> test.
    p=commands.add_parser("certify-freeze");p.add_argument("--config",type=Path)
    p.add_argument("--methods",nargs="+",choices=METHODS,required=True)
    p.add_argument("--calibration-seed-start",type=int,default=30000);p.add_argument("--calibration-n-seeds",type=int,default=60)
    p.add_argument("--test-seed-start",type=int,default=40000);p.add_argument("--test-n-seeds",type=int,default=80)
    p.add_argument("--alpha",type=float,default=.03);p.add_argument("--delta",type=float,default=.1)
    p.add_argument("--output",type=Path,default=Path("results/v6/protocol.json"))
    p=commands.add_parser("certify-calibrate");p.add_argument("--protocol",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("results/v6/calibration"));p.add_argument("--workers",type=int,default=1)
    p=commands.add_parser("certify-test");p.add_argument("--protocol",type=Path,required=True)
    p.add_argument("--certificate",type=Path,required=True)
    p.add_argument("--split",choices=("certified","certified_stress"),default="certified")
    p.add_argument("--output",type=Path,default=Path("results/v6/test"));p.add_argument("--workers",type=int,default=1)
    return parser


def main(argv=None):
    args=build_parser().parse_args(argv);root=args.root.resolve()
    resolve=lambda p:p if p.is_absolute() else root/p
    cfg=ControllerConfig.load(resolve(args.config)) if getattr(args,"config",None) else ControllerConfig.load(root/"configs/main.json") if (root/"configs/main.json").exists() else ControllerConfig()
    if args.command=="train-vision":
        from .vision import train_vision
        result=train_vision(root/cfg.vision_artifact,epochs=args.epochs)
    elif args.command=="episode":
        from .vision import NeuralVision
        from .simulation import scenarios
        from .experiments import run_episode
        v=NeuralVision(root/cfg.vision_artifact)
        s=next(s for s in scenarios() if s.name==args.scenario)
        result=run_episode(args.seed,s,cfg,args.method,v,root=root);write_json(resolve(args.output),result)
        result={k:v for k,v in result.items() if k not in ("trace","feature_rows","grid","trajectory")}
    elif args.command=="freeze":
        from .experiments import freeze_protocol
        result=freeze_protocol(root,resolve(args.output),cfg,range(args.seed_start,args.seed_start+args.n_seeds),args.methods,split=args.split)
        result={"protocol_hash":result["protocol_hash"],"path":str(resolve(args.output))}
    elif args.command=="suite":
        from .experiments import run_suite
        result=run_suite(root,resolve(args.output),cfg,range(args.seed_start,args.seed_start+args.n_seeds),args.methods,args.split,
                         protocol_path=resolve(args.protocol) if args.protocol else None,workers=args.workers)
        result={"n_episodes":result["n_episodes"],"methods":result["methods"]}
    elif args.command=="collect-policy":
        from .resource_policy import collect_dataset
        cfg=replace(cfg,resource_policy_mode="heuristic")
        data=collect_dataset(root,cfg,range(args.seed_start,args.seed_start+args.n_seeds),resolve(args.output),max_states_per_episode=args.max_states,workers=args.workers)
        result={"n_states":len(data["rows"]),"wall_seconds":data["wall_seconds"]}
    elif args.command=="train-policy":
        from .resource_policy import train_policy
        result=train_policy(resolve(args.dataset),resolve(args.output),range(2000,2028),range(2028,2034),range(2034,2040))
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
    elif args.command=="certify-freeze":
        from .certified import freeze
        from .experiments import DEVELOPMENT_SEEDS
        m=freeze(root,resolve(args.output),cfg,args.methods,
                 range(args.calibration_seed_start,args.calibration_seed_start+args.calibration_n_seeds),
                 range(args.test_seed_start,args.test_seed_start+args.test_n_seeds),args.alpha,args.delta,
                 development_seeds=DEVELOPMENT_SEEDS)
        result={"protocol_hash":m["protocol_hash"],"path":str(resolve(args.output))}
    elif args.command=="certify-calibrate":
        from .certified import calibrate
        r=calibrate(root,resolve(args.protocol),resolve(args.output),args.workers)
        result={m:{k:v[k] for k in ("selected_setting","no_certificate")} for m,v in r["methods"].items()}
    elif args.command=="certify-test":
        from .certified import test
        r=test(root,resolve(args.protocol),resolve(args.certificate),resolve(args.output),args.workers,args.split)
        result=r["certified_methods"]
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":main()
