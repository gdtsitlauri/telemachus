from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import base64
import io
from typing import Literal
import numpy as np
from pydantic import BaseModel,Field,field_validator
from fastapi import FastAPI,HTTPException
from . import __version__
from .config import ControllerConfig
from .vision import NeuralVision
from .controller import METHODS
from .simulation import scenarios
from .experiments import run_episode


class EpisodeRequest(BaseModel):
    seed:int=Field(default=100,ge=0,le=2**31-1)
    scenario:Literal["nominal","visual","localization","dynamics","radio","compound"]="nominal"
    method:Literal["legacy_v4","nominal","confidence_gate","budget_voi","telemachus","no_learning","no_reserve","no_source","no_model_update","fixed_compute"]="telemachus"
    max_cost:float=Field(default=120.,gt=0,le=200.)
    max_steps:int=Field(default=160,ge=1,le=200)


class PerceptionRequest(BaseModel):
    tiles:list[list[list[list[float]]]]

    @field_validator("tiles")
    @classmethod
    def validate_tiles(cls,value):
        x=np.asarray(value,float)
        if x.ndim!=4 or x.shape[1:]!=(3,12,12) or not 1<=len(x)<=128:
            raise ValueError("Supply 1 to 128 RGB tiles, N x 3 x 12 x 12")
        if not np.isfinite(x).all() or x.min()<0 or x.max()>1:
            raise ValueError("Pixels must be finite and in [0,1]")
        return value


class MissionRequest(BaseModel):
    image_base64:str=Field(min_length=8,max_length=350000)
    instruction:str=Field(default="Identify the red landing pad",min_length=3,max_length=300)
    left_goal:tuple[float,float]=(8.25,9.25)
    right_goal:tuple[float,float]=(9.25,9.25)

    @field_validator("left_goal","right_goal")
    @classmethod
    def finite_goal(cls,value):
        if not all(np.isfinite(x) and .25<=x<=9.75 for x in value):
            raise ValueError("Goal coordinates must lie inside the simulated world")
        return value


def create_app(root,model_cache=None):
    root=Path(root)
    app=FastAPI(title="TELEMACHUS v5 research simulator",version=__version__)
    @lru_cache(maxsize=1)
    def vision():return NeuralVision(root/ControllerConfig().vision_artifact)
    @lru_cache(maxsize=1)
    def resolver():
        from .vlm import VisualMissionResolver
        return VisualMissionResolver(cache_dir=model_cache or root/".cache/models")

    @app.get("/health")
    def health():
        return {"status":"ok","version":__version__,"runtime":"CPU","scope":"simulated research"}

    @app.post("/perceive")
    def perceive(request:PerceptionRequest):
        p,v,e=vision().predict(np.asarray(request.tiles,np.float32))
        return {"occupancy_probability":p.tolist(),"ensemble_variance":v.tolist(),"entropy":e.tolist()}

    @app.post("/episode")
    def episode(request:EpisodeRequest):
        from dataclasses import replace
        cfg=ControllerConfig.load(root/"configs/main.json")
        cfg=replace(cfg,max_steps=request.max_steps,budget=replace(cfg.budget,max_cost=request.max_cost))
        scenario=next(s for s in scenarios() if s.name==request.scenario)
        result=run_episode(request.seed,scenario,cfg,request.method,vision(),root=root)
        # Evaluation truth is returned by this simulator endpoint, never fed
        # into the agent. No endpoint commands real hardware.
        return result

    @app.post("/ground-mission")
    def ground_mission(request:MissionRequest):
        from PIL import Image,UnidentifiedImageError
        try:
            raw=base64.b64decode(request.image_base64,validate=True)
            image=Image.open(io.BytesIO(raw))
            if image.width*image.height>512*512 or min(image.size)<12:
                raise ValueError("Use an image between 12 pixels and 512 x 512")
            image.load()
            image=image.convert("RGB")
        except (ValueError,UnidentifiedImageError,OSError) as exc:
            raise HTTPException(422,"Invalid mission image") from exc
        result=resolver().resolve(image,request.instruction)
        if not result["accepted"]:
            raise HTTPException(422,{"reason":"ambiguous_language_answer","raw_answer":result["raw_answer"]})
        result["goal"]=list(request.left_goal if result["landmark_id"]=="left" else request.right_goal)
        return result
    return app
