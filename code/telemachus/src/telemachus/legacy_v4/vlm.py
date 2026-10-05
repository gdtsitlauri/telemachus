from __future__ import annotations

from pathlib import Path
import hashlib
import re
import time
import numpy as np
from PIL import Image, ImageDraw
from .provenance import write_json, utc_now


MODEL_ID = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
MODEL_REVISION = "067788b187b95ebe7b2e040b3e4299e342e5b8fd"


def landmark_scene(red_left=True):
    image=Image.new("RGB",(320,160),(238,238,233))
    d=ImageDraw.Draw(image)
    colors=[(230,25,25),(25,60,220)] if red_left else [(25,60,220),(230,25,25)]
    for x,color in zip((25,185),colors):
        d.rectangle((x,25,x+110,135),fill=color,outline=(40,40,40),width=3)
    return image


class VisualMissionResolver:
    """Actual pretrained VLM for mission grounding, outside the safety loop.

    The model translates a visual/language mission into a constrained landmark
    ID. The continuous controller verifies its own actions independently. An
    unparseable model answer is rejected; there is no hidden oracle fallback.
    """
    def __init__(self, cache_dir=None, threads=2, local_files_only=False):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        torch.set_num_threads(threads)
        started=time.perf_counter()
        self.processor=AutoProcessor.from_pretrained(MODEL_ID,revision=MODEL_REVISION,
                         cache_dir=cache_dir,trust_remote_code=False,local_files_only=local_files_only)
        self.model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,revision=MODEL_REVISION,
                     cache_dir=cache_dir,trust_remote_code=False,local_files_only=local_files_only,
                     dtype=torch.float32,attn_implementation="sdpa")
        self.model.eval()
        self.load_wall_seconds=time.perf_counter()-started

    def resolve(self, image, instruction="Identify the red landing pad", max_new_tokens=16):
        import torch
        if not isinstance(image,Image.Image):
            raise TypeError("Expected a PIL image")
        question=instruction+". Is it on the left or on the right of the image? Answer with only left or right."
        messages=[{"role":"user","content":[{"type":"image","image":image},
                                              {"type":"text","text":question}]}]
        inputs=self.processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,
                         return_dict=True,return_tensors="pt")
        started=time.perf_counter()
        with torch.inference_mode():
            output=self.model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False)
        raw=self.processor.batch_decode(output[:,inputs["input_ids"].shape[1]:],skip_special_tokens=True)[0].strip()
        choices=set(re.findall(r"\b(left|right)\b",raw.lower()))
        side=next(iter(choices)) if len(choices)==1 else None
        return {"landmark_id":side,"accepted":side is not None,"raw_answer":raw,
                "inference_wall_seconds":time.perf_counter()-started,
                "model":MODEL_ID,"revision":MODEL_REVISION,
                "image_sha256":hashlib.sha256(image.convert("RGB").tobytes()).hexdigest(),
                "instruction":instruction,"role":"visual mission grounding, not low-level control"}


def run_vlm_validation(root,cache_dir,local_files_only=False):
    root=Path(root);out=root/"results"/"vlm";out.mkdir(parents=True,exist_ok=True)
    resolver=VisualMissionResolver(cache_dir,local_files_only=local_files_only)
    from .vision import NeuralVision
    from .config import ControllerConfig
    from .simulation import scenarios
    from .experiments import run_episode
    cfg=ControllerConfig(channel_backend="sionna")
    vision=NeuralVision(root/cfg.vision_artifact)
    cases=[]
    for red_left in (True,False):
        for color in ("red","blue"):
            image=landmark_scene(red_left)
            image_name=f"scene_red_{'left' if red_left else 'right'}.png"
            image.save(out/image_name)
            result=resolver.resolve(image,f"Identify the {color} landing pad")
            expected="left" if red_left==(color=="red") else "right"
            result.update({"expected":expected,"correct":result["landmark_id"]==expected,"image_file":image_name})
            # Landmark coordinates come from mission metadata. The VLM must
            # select the ID; expected labels are used only AFTER inference.
            landmarks={"left":[8.25,9.25],"right":[9.25,9.25]}
            if result["accepted"]:
                nav=run_episode(200+len(cases),scenarios()[len(cases)%len(scenarios())],cfg,"telemachus",vision,
                                goal=landmarks[result["landmark_id"]])
                write_json(out/f"mission_{len(cases)}.json",nav)
                result["navigation_outcome"]=nav["outcome"]
                result["mission_success"]=result["correct"] and nav["outcome"]=="success"
                result["navigation_wall_seconds"]=nav["wall_seconds"]
                result["navigation_resources"]=nav["resources"]
            else:
                result["navigation_outcome"]="rejected_language_answer"
                result["mission_success"]=False
            cases.append(result)
            print(f"VLM {red_left=} {color}: {result['raw_answer']!r}, correct={result['correct']}",flush=True)
    report={"created":utc_now(),"model":MODEL_ID,"revision":MODEL_REVISION,"cases":cases,
            "n":len(cases),"accuracy":float(np.mean([r["correct"] for r in cases])),
            "end_to_end_mission_success":float(np.mean([r["mission_success"] for r in cases])),
            "load_wall_seconds":resolver.load_wall_seconds,
            "scope":"four synthetic two-landmark mission tests; not general VLM accuracy"}
    write_json(out/"validation.json",report)
    return report
