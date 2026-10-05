from pathlib import Path
import json
import numpy as np
import pytest
from telemachus.belief import MapBelief,ObservableDiagnosis,FEATURE_NAMES
from telemachus.config import ControllerConfig
from telemachus.experiments import freeze_protocol,verify_protocol


ROOT=Path(__file__).resolve().parents[1]


def test_config_roundtrip_matches_persisted_protocol():
    cfg=ControllerConfig()
    assert cfg.to_dict()==json.loads(json.dumps(cfg.to_dict()))
    assert ControllerConfig.from_dict(cfg.to_dict())==cfg


def test_low_quality_frame_cannot_erase_clean_obstacle_evidence():
    m=MapBelief(20,.5);centers=np.array([[3.25,3.25]])
    m._fuse(centers,np.array([.999]),[.75,.75,0],True,1.)
    for _ in range(4):m._fuse(centers,np.array([.30]),[.75,.75,0],False,.3)
    assert m.probability[6,6]>.9


def test_adapted_gain_is_not_mistaken_for_remaining_model_error():
    f=np.zeros(len(FEATURE_NAMES));f[FEATURE_NAMES.index("gain_deviation")]=.5
    f[FEATURE_NAMES.index("gain_uncertainty")]=.025
    assert ObservableDiagnosis().predict(f)["model"]<.2


def test_frozen_protocol_rejects_changed_sources_and_changed_model(tmp_path):
    (tmp_path/"src").mkdir();(tmp_path/"configs").mkdir();(tmp_path/"artifacts").mkdir()
    (tmp_path/"src/core.py").write_text("value=1\n")
    (tmp_path/"artifacts/vision_ensemble.pt").write_bytes(b"model1")
    protocol=freeze_protocol(tmp_path,tmp_path/"protocol.json",ControllerConfig(),[11000],["telemachus"])
    verify_protocol(tmp_path,protocol)
    (tmp_path/"src/core.py").write_text("value=2\n")
    with pytest.raises(ValueError,match="Source/config changed"):verify_protocol(tmp_path,protocol)
