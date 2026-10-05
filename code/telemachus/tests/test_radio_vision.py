from pathlib import Path
import numpy as np
import pytest
from telemachus.radio import RadioLink
from telemachus.vision import NeuralVision,render_tiles


ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("backend",["analytic","sionna"])
def test_phy_decodes_crc_and_uses_received_pilot(backend):
    if backend=="sionna":pytest.importorskip("sionna")
    radio=RadioLink(backend);payload=b"measured evidence"*20
    a=radio.transmit(payload,10,np.random.default_rng(8))
    b=radio.transmit(payload,10,np.random.default_rng(8))
    assert a.delivered and a.payload==payload
    assert a.pilot_snr_db==b.pilot_snr_db
    assert 7<a.pilot_snr_db<13 and a.bytes_sent>len(payload)
    bad=radio.transmit(payload,-12,np.random.default_rng(8))
    assert not bad.delivered and bad.payload is None


def test_neural_rgb_perception_receives_no_labels():
    y=np.tile([False,True],20);x=render_tiles(y,np.random.default_rng(18))
    v=NeuralVision(ROOT/"artifacts/vision_ensemble.pt");p,var,entropy=v.predict(x)
    assert np.mean((p>=.5)==y)>.9
    assert np.isfinite(p).all() and (var>=0).all() and (entropy>=0).all()
    with pytest.raises(ValueError):v.predict(np.full((1,3,12,12),np.nan))
