from pathlib import Path
import os
import numpy as np
import pytest
from hpgsbs import fit, GSBSConfig, GSBSResult
from hpgsbs.validation import compare


@pytest.mark.parametrize("index",range(4))
def test_frozen_eeg_development(index):
    path=Path(os.environ.get("HPGSBS_FIXTURES","results/fixtures"))/f"{index}_dev96_reference.npz"
    if not path.exists(): pytest.skip("Run scripts/freeze_reference.py first")
    ref=GSBSResult.load(path)
    fast=fit(ref.x,GSBSConfig(**ref.metadata["algorithm"],trace=True))
    report=compare(ref,fast)
    assert report["passed"], report


@pytest.mark.parametrize("kind",["low_noise","high_noise","correlated","short","near_tie","finetune"])
def test_structured(kind):
    rng=np.random.default_rng(41)
    p=8; t=36
    means=rng.normal(size=(4,p)); labels=np.repeat(np.arange(4),9)
    if kind=="short": labels[:5]=[0,1,2,3,1]
    x=means[labels]+rng.normal(size=(t,p))*(2 if kind=="high_noise" else .1)
    if kind=="correlated": x=x[:,[0]]+x*1e-3
    if kind=="near_tie": x=np.tile(means[0],(t,1))+rng.normal(size=(t,p))*1e-9
    c=GSBSConfig(10,blocksize=7,trace=True,finetune=2 if kind=="finetune" else 1)
    report=compare(fit(x,c,"reference"),fit(x,c))
    assert report["passed"], report


def test_ill_conditioned_spatial_pattern_fallback():
    x=1e10+np.random.default_rng(12).normal(size=(16,5))*1e-5
    c=GSBSConfig(7,trace=True)
    ref=fit(x,c,"reference"); fast=fit(x,c)
    assert fast.metadata["backend"]=="reference"
    assert fast.metadata["requested_backend"]=="cpu-fast"
    assert compare(ref,fast)["passed"]


def test_exact_cancellation_interval():
    x=np.tile(np.array([[1.,2.,3.],[-1.,-2.,-3.]]),(6,1))
    c=GSBSConfig(7,trace=True)
    assert compare(fit(x,c,"reference"),fit(x,c))["passed"]
