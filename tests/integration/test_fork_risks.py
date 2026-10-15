"""Small reproductions of the actual fork worker write and process-return risks."""
from pathlib import Path
import importlib.util
import numpy as np
import pytest
from concurrent.futures import ProcessPoolExecutor


def write_private(array):
    array[0]=42
    # Same return contract as PR #2 cpu_bound_task: no score returned.


def test_private_process_write_not_returned():
    x=np.zeros(4)
    with ProcessPoolExecutor(max_workers=1) as pool:
        assert pool.submit(write_private,x).result() is None
    np.testing.assert_array_equal(x,np.zeros(4))


def test_cwilson_readonly_memmap(tmp_path):
    source=Path("vendor/cwilson51/statesegmentation/gsbs.py")
    if not source.exists(): pytest.skip("Optional audit checkout unavailable")
    pytest.importorskip("joblib")
    spec=importlib.util.spec_from_file_location("cwilson_audit",source)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    x=np.random.default_rng(1).normal(size=(10,4))
    p=tmp_path/"means.bin"; w=np.memmap(p,dtype=float,mode="w+",shape=x.shape); w[:]=x.mean(0); w.flush(); del w
    means=np.memmap(p,dtype=float,mode="r",shape=x.shape)
    with pytest.raises(ValueError,match="read-only"):
        mod.GSBS._wdist_state_inner_search(1,means,np.ones(10),x,mod.GSBS._zscore(x))
