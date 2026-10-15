"""Call official GSBS unchanged. Optional tracing observes locals, never patches code."""
import inspect
import hashlib
import sys
from importlib.metadata import version
from time import perf_counter
from pathlib import Path
import numpy as np
from statesegmentation import GSBS
from .result import GSBSResult


def provenance():
    p = Path(inspect.getfile(GSBS))
    return {"version": version("statesegmentation"), "source_sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def fit_reference(x, config, y=None):
    model = GSBS(x=x, y=y, **config.algorithm())
    trace, state = [], {}
    source, line0 = inspect.getsourcelines(GSBS.fit)
    lines = {line0+i: s.strip() for i,s in enumerate(source)}

    def observer(frame, event, arg):
        if frame.f_code is not GSBS.fit.__code__:
            return None
        if event == "line":
            line, v = lines.get(frame.f_lineno, ""), frame.f_locals
            if line == "states = self._states(self._deltas)":
                state["before"] = model._bounds.tolist()
            if line.startswith("if self.finetune !="):
                state["pre_finetune"] = model._bounds.tolist()
            if line == "k = k + increment":
                ws = v.get("wdists_s")
                double = np.unravel_index(ws.argmax(),ws.shape) if isinstance(ws,np.ndarray) else None
                trace.append(dict(k=int(v["k"]), before=state["before"],
                    single=int(v["argmax"]), double=list(map(int,double)) if double else None,
                    single_score=float(v["wdists"].max()), double_score=float(ws.max()) if double else None,
                    single_t=float(v["tdist"]), double_t=float(v["tdist_s"]) if double else None,
                    increment=int(v["increment"]), pre_finetune=state["pre_finetune"], after=model._bounds.tolist()))
        return observer

    start = perf_counter()
    old = sys.gettrace()
    try:
        if config.trace: sys.settrace(observer)
        model.fit(showProgressBar=False)
    finally:
        if config.trace: sys.settrace(old)
    return GSBSResult(model, dict(backend="reference", precision="float64", input_shape=list(x.shape),
        elapsed_s=perf_counter()-start, status="success", algorithm=config.algorithm(),
        tdistance_input="x" if y is None else "y",y_shape=None if y is None else list(y.shape),
        upstream=provenance(), compatibility="official-unmodified"), trace)
