from dataclasses import dataclass, field
from pathlib import Path
import json
import os
import tempfile
import numpy as np
from statesegmentation import GSBS


@dataclass
class GSBSResult:
    model: GSBS
    metadata: dict
    trace: list = field(default_factory=list)

    def __getattr__(self, name):
        # During unpickling dataclass fields have not yet been restored.
        model = self.__dict__.get("model")
        if model is None: raise AttributeError(name)
        return getattr(model, name)

    def to_statesegmentation_compatible(self):
        return self.model

    def save(self, path, *, times=None, overwrite=False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        arrays = dict(x=self.model.x, all_bounds=self.all_bounds, tdists=self.tdists,
                      nstates=np.array(self.nstates), bounds=self.bounds, deltas=self.deltas,
                      states=self.states, state_patterns=self.state_patterns, strengths=self.strengths,
                      metadata=np.array(json.dumps(self.metadata)), trace=np.array(json.dumps(self.trace)))
        if self.model.y is not None: arrays["y"]=self.model.y
        edges = np.r_[0, np.flatnonzero(self.deltas), len(self.deltas)]
        arrays["state_durations_samples"] = np.diff(edges)
        if times is not None:
            times = np.asarray(times)
            if times.shape != self.deltas.shape or not np.all(np.diff(times) > 0):
                raise ValueError("Time map must be strictly increasing and match T")
            arrays.update(times=times, boundary_timestamps=times[self.deltas.astype(bool)])
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez_compressed(f, **arrays)
            if overwrite:
                os.replace(tmp, path)
            else:
                # Atomic no-clobber hard link, supported on NTFS and Linux.
                os.link(tmp, path)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as a:
            meta = json.loads(str(a["metadata"]))
            model = GSBS(x=a["x"], y=a["y"] if "y" in a.files else None, **meta["algorithm"])
            model.all_bounds = a["all_bounds"]
            model._tdists = a["tdists"]
            model._argmax = int(a["nstates"])
            model._bounds = a["bounds"]
            model._deltas = a["deltas"].astype(bool)
            if model.all_bounds.shape != (model.kmax+2,len(model.x)):
                raise ValueError("Corrupt result shape")
            return cls(model, meta, json.loads(str(a["trace"])))
