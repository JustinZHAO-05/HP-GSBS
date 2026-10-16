import numpy as np
from .state_utils import boundary_strengths


def finetune(model, search, trace=None):
    bounds = model._bounds.copy()
    strengths = boundary_strengths(model.x,model.all_bounds[int(np.sum(bounds>0))+1])
    order = np.argsort(strengths if model.finetune_order else -strengths)
    for ind in order[strengths[order] != 0]:
        state_id = bounds[ind]
        bounds[ind] = 0
        lo,hi = (1,len(bounds)) if model.finetune < 0 else (max(1,ind-model.finetune),min(len(bounds),ind+model.finetune+1))
        pos,_ = search.window(bounds,lo,hi)
        if trace is not None:
            trace.append(dict(deleted=int(ind),window=[int(lo),int(hi)],winner=int(pos),
                              birth=int(state_id),after_delete=bounds.tolist()))
        bounds[pos] = state_id
        if trace is not None: trace[-1]["after_insert"]=bounds.tolist()
    return bounds
