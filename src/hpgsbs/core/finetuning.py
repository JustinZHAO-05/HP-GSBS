import numpy as np
from .state_utils import boundary_strengths


def finetune(model, search):
    bounds = model._bounds.copy()
    strengths = boundary_strengths(model.x,model.all_bounds[int(np.sum(bounds>0))+1])
    order = np.argsort(strengths if model.finetune_order else -strengths)
    for ind in order[strengths[order] != 0]:
        state_id = bounds[ind]
        bounds[ind] = 0
        opts = np.arange(1,len(bounds)) if model.finetune < 0 else np.arange(max(1,ind-model.finetune),min(len(bounds),ind+model.finetune+1))
        pos,_ = search.best(bounds>0,boundopt=opts)
        bounds[pos] = state_id
    return bounds
