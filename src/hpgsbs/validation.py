import numpy as np


def compare(reference, candidate, atol=1e-8, rtol=1e-8):
    report={"bounds_equal": bool(np.array_equal(reference.all_bounds,candidate.all_bounds)),
            "nstates_equal": bool(reference.nstates==candidate.nstates), "first_divergence": None}
    a,b=reference.tdists,candidate.tdists
    if a.shape!=b.shape:
        report.update(tdists_close=False,max_t_error=None)
    else:
        report["tdists_close"]=bool(np.allclose(a,b,atol=atol,rtol=rtol,equal_nan=True))
        valid=np.isfinite(a)&np.isfinite(b)
        report["max_t_error"]=float(np.max(np.abs(a[valid]-b[valid]))) if valid.any() else None
    report["patterns_close"]=reference.state_patterns.shape==candidate.state_patterns.shape and bool(np.allclose(reference.state_patterns,candidate.state_patterns,atol=atol,rtol=rtol,equal_nan=True))
    report["strengths_close"]=reference.strengths.shape==candidate.strengths.shape and bool(np.allclose(reference.strengths,candidate.strengths,atol=atol,rtol=rtol,equal_nan=True))
    report["trace_compared"]=bool(reference.trace and candidate.trace)
    report["trace_values_close"]=True
    report["first_numerical_divergence"]=None
    if report["trace_compared"]:
        for a,b in zip(reference.trace,candidate.trace):
            keys = ["k","before","single","double","increment","pre_finetune","after"]
            if "finetune_moves" in a and "finetune_moves" in b: keys.append("finetune_moves")
            for key in keys:
                if a[key]!=b[key]:
                    report["first_divergence"]={"k":a["k"],"field":key,"reference":a[key],"candidate":b[key]}
                    break
            if report["first_divergence"]: break
        if len(reference.trace)!=len(candidate.trace) and not report["first_divergence"]:
            report["first_divergence"]={"field":"trace_length"}
        for a,b in zip(reference.trace,candidate.trace):
            for key in ("single_score","double_score","single_t","double_t"):
                av,bv=a[key],b[key]
                close=av is bv if av is None or bv is None else bool(np.isclose(av,bv,atol=atol,rtol=rtol,equal_nan=True))
                if not close:
                    report["trace_values_close"]=False
                    if report["first_numerical_divergence"] is None:
                        report["first_numerical_divergence"]={"k":a["k"],"field":key,"reference":av,"candidate":bv}
    report["passed"]=all(report[k] for k in ("bounds_equal","nstates_equal","tdists_close","patterns_close","strengths_close","trace_values_close")) and report["first_divergence"] is None
    return report
