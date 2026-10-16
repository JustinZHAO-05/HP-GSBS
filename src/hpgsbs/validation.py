import numpy as np


def compare(reference, candidate, atol=1e-8, rtol=1e-8, *, level=None):
    aliases={"outputs-only":"output-only","iteration-trace":"full-search-trace"}
    level=aliases.get(level,level)
    if level not in (None,"output-only","full-search-trace","finetune-move-trace","numeric-stress"):
        raise ValueError("Unknown validation level")
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
    required=level in ("full-search-trace","finetune-move-trace","numeric-stress")
    has_trace=bool(reference.trace and candidate.trace)
    report["validation_level"]=level or "auto-available-trace"
    report["trace_compared"]=has_trace and level!="output-only"
    report["moves_compared"]=False
    report["trace_skipped_reason"]=None if report["trace_compared"] else "trace missing or output-only requested"
    report["moves_skipped_reason"]="per-move fields unavailable or not requested"
    report["required_evidence_present"]=not required or has_trace
    report["trace_values_close"]=True
    report["first_numerical_divergence"]=None
    if report["trace_compared"]:
        fields={"k","before","single","double","increment","pre_finetune","after",
                "single_score","double_score","single_t","double_t"}
        if any(not fields.issubset(row) for row in reference.trace+candidate.trace):
            report.update(required_evidence_present=False,trace_values_close=False,
                trace_skipped_reason="malformed search trace")
            report["passed"]=False
            return report
        move_fields={"deleted","window","winner","birth","after_delete","after_insert"}
        moves=all("finetune_moves" in row and isinstance(row["finetune_moves"],list) and
            all(move_fields.issubset(m) for m in row["finetune_moves"])
            for row in reference.trace+candidate.trace)
        report["moves_compared"]=moves
        if moves: report["moves_skipped_reason"]=None
        if level=="finetune-move-trace" and not moves: report["required_evidence_present"]=False
        for a,b in zip(reference.trace,candidate.trace):
            keys = ["k","before","single","double","increment","pre_finetune","after"]
            if moves: keys.append("finetune_moves")
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
    report["passed"]=all(report[k] for k in ("bounds_equal","nstates_equal","tdists_close","patterns_close","strengths_close","trace_values_close","required_evidence_present")) and report["first_divergence"] is None
    return report
