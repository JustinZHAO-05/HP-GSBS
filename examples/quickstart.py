"""A complete, dataset-free example: python examples/quickstart.py."""
import argparse
from pathlib import Path
import numpy as np
from hpgsbs import GSBSConfig, GSBSResult, fit, fit_many
from hpgsbs.validation import compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cpu-fast")
    parser.add_argument("--output", default="examples/generated")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    patterns = rng.normal(size=(4, 8))
    x = np.repeat(patterns, 20, axis=0) + 0.15 * rng.normal(size=(80, 8))
    config = GSBSConfig(kmax=12, dmin=1, dtype="float64", q_strategy="full",
                        candidate_backend="streaming", td_backend="prefix2d", trace=True)
    result = fit(x, config, backend=args.backend)
    reference = fit(x, config, backend="reference")
    report = compare(reference, result, level="finetune-move-trace")
    print("Reference parity:", report)
    if not report["passed"]:
        raise RuntimeError("Reference parity failed; inspect the first divergence above")
    # Result.save refuses to overwrite an existing result.
    result.save(out / "fit.npz")
    restored=GSBSResult.load(out / "fit.npz")
    assert compare(result,restored,level="output-only")["passed"]
    batch=fit_many([x,x.copy()],config,backend=args.backend,batch_size=2)
    assert all(compare(result,r,level="finetune-move-trace")["passed"] for r in batch)
    np.save(out / "input.npy", x)
    print("Selected states:", result.nstates)
    print("Boundary indices:", np.flatnonzero(result.deltas).tolist())
    print("K=6 boundaries:", np.flatnonzero(result.get_bounds(6)).tolist())
    print("Full t-distance curve length:", len(result.tdists))
    print("Saved input.npy and fit.npz to", out.resolve())


if __name__ == "__main__":
    main()
