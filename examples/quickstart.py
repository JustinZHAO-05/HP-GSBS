"""A complete, dataset-free example: python examples/quickstart.py."""
import argparse
from pathlib import Path
import numpy as np
from hpgsbs import GSBSConfig, fit
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
    config = GSBSConfig(kmax=12, trace=True)
    result = fit(x, config, backend=args.backend)
    reference = fit(x, config, backend="reference")
    report = compare(reference, result)
    print("Reference parity:", report)
    if not report["passed"]:
        raise RuntimeError("Reference parity failed; inspect the first divergence above")
    # Result.save refuses to overwrite an existing result.
    result.save(out / "fit.npz")
    np.save(out / "input.npy", x)
    print("Selected states:", result.nstates)
    print("Boundary indices:", np.flatnonzero(result.deltas).tolist())
    print("Saved input.npy and fit.npz to", out.resolve())


if __name__ == "__main__":
    main()
