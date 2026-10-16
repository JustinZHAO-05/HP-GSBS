"""Verify final evidence completeness and measured-source identities."""
import hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    base=Path("results/v02")
    read=lambda p:json.loads((base/p).read_text())
    provenance=read("accepted/provenance.json")
    assert len(provenance["jobs"])==4 and all(r["returncode"]==0 for r in provenance["jobs"])
    for name,digest in provenance["source_hashes"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    pair=read("accepted/paired/trials.json")
    assert len(pair)==120 and all(r["report"]["passed"] for r in pair)
    throughput=read("accepted/suite/throughput.json")
    assert all(r["all_outputs_passed"] for r in throughput if "iteration" in r)
    real=read("real_accepted/validation.json")
    assert len(real)==16 and all(r["report"]["passed"] for r in real)
    long=read("accepted/long/validation.json")
    assert len(long)==4 and all(c["passed"] for r in long for c in r["checks"])
    assert all(not r["full_official_trace_certified"] for r in long)
    tests={}
    for platform in ("windows","wsl"):
        suite=ET.parse(base/f"accepted_tests_{platform}.xml").getroot().find("testsuite")
        assert int(suite.attrib["failures"])==int(suite.attrib["errors"])==0
        tests[platform]={k:int(suite.attrib[k]) for k in ("tests","failures","errors","skipped")}
    assert not list(base.rglob("*.npy")) and not list(base.rglob("*.npz"))
    report=dict(source_revision=provenance["revision"],source_hashes_match=True,
        paired_full_fits=len(pair),paired_all_passed=True,
        throughput_trials=sum("iteration" in r for r in throughput),throughput_all_passed=True,
        private_real_comparisons=len(real),private_real_all_passed=True,
        long_n_shapes=[r["shape"] for r in long],long_n_official_full_trace_certified=False,
        tests=tests,no_raw_eeg_numpy_files=True)
    (base/"delivery_checks.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
