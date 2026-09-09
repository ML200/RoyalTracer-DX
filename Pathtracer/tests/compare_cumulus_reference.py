"""Require bit-identical primary cloud output against the recovered slow renderer.

Run both run_cumulus_tests.ps1 modes at the same resolution first. Comparison covers
all 16 float channels, including radiance, transmittance and every diagnostic guide.
Only Python's standard library is required; this does not compare secondary quality.
"""
import argparse
import hashlib
from pathlib import Path


CASES = (
    "ground", "sample0", "sample1", "dawn", "dusk", "finite-before-cloud",
    "wind", "above", "rebased", "side", "inside", "high-altitude", "disabled",
    "zero-coverage",
)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.digest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    failures = []
    for name in CASES:
        reference = args.reference / (name + ".bin")
        candidate = args.candidate / (name + ".bin")
        if not reference.is_file() or not candidate.is_file():
            failures.append(name + ": missing render")
            continue
        size = reference.stat().st_size
        if not size or size % 64 or candidate.stat().st_size != size:
            failures.append(name + ": invalid or mismatched dimensions")
        elif digest(reference) != digest(candidate):
            failures.append(name + ": primary output changed")
        else:
            print("PASS " + name + ": all 16 channels bit-identical")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"All {len(CASES)} primary cloud cases match the slow reference exactly.")


if __name__ == "__main__":
    main()
