"""Compare a regenerated results.json with the committed one.

Usage: python tests/compare_results.py app/results.json reproduced.json
Non-numeric fields must match exactly; floats may differ by floating-point
noise (relative 1e-9), which pyAgrum's inference produces between runs.
Exits 1 and prints the first differences otherwise.
"""
import json
import sys

TOLERANCE = 1e-9


def differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append(f"{path}/{key}: missing on one side")
            else:
                out += differences(a[key], b[key], f"{path}/{key}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, (int, float)) or isinstance(b, float) and isinstance(a, (int, float)):
        return [] if abs(a - b) <= TOLERANCE * max(1.0, abs(a), abs(b)) else [f"{path}: {a} vs {b}"]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def main(committed, reproduced):
    with open(committed, encoding="utf-8") as f:
        a = json.load(f)
    with open(reproduced, encoding="utf-8") as f:
        b = json.load(f)
    found = differences(a, b)
    for line in found[:20]:
        print(line)
    if found:
        print(f"{len(found)} differences beyond tolerance {TOLERANCE}.")
        return 1
    print(f"Reproduced: every field of {committed} matches {reproduced} (floats within {TOLERANCE} relative).")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
