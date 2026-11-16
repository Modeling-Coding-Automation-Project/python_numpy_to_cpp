#!/usr/bin/env python3
"""Compare fixed threshold bases using unchanged regression assertions.

Production headers are copied to a temporary directory. Only library epsilon
calls are substituted; test tolerances and the worktree remain unchanged.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

from run_epsilon_gmres_sweep import DEFAULT_REVISION, snapshot


ROOT = Path(__file__).resolve().parents[2]
SOURCES = Path(__file__).resolve().parent
VALUES = ["machine"] + ["1e-" + str(power) for power in range(6, 17)]
INCLUDES = ["external_libraries/python_math_to_cpp/base_math",
            "external_libraries/python_math_to_cpp/python_math",
            "external_libraries/base_utility_cpp/base_utility"]


def tagged_source(name, revision):
    path = "test_vs/algorithm_improvements/" + name + "_regression.cpp"
    source = subprocess.check_output(["git", "show", revision + ":" + path],
                                     cwd=ROOT, text=True)
    # Tag the existing per-type calls for diagnostics; do not change assertions.
    return re.sub(r"^(  )((?:run|check_\w+)<(float|double)>\(\);)$",
                  lambda match: match[1] + 'std::puts("TYPE:' + match[3] + '");\n'
                  + match[0], source, flags=re.M)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=SOURCES / "results/epsilon_parameter_sweep")
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--values", nargs="+", default=VALUES)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "regression_logs"
    logs.mkdir(exist_ok=True)
    metadata = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "compiler": subprocess.check_output([args.compiler, "--version"],
                                            text=True).splitlines()[0],
        "flags": "-std=c++11 -O2 -Wall -Wextra -Werror",
        "values": args.values,
        "assertions": "Unchanged; only main() type labels added to test copies.",
    }
    rows = []
    with tempfile.TemporaryDirectory(prefix="epsilon_regression_") as directory:
        temporary = Path(directory)
        commit, dependencies = snapshot(temporary, args.revision)
        headers = temporary / "base_matrix"
        metadata["commit"] = commit
        metadata["submodule_revisions"] = dependencies
        metadata["library_header_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(headers.glob("*.hpp"))}
        originals = {name: (headers / name).read_text() for name in
                     ["base_matrix_gmres.hpp", "base_matrix_eigen_solver.hpp"]}
        for name in ["gmres", "eigen"]:
            (temporary / (name + ".cpp")).write_text(tagged_source(name, commit))
        for value in args.values:
            for name, original in originals.items():
                replacement = original
                if value != "machine":
                    replacement, count = re.subn(
                        r"std::numeric_limits<(T|Real)>::epsilon\(\)",
                        lambda match: "static_cast<" + match[1] + ">(" + value + ")",
                        original)
                    expected = 1 if name == "base_matrix_gmres.hpp" else 5
                    if count != expected:
                        raise RuntimeError("Unexpected epsilon call count: " + name)
                (headers / name).write_text(replacement)
            for math in ["default", "std"]:
                for suite in ["gmres", "eigen"]:
                    label = suite + "_" + math + "_" + value
                    binary = temporary / label
                    command = [args.compiler, "-std=c++11", "-O2", "-Wall",
                               "-Wextra", "-Werror", "-DALGORITHM_IMPROVEMENTS"]
                    if math == "std":
                        command += ["-DBASE_MATH_USE_STD_MATH_"]
                    command += ["-I" + str(headers)]
                    command += ["-I" + str(temporary / include) for include in INCLUDES]
                    command += [str(temporary / (suite + ".cpp")), "-o", str(binary)]
                    build = subprocess.run(command, text=True, capture_output=True)
                    if build.returncode:
                        (logs / (label + ".build.txt")).write_text(build.stdout + build.stderr)
                        raise RuntimeError("Build failed: " + label)
                    result = subprocess.run([str(binary)], text=True, capture_output=True)
                    (logs / (label + ".txt")).write_text(result.stdout + result.stderr)
                    dtype = "double"
                    failures = {"float": [], "double": []}
                    for line in result.stdout.splitlines():
                        if line.startswith("TYPE:"):
                            dtype = line.split(":", 1)[1]
                        if line.startswith("FAIL:"):
                            failures[dtype].append(line[6:])
                    if result.returncode and not any(failures.values()):
                        raise RuntimeError("Unexpected process failure: " + label)
                    for dtype in ["float", "double"]:
                        rows.append({"epsilon_base": value, "math": math,
                                     "suite": suite, "type": dtype,
                                     "failures": len(failures[dtype]),
                                     "messages": "; ".join(failures[dtype])})
                    print(label + ": float=" + str(len(failures["float"])) +
                          ", double=" + str(len(failures["double"])), flush=True)
                    with (output / "regression_summary.csv").open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
    (output / "regression_metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
