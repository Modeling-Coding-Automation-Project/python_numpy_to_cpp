#!/usr/bin/env python3
"""Validate one direct-constant prototype in an immutable temporary snapshot.

The candidate uses existing fields and changes their defaults, removing only
the epsilon coupling. Production headers and regression assertions are intact.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

from run_epsilon_gmres_sweep import DEFAULT_REVISION, ROOT, snapshot, archive


PARAMETERS = {"breakdown": "8e-12", "eigenvalue": "8e-9",
              "eigenvector": "3.2e-6", "shift": "3e-4", "inner": "5e-3"}
INCLUDES = ["base_matrix", "python_numpy",
            "external_libraries/python_math_to_cpp/base_math",
            "external_libraries/python_math_to_cpp/python_math",
            "external_libraries/base_utility_cpp/base_utility"]


def replace(code, pattern, replacement, expected):
    updated, count = re.subn(pattern, replacement, code)
    if count != expected:
        raise RuntimeError("Unexpected replacement count: " + pattern)
    return updated


def configure(headers):
    gmres = headers / "base_matrix_gmres.hpp"
    code = replace(gmres.read_text(),
        r"static_cast<Real>\(GMRES_BREAKDOWN_EPSILON_MULTIPLIER\)\s*\*\s*std::numeric_limits<Real>::epsilon\(\)",
        "static_cast<Real>(ResearchParameters::breakdown)", 1)
    gmres.write_text(code)
    eigen = headers / "base_matrix_eigen_solver.hpp"
    code = eigen.read_text()
    for name, constant, count in [
            ("eigenvalue", "EIGEN_RELATIVE_TOLERANCE_EPSILON_MULTIPLIER", 2),
            ("eigenvector", "EIGENVECTOR_TOLERANCE_EPSILON_MULTIPLIER", 1)]:
        code = replace(code, r"static_cast<T>\(" + constant +
            r"\)\s*\*\s*std::numeric_limits<T>::epsilon\(\)",
            "static_cast<T>(ResearchParameters::" + name + ")", count)
    code = replace(code, r"small_value\(static_cast<T>\(EIGEN_SMALL_VALUE\)\)",
                   "small_value(static_cast<T>(ResearchParameters::shift))", 1)
    code = replace(code,
        r"    T precision_shift = square_root\(std::numeric_limits<T>::epsilon\(\)\);\n"
        r"    if \(this->small_value < precision_shift\) \{\n"
        r"      this->small_value = precision_shift;\n    \}\n", "", 1)
    code = replace(code,
        r"static_cast<T>\(EIGEN_GMRES_TOLERANCE_SQRT_EPSILON_MULTIPLIER\)\s*\*\s*"
        r"EigenOperation::square_root\(\s*std::numeric_limits<T>::epsilon\(\)\)",
        "static_cast<T>(ResearchParameters::inner)", 1)
    assert "numeric_limits" not in code
    code = code.replace("#include <limits>\n", "")
    eigen.write_text(code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent /
                        "results/independent_numerical_parameters/prototype_validation")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="independent_parameters_check_") as directory:
        tree = Path(directory)
        commit, dependencies = snapshot(tree, args.revision)
        archive(ROOT, commit, tree, ["python_numpy", "test_vs/algorithm_improvements"])
        configure(tree / "base_matrix")
        settings = tree / "settings.hpp"
        definitions = ["namespace Base { namespace Matrix { namespace ResearchParameters {"]
        definitions += ["constexpr double " + key + " = " + value + ";"
                        for key, value in PARAMETERS.items()]
        definitions += ['static_assert(breakdown >= 0.0, "Nonnegative breakdown coefficient required");',
                        "} } }"]
        settings.write_text("\n".join(definitions) + "\n")
        poison = tree / "forbidden_headers"
        poison.mkdir()
        (poison / "cmath").write_text('#error "Target default math must not include cmath"\n')
        for mode in ["template", "for"]:
            for suite in ["gmres", "eigen", "constraints"]:
                label = suite + "_" + mode
                binary = tree / label
                command = ["g++", "-std=c++11", "-O2", "-Wall", "-Wextra", "-Werror",
                           "-DALGORITHM_IMPROVEMENTS", "-include", str(settings)]
                if mode == "for":
                    command += ["-DBASE_MATRIX_USE_FOR_LOOP_OPERATION_"]
                if suite == "constraints":
                    command += ["-fno-exceptions", "-I" + str(poison)]
                command += ["-I" + str(tree / name) for name in INCLUDES]
                command += [str(tree / "test_vs/algorithm_improvements" /
                                (suite + "_regression.cpp")), "-o", str(binary)]
                built = subprocess.run(command, text=True, capture_output=True)
                (output / (label + ".build.txt")).write_text(built.stdout + built.stderr)
                if built.returncode:
                    raise RuntimeError("Build failed: " + label)
                result = subprocess.run([str(binary)], text=True, capture_output=True)
                (output / (label + ".txt")).write_text(result.stdout + result.stderr)
                rows.append(dict(suite=suite, loop_mode=mode, exit_code=result.returncode,
                                 result=result.stdout.strip()))
                print(label, "exit=", result.returncode, result.stdout.strip(), flush=True)
        metadata = {"revision": commit, "dependencies": dependencies,
                    "parameters": PARAMETERS, "math": "default",
                    "flags": "-std=c++11 -O2 -Wall -Wextra -Werror",
                    "compiler": subprocess.check_output(["g++", "--version"], text=True),
                    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "note": "Prototype-only edits. Test assertions unchanged. Eigen limits include removed only in temporary prototype; GMRES retains limits for min()."}
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    failures = [row["suite"] + "_" + row["loop_mode"]
                for row in rows if row["exit_code"] != 0]
    if failures:
        raise RuntimeError("Regression failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
