#!/usr/bin/env python3
"""Reproduce host regression tests and before/after algorithm measurements.

The baseline is an immutable source archive, including the recorded submodules.
No checkout, network access, or changes to the user's worktree are required.
"""
import argparse
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SOURCES = Path(__file__).resolve().parent
BASELINE_COMMIT = "b7faabf3bbac107d1be1462d139f4be0fa25a6b7"
INCLUDES = ["base_matrix", "python_numpy", "external_libraries/python_math_to_cpp/base_math",
            "external_libraries/python_math_to_cpp/python_math",
            "external_libraries/base_utility_cpp/base_utility", "test_vs/MCAP_tester/include"]


def execute(arguments, output=None, timing_cpu=None):
    setup = None if timing_cpu is None else lambda: os.sched_setaffinity(0, {timing_cpu})
    result = subprocess.run([str(item) for item in arguments], cwd=ROOT,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            preexec_fn=setup)
    if output:
        Path(output).write_text(result.stdout)
    if result.returncode:
        print(result.stdout)
        raise RuntimeError("Command failed: " + " ".join(str(item) for item in arguments))
    return result.stdout


def archive(repository, revision, destination):
    destination.mkdir(parents=True, exist_ok=True)
    data = subprocess.check_output(["git", "-C", str(repository), "archive", revision])
    with tarfile.open(fileobj=io.BytesIO(data)) as source:
        source.extractall(destination, filter="data")


def snapshot(destination):
    archive(ROOT, BASELINE_COMMIT, destination)
    for path in ["external_libraries/base_utility_cpp", "external_libraries/python_math_to_cpp",
                 "test_vs/MCAP_tester"]:
        entry = execute(["git", "ls-tree", BASELINE_COMMIT, path]).split()
        archive(ROOT / path, entry[2], destination / path)


def build(source, tree, binary, defines=(), sanitize=False, extra_flags=(), optimization="-O2"):
    command = ["g++", "-std=c++11", "-Wall", "-Wextra", "-Werror",
               "-O1" if sanitize else optimization, "-fstack-usage"]
    command += list(extra_flags)
    if sanitize:
        command += ["-g", "-fsanitize=address,undefined,float-divide-by-zero",
                    "-fno-omit-frame-pointer"]
    command += ["-D" + define for define in defines]
    command += ["-I" + str(tree / path) for path in INCLUDES]
    command += [str(source), "-o", str(binary)]
    execute(command, str(binary) + ".build.log")


def audit_algorithm_usage():
    # The algorithm restriction applies to this repository, not submodules.
    for directory in ("base_matrix", "python_numpy"):
        for source in sorted((ROOT / directory).rglob("*.hpp")):
            code = re.sub(r"/\*.*?\*/|//[^\n]*", "", source.read_text(), flags=re.S)
            if re.search(r'#\s*include\s*[<"]algorithm[>"]|std::(?:copy|fill)\b', code):
                raise RuntimeError("Forbidden algorithm usage in " + str(source))


def audit_square_root_usage():
    # Root algorithms and their accuracy settings belong to python_math_to_cpp.
    forbidden = (r"Base::Math::(?:r?sqrt_newton_method|sqrt_extraction|"
                 r"fast_(?:inverse_)?square_root|SQRT_REPEAT_NUMBER(?:_MOSTLY_ACCURATE)?|"
                 r"SQRT_EXTRACTION_REPEAT_NUMBER)\b|"
                 r"Base::Math::(?:sqrt|rsqrt)\s*<[^>]*,")
    for directory in ("base_matrix", "python_numpy"):
        for source in sorted((ROOT / directory).rglob("*.hpp")):
            code = re.sub(r"/\*.*?\*/|//[^\n]*", "", source.read_text(), flags=re.S)
            if re.search(forbidden, code):
                raise RuntimeError("Matrix-local square-root implementation or precision in " + str(source))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(tempfile.mkdtemp(prefix="matrix_validation_")))
    parser.add_argument("--baseline", type=Path, help="Existing immutable baseline tree")
    parser.add_argument("--legacy-configs", action="store_true", help="Run the eight existing CI configurations")
    parser.add_argument("--sanitize", action="store_true", help="Also run sanitizers on the regression tests")
    parser.add_argument("--repeats", type=int, default=3, help="Independent benchmark processes (default 3)")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline = args.baseline.resolve() if args.baseline else output / "baseline"
    if not args.baseline:
        snapshot(baseline)
    timing_cpu = min(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    metadata = {"timing_cpu": timing_cpu, "baseline_commit": BASELINE_COMMIT, "compiler": execute(["g++", "--version"]),
                "host": platform.platform(), "machine": platform.machine(),
                "flags": "-std=c++11 -Wall -Wextra -Werror -O2 -fstack-usage",
                "repeats": args.repeats, "cpu": Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else "",
                "working_diff": execute(["git", "diff", "--stat"])}
    (output / "environment.json").write_text(json.dumps(metadata, indent=2))
    forbidden_headers = output / "forbidden_headers"
    forbidden_headers.mkdir(exist_ok=True)
    # Remove an old guard when reusing output from the former all-dependency ban.
    (forbidden_headers / "algorithm").unlink(missing_ok=True)
    (forbidden_headers / "cmath").write_text('#error "Forbidden target header: cmath"\n')
    audit_algorithm_usage()
    audit_square_root_usage()
    constrained = output / "constraints_no_exceptions"
    build(SOURCES / "constraints_regression.cpp", ROOT, constrained,
          extra_flags=["-fno-exceptions", "-I" + str(forbidden_headers)])
    execute([constrained], str(constrained) + ".log")
    constrained_copy = output / "constraints_legacy_copy_macro"
    build(SOURCES / "constraints_regression.cpp", ROOT, constrained_copy,
          defines=["BASE_UTILITY_USE_STD_COPY_"],
          extra_flags=["-fno-exceptions", "-I" + str(forbidden_headers)])
    execute([constrained_copy], str(constrained_copy) + ".log")
    target_names = ["qr_decomposition", "eigen_solver", "lu_decomposition",
                    "cholesky_decomposition", "inverse", "gmres", "numerical"]
    for name in target_names:
        source = ROOT / "base_matrix" / ("base_matrix_" + name + ".hpp")
        code = re.sub(r"/\*.*?\*/|//[^\n]*", "", source.read_text(), flags=re.S)
        if re.search(r"\b(?:while|throw|try|catch|new|malloc|calloc|realloc)\b|std::(?:copy|fill)\b", code):
            raise RuntimeError("Forbidden construct in " + str(source))
    (output / "static_constraints.log").write_text(
        "Repository library headers: algorithm/copy/fill scan passed (external libraries excluded)\n"
        "Repository library headers: public sqrt/rsqrt delegation scan passed\n"
        "Seven target headers: forbidden construct scan passed\n")
    for source in sorted(SOURCES.glob("*_regression.cpp")):
        print("Regression:", source.name, flush=True)
        binary = output / source.stem
        build(source, ROOT, binary, defines=["ALGORITHM_IMPROVEMENTS"])
        execute([binary], str(binary) + ".log")
        if source.stem != "constraints_regression":
            # Host-only oracle configuration: precision is selected by the math
            # dependency, with the same matrix API and no matrix-local setting.
            standard_math = output / (source.stem + "_std_math")
            build(source, ROOT, standard_math,
                  defines=["ALGORITHM_IMPROVEMENTS", "BASE_MATH_USE_STD_MATH_"])
            execute([standard_math], str(standard_math) + ".log")
        if args.sanitize:
            binary = output / (source.stem + "_sanitized")
            build(source, ROOT, binary, defines=["ALGORITHM_IMPROVEMENTS"], sanitize=True)
            execute([binary], str(binary) + ".log")
    for source in sorted(SOURCES.glob("*_benchmark.cpp")):
        binaries = []
        for label, tree in [("before", baseline), ("after", ROOT)]:
            print("Benchmark:", source.name, label, flush=True)
            binary = output / (source.stem + "_" + label)
            build(source, tree, binary, defines=["ALGORITHM_IMPROVEMENTS"] if label == "after" else [])
            binaries.append(binary)
        for repeat in range(args.repeats):
            for binary in binaries:
                execute([binary], str(binary) + "." + str(repeat) + ".csv", timing_cpu=timing_cpu)
    execute([sys.executable, SOURCES / "gmres_oracle.py", "--baseline", baseline,
             "--output", output / "gmres_oracle"], output / "gmres_oracle.log")
    if args.legacy_configs:
        switches = ["BASE_MATRIX_USE_CONVENTIONAL_TEMPLATE_PATTERN_",
                    "BASE_MATRIX_USE_STD_VECTOR_", "BASE_MATRIX_USE_FOR_LOOP_OPERATION_"]
        for mask in range(8):
            print("Existing CI configuration:", mask, flush=True)
            binary = output / ("legacy_" + str(mask))
            build(ROOT / "test_vs/test_vs_main.cpp", ROOT, binary,
                  defines=[switch for bit, switch in enumerate(switches) if mask & (1 << bit)],
                  optimization="-O0")
            execute([binary], str(binary) + ".log")
    print("Evidence written to", output)


if __name__ == "__main__":
    main()
