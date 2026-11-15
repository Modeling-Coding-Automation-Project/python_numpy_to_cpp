#!/usr/bin/env python3
"""Independent high-precision GMRES solution checks for exact stored inputs.

The C++ probe exports each binary float with %a, including the final RHS after
all 1000 benchmark perturbations. Decimal.from_float then recovers that binary
value exactly. Decimal Gaussian elimination uses 80 significant digits and
partial pivoting. No third-party package or network access is required.
"""
import argparse
import csv
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("gmres_oracle_probe.cpp")
INCLUDES = ["base_matrix", "external_libraries/python_math_to_cpp/base_math",
            "external_libraries/python_math_to_cpp/python_math",
            "external_libraries/base_utility_cpp/base_utility"]
ZERO = Decimal(0)
ONE = Decimal(1)


def compile_probe(tree, binary, current):
    command = ["g++", "-std=c++11", "-O2", "-Wall", "-Wextra", "-Werror"]
    if current:
        command.append("-DALGORITHM_IMPROVEMENTS")
    command += ["-I" + str(tree / path) for path in INCLUDES]
    command += [str(SOURCE), "-o", str(binary)]
    subprocess.run(command, cwd=ROOT, check=True)


def number(text):
    # A promoted binary32 remains exact in binary64; %a is lossless for both.
    value = float.fromhex(text)
    if not math.isfinite(value):
        raise ValueError("Probe produced a nonfinite value: " + text)
    return Decimal.from_float(value)


def solve(matrix, rhs):
    n = len(rhs)
    work = [list(row) for row in matrix]
    result = list(rhs)
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(work[i][k]))
        if work[pivot][k] == ZERO:
            raise ValueError("Exact rounded fixture is singular")
        work[k], work[pivot] = work[pivot], work[k]
        result[k], result[pivot] = result[pivot], result[k]
        for i in range(k + 1, n):
            multiplier = work[i][k] / work[k][k]
            for j in range(k + 1, n):
                work[i][j] -= multiplier * work[k][j]
            result[i] -= multiplier * result[k]
    for i in range(n - 1, -1, -1):
        result[i] = (result[i] - sum((work[i][j] * result[j]
                                     for j in range(i + 1, n)), ZERO)) / work[i][i]
    return result


def norm(values):
    return sum((value * value for value in values), ZERO).sqrt()


def residual(matrix, rhs, solution):
    return [rhs[i] - sum((a * x for a, x in zip(row, solution)), ZERO)
            for i, row in enumerate(matrix)]


def ratio(numerator, denominator):
    return numerator / denominator if denominator != ZERO else numerator


def evaluate(text, revision, reference_cache):
    results = []
    for fields in csv.reader(text.splitlines()):
        mode, dtype, n, fixture, k, cycles, passes, precondition = fields[:8]
        n = int(n)
        converged, breakdown = fields[8:10]
        values = [number(value) for value in fields[10:]]
        reported_rho = values[0]
        matrix_values = values[1:1 + n * n]
        matrix = [matrix_values[i * n:(i + 1) * n] for i in range(n)]
        rhs = values[1 + n * n:1 + n * n + n]
        solution = values[1 + n * n + n:]
        if len(solution) != n:
            raise ValueError("Malformed probe row")
        cache_key = (dtype, n, fixture, tuple(matrix_values), tuple(rhs))
        if cache_key not in reference_cache:
            reference = solve(matrix, rhs)
            # Cross-check that a tiny backward error is not hiding insufficient
            # forward precision in the most ill-conditioned rounded fixtures.
            with localcontext() as verification_context:
                verification_context.prec = 120
                reference_120 = solve(matrix, rhs)
                precision_agreement = ratio(
                    norm([a - b for a, b in zip(reference, reference_120)]),
                    norm(reference_120))
            if precision_agreement > Decimal("1e-50"):
                raise ValueError("80/120-digit reference solutions disagree")
            reference_residual = ratio(norm(residual(matrix, rhs, reference)), norm(rhs))
            # Accuracy of the oracle is checked independently of solver output.
            if reference_residual > Decimal("1e-60"):
                raise ValueError("Insufficient oracle precision")
            reference_cache[cache_key] = reference, reference_residual, precision_agreement
        reference, reference_residual, precision_agreement = reference_cache[cache_key]
        absolute_residual = norm(residual(matrix, rhs, solution))
        solution_error = norm([x - target for x, target in zip(solution, reference)])
        normalized_error = ratio(solution_error, norm(reference))
        target_error = norm([x - ONE for x in solution]) / Decimal(n).sqrt()
        reference_distance = norm([x - ONE for x in reference]) / Decimal(n).sqrt()
        results.append({
            "revision": revision, "mode": mode, "type": dtype, "n": n,
            "fixture": fixture, "k": k, "cycles": cycles, "passes": passes,
            "precondition": precondition,
            "converged": converged, "breakdown": breakdown,
            "relative_solution_error": f"{normalized_error:.12E}",
            "relative_true_residual": f"{ratio(absolute_residual, norm(rhs)):.12E}",
            "absolute_true_residual": f"{absolute_residual:.12E}",
            "reported_rho": f"{reported_rho:.12E}",
            "manufactured_target_rms_error": f"{target_error:.12E}",
            "reference_distance_from_ones_rms": f"{reference_distance:.12E}",
            "reference_relative_residual": f"{reference_residual:.12E}",
            "reference_80_120_agreement": f"{precision_agreement:.12E}",
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path,
                        default=Path("/tmp/matrix_algorithm_baseline"))
    parser.add_argument("--output", type=Path,
                        default=Path(tempfile.mkdtemp(prefix="gmres_oracle_")))
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not (arguments.baseline / "base_matrix").is_dir():
        parser.error("Supply the immutable baseline produced by run_validation.py")
    metadata = {"precision": 80, "verification_precision": 120, "flags": "-std=c++11 -O2 -Wall -Wextra -Werror",
                "baseline": str(arguments.baseline.resolve()),
                "compiler": subprocess.check_output(["g++", "--version"], text=True),
                "rhs": "A*ones in T, then the benchmark's 1000 alternating perturbations"}
    all_results = []
    reference_cache = {}
    with localcontext() as context:
        context.prec = 80
        for revision, tree in [("before", arguments.baseline.resolve()), ("after", ROOT)]:
            binary = output / (revision + "_probe")
            compile_probe(tree, binary, revision == "after")
            text = subprocess.check_output([str(binary)], text=True)
            (output / (revision + "_exact_hex.csv")).write_text(text)
            all_results.extend(evaluate(text, revision, reference_cache))
    with (output / "gmres_oracle.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_results[0]))
        writer.writeheader()
        writer.writerows(all_results)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print("Wrote", len(all_results), "solver comparisons and",
          len(reference_cache), "independent systems to", output)


if __name__ == "__main__":
    main()
