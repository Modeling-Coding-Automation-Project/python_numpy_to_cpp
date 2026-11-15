#!/usr/bin/env python3
"""Host-only eigenvalue oracle; requires NumPy. No package is needed by C++."""
import argparse
import itertools
from pathlib import Path
import subprocess
import tempfile

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument("--compiler", default="g++")
parser.add_argument("--configuration", choices=("both", "default", "std"), default="both")
args = parser.parse_args()
rng = np.random.default_rng(20261110)
matrices = []
for n in (3, 4, 6):
    for _ in range(12):
        matrices.append(rng.normal(size=(n, n)))
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    matrices.append(q @ np.diag(1 + np.arange(n) * 1e-6) @ q.T)
    matrices.append(q @ np.diag(np.ones(n)) @ q.T)
    matrices.append(np.roll(np.eye(n), 1, axis=0))

parts = ['#include "base_matrix_eigen_solver.hpp"', '#include <cstdio>',
         'using namespace Base::Matrix;', 'int main() {']
for index, matrix in enumerate(matrices):
    n = len(matrix)
    rows = ','.join('{' + ','.join(format(x, '.17g') for x in row) + '}'
                    for row in matrix)
    parts.append(f'{{ Matrix<double,{n},{n}> a({{{rows}}});')
    parts.append(f'EigenSolverComplex<double,{n}> solver; solver.iteration_max=512;')
    parts.append('solver.solve_eigen_values(a);')
    parts.append(f'std::printf("{index} %d", int(solver.eigen_values_converged()));')
    parts.append('for(auto v:solver.get_eigen_values()) std::printf(" %.17g %.17g",v.real,v.imag);')
    parts.append('std::puts(""); }')
parts.append('}')


def evaluate(output, configuration, tolerance):
    failures = []
    maximum = 0.0
    converged_count = 0
    for line in output.splitlines():
        fields = line.split()
        index, converged = map(int, fields[:2])
        converged_count += converged
        actual = np.asarray(list(map(float, fields[2:]))).reshape(-1, 2)
        actual = actual[:, 0] + 1j * actual[:, 1]
        expected = np.linalg.eigvals(matrices[index])
        # N <= 6: exhaustive matching avoids an order-dependent greedy oracle.
        error = min(max(abs(actual[i] - expected[j]) for i, j in enumerate(order))
                    for order in itertools.permutations(range(len(expected))))
        scale = max(1.0, np.linalg.norm(matrices[index], 2))
        relative = error / scale
        maximum = max(maximum, relative)
        if not converged or relative > tolerance:
            failures.append((index, converged, float(relative)))
    print(f'NumPy {np.__version__}, math={configuration}: cases={len(matrices)}, '
          f'converged={converged_count}, maximum scaled eigenvalue error={maximum:.6g}, '
          f'limit={tolerance:.6g}')
    for failure in failures:
        print('FAIL', failure)
    return failures


failures = []
with tempfile.TemporaryDirectory(prefix='eigen_numpy_oracle_') as directory:
    source = Path(directory) / 'oracle.cpp'
    executable = Path(directory) / 'oracle'
    source.write_text('\n'.join(parts))
    includes = [args.root / 'base_matrix',
                args.root / 'external_libraries/python_math_to_cpp/base_math',
                args.root / 'external_libraries/python_math_to_cpp/python_math',
                args.root / 'external_libraries/base_utility_cpp/base_utility']
    configurations = ('default', 'std') if args.configuration == 'both' else (args.configuration,)
    for configuration in configurations:
        defines = ['-DBASE_MATH_USE_STD_MATH_'] if configuration == 'std' else []
        subprocess.run([args.compiler, '-std=c++11', '-O2', '-Wall', '-Wextra', '-Werror',
                        *defines, *('-I' + str(p) for p in includes),
                        str(source), '-o', str(executable)], check=True)
        output = subprocess.check_output([str(executable)], text=True)
        # The default math sqrt has about 9e-8 relative error; matrix
        # transformations accumulate it. Standard math keeps the old bound.
        tolerance = 2e-10 if configuration == 'std' else 2e-6
        failures.extend(evaluate(output, configuration, tolerance))
raise SystemExit(bool(failures))
