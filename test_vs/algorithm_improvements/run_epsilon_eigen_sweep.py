#!/usr/bin/env python3
"""Host-only fixed-threshold eigen sweep against exact stored inputs using NumPy.

Source copies are isolated under --work. Production headers are never edited.
Default math is the target configuration; std math is a diagnostic host oracle.
"""
import argparse
import csv
import hashlib
import itertools
import io
import json
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import tarfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name('epsilon_eigen_probe.cpp')
BASELINE_COMMIT = '8e22c7b327fa4d783a0db33527cc25b5d941b8ba'
DEPENDENCIES = ['external_libraries/python_math_to_cpp', 'external_libraries/base_utility_cpp']
FIXTURES = ['ordinary', 'nonnormal', 'clustered_1e-6', 'clustered_1e-10',
            'repeated', 'complex_pair', 'cyclic', 'scaled_1e-12', 'scaled_1e12']


def command(args, log=None):
    result = subprocess.run([str(arg) for arg in args], capture_output=True, text=True)
    if log:
        log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError('Command failed: ' + ' '.join(map(str, args)) + '\n' + result.stdout + result.stderr)
    return result.stdout


def snapshot_headers(revision, work):
    data = subprocess.check_output(['git', '-C', str(ROOT), 'archive', revision, 'base_matrix'])
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        archive.extractall(work, filter='data')
    for dependency in DEPENDENCIES:
        revision_id = command(['git', '-C', ROOT, 'ls-tree', revision, dependency]).split()[2]
        destination = work / dependency
        destination.mkdir(parents=True, exist_ok=True)
        data = subprocess.check_output(['git', '-C', str(ROOT / dependency), 'archive', revision_id])
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            archive.extractall(destination, filter='data')
    return work / 'base_matrix'


def evaluate(text, mode, math):
    records = []
    for fields in csv.reader(text.splitlines()):
        dtype, algorithm = fields[:2]
        kind, converged, vector_converged, inner_failed, status, sweeps, object_bytes = map(int, fields[2:9])
        rel_tol, vec_tol, shift = map(float, fields[9:12])
        raw = np.array([float.fromhex(field) for field in fields[12:]])
        matrix = raw[:9].reshape(3, 3)
        parts = raw[9:15].reshape(3, 2)
        values = parts[:, 0] + 1j * parts[:, 1]
        parts = raw[15:].reshape(3, 3, 2)
        vectors = parts[:, :, 0] + 1j * parts[:, :, 1]
        reference = np.linalg.eigvalsh(matrix) if algorithm == 'jacobi' else np.linalg.eigvals(matrix)
        scale = np.linalg.norm(matrix, 'fro')
        error = min(max(abs(values[i] - reference[j]) for i, j in enumerate(order))
                    for order in itertools.permutations(range(3))) / scale
        residuals = []
        for j in range(3):
            length = np.linalg.norm(vectors[:, j])
            residuals.append(float(np.linalg.norm(matrix @ vectors[:, j] - values[j] * vectors[:, j]) /
                                   (scale * length)) if length else float('inf'))
        records.append(dict(epsilon=mode, math=math, type=dtype, algorithm=algorithm,
                            fixture=FIXTURES[kind], eigenvalue_error=float(error),
                            eigenvector_residual=max(residuals),
                            values_converged=converged, vectors_converged=vector_converged,
                            inner_failed=inner_failed, status=status, qr_sweeps=sweeps,
                            object_bytes=object_bytes, relative_tolerance=rel_tol,
                            vector_tolerance=vec_tol, shift=shift))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'test_vs/algorithm_improvements/results/epsilon_parameter_sweep')
    parser.add_argument('--work', type=Path, default=Path(tempfile.mkdtemp(prefix='eigen_epsilon_sweep_')))
    parser.add_argument('--values', nargs='+', default=['machine'] + ['1e-' + str(i) for i in range(6, 17)])
    parser.add_argument('--revision', default=BASELINE_COMMIT)
    parser.add_argument('--configuration', choices=['both', 'default', 'std'], default='both')
    args = parser.parse_args()
    output, work = args.output.resolve(), args.work.resolve()
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    headers = snapshot_headers(args.revision, work)
    configurations = ['default', 'std'] if args.configuration == 'both' else [args.configuration]
    # A test-only override, with the production expression retained for baseline.
    for name in ['base_matrix_eigen_solver.hpp', 'base_matrix_gmres.hpp']:
        path = headers / name
        code = path.read_text()
        code = re.sub(r'std::numeric_limits<(T|Real)>::epsilon\(\)',
                      lambda match: 'test_epsilon_value<' + match.group(1) + '>()', code)
        path.write_text(code)
    override = work / 'override.hpp'
    override.write_text('''#include <limits>\ntemplate <typename T> inline T test_epsilon_value() {\n#ifdef MATRIX_TEST_FIXED_EPSILON\n  return static_cast<T>(MATRIX_TEST_FIXED_EPSILON);\n#else\n  return std::numeric_limits<T>::epsilon();\n#endif\n}\n''')
    includes = [headers] + [work / name for name in ['external_libraries/python_math_to_cpp/base_math',
                'external_libraries/python_math_to_cpp/python_math', 'external_libraries/base_utility_cpp/base_utility']]
    flags = ['-std=c++11', '-O2', '-Wall', '-Wextra', '-Werror']
    records = []
    for math in configurations:
        for mode in args.values:
            label = 'eigen_' + math + '_' + mode
            binary = work / label
            defines = [] if mode == 'machine' else ['-DMATRIX_TEST_FIXED_EPSILON=' + mode]
            if math == 'std': defines.append('-DBASE_MATH_USE_STD_MATH_')
            command(['g++', *flags, *defines, '-include', override,
                     *('-I' + str(path) for path in includes), SOURCE, '-o', binary], work / (label + '.build.log'))
            text = command([binary], output / (label + '.raw.csv'))
            rows = evaluate(text, mode, math)
            records.extend(rows)
            print(label, 'rows=', len(rows), flush=True)
    with (output / 'eigen_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = []
    for math in configurations:
        for mode in args.values:
            for dtype in ['float', 'double']:
                selected = [r for r in records if r['math'] == math and r['epsilon'] == mode and r['type'] == dtype]
                general = [r for r in selected if r['algorithm'] != 'jacobi']
                summary.append(dict(epsilon=mode, math=math, type=dtype, cases=len(selected),
                    values_converged=sum(r['values_converged'] for r in selected),
                    vectors_converged=sum(r['vectors_converged'] for r in general),
                    general_cases=len(general), inner_failed=sum(r['inner_failed'] for r in general),
                    max_eigenvalue_error=max(r['eigenvalue_error'] for r in selected),
                    max_eigenvector_residual=max(r['eigenvector_residual'] for r in selected),
                    qr_sweeps=sum(r['qr_sweeps'] for r in general)))
    with (output / 'eigen_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in [SOURCE, Path(__file__)]}
    for name in ['base_matrix_eigen_solver.hpp', 'base_matrix_gmres.hpp']:
        contents = subprocess.check_output(['git', '-C', str(ROOT), 'show', args.revision + ':base_matrix/' + name])
        hashes['base_matrix/' + name] = hashlib.sha256(contents).hexdigest()
    metadata = dict(commit=command(['git', '-C', ROOT, 'rev-parse', args.revision]).strip(),
                    submodules={name: command(['git', '-C', ROOT, 'ls-tree', args.revision, name]).split()[2] for name in DEPENDENCIES},
                    compiler=command(['g++', '--version']), flags=flags,
                    numpy=np.__version__, platform=platform.platform(), source_hashes=hashes,
                    oracle='NumPy eigenvalues of exact stored input exported via %a; residuals evaluated in binary64',
                    budget='N=3, general QR 128 slots, general inverse iteration default 30 slots, Jacobi default 10 sweeps',
                    note='Jacobi vector convergence/status and actual sweep count are unavailable and recorded as -1.')
    (output / 'eigen_metadata.json').write_text(json.dumps(metadata, indent=2))
    print('Saved', len(records), 'measurements under', output, flush=True)

if __name__ == '__main__':
    main()
