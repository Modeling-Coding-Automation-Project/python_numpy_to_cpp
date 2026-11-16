#!/usr/bin/env python3
"""Sweep a fixed numerical epsilon without editing production headers.

The independent Decimal oracle solves the exact rounded matrix/RHS exported by
C++ as hexadecimal floats. Complex systems are expanded to real block systems.
"""
import argparse
import csv
from decimal import Decimal, localcontext
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

import gmres_oracle as oracle

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name('epsilon_gmres_probe.cpp')
DEFAULT_REVISION = '8e22c7b327fa4d783a0db33527cc25b5d941b8ba'
DEPENDENCIES = ['external_libraries/python_math_to_cpp',
                'external_libraries/base_utility_cpp']
VALUES = ['baseline'] + ['1e-' + str(power) for power in range(6, 17)]
INCLUDES = ['external_libraries/python_math_to_cpp/base_math',
            'external_libraries/python_math_to_cpp/python_math',
            'external_libraries/base_utility_cpp/base_utility']



def archive(repository, revision, destination, paths=()):
    """Read immutable Git content; never change the working tree or submodules."""
    destination.mkdir(parents=True, exist_ok=True)
    data = subprocess.check_output(['git', '-C', str(repository), 'archive', revision, *paths])
    with tarfile.open(fileobj=io.BytesIO(data)) as source:
        source.extractall(destination, filter='data')


def snapshot(destination, revision):
    """Archive library headers and the dependency commits pinned by revision."""
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse',
                                      revision + '^{commit}'], text=True).strip()
    archive(ROOT, commit, destination, ['base_matrix'])
    dependencies = {}
    for path in DEPENDENCIES:
        entry = subprocess.check_output(['git', '-C', str(ROOT), 'ls-tree', commit,
                                         '--', path], text=True).split()
        if len(entry) != 4 or entry[0] != '160000' or entry[1] != 'commit':
            raise ValueError('Expected a pinned submodule at ' + path)
        dependencies[path] = entry[2]
        archive(ROOT / path, entry[2], destination / path)
    return commit, dependencies


def evaluate(output, setting, math, cache):
    rows = []
    for fields in csv.reader(output.splitlines()):
        dtype, scalar, fixture, configuration = fields[:4]
        n, k, passes, cycles, iterations, converged, breakdown, storage = map(int, fields[4:12])
        tolerance, rho = map(oracle.number, fields[12:14])
        components = list(map(oracle.number, fields[14:]))
        values = list(zip(components[::2], components[1::2]))
        matrix_pairs = [values[i * n:(i + 1) * n] for i in range(n)]
        rhs_pairs = values[n * n:n * n + n]
        solution_pairs = values[n * n + n:]
        if len(solution_pairs) != n:
            raise ValueError('Invalid probe row')
        # Real systems use n dimensions; complex systems use [Re x, Im x].
        matrix = [[pair[0] for pair in row] for row in matrix_pairs]
        rhs = [pair[0] for pair in rhs_pairs]
        solution = [pair[0] for pair in solution_pairs]
        if scalar == 'complex':
            matrix = [row + [-pair[1] for pair in pairs]
                      for row, pairs in zip(matrix, matrix_pairs)]
            matrix += [[pair[1] for pair in pairs] + [pair[0] for pair in pairs]
                       for pairs in matrix_pairs]
            rhs += [pair[1] for pair in rhs_pairs]
            solution += [pair[1] for pair in solution_pairs]
        key = (dtype, scalar, fixture, tuple(tuple(row) for row in matrix), tuple(rhs))
        if key not in cache:
            reference = oracle.solve(matrix, rhs)
            with localcontext() as verification:
                verification.prec = 120
                exact = oracle.solve(matrix, rhs)
                agreement = oracle.ratio(oracle.norm([a - b for a, b in zip(reference, exact)]), oracle.norm(exact))
            if agreement > Decimal('1e-50'):
                raise ValueError('Independent precision disagreement')
            cache[key] = reference
        reference = cache[key]
        forward_error = oracle.ratio(oracle.norm([a - b for a, b in zip(solution, reference)]), oracle.norm(reference))
        relative_residual = oracle.ratio(oracle.norm(oracle.residual(matrix, rhs, solution)), oracle.norm(rhs))
        assert iterations <= k * cycles
        rows.append(dict(epsilon=setting, math=math, type=dtype, scalar=scalar,
                         fixture=fixture, configuration=configuration, n=n, k=k,
                         passes=passes, cycles=cycles, iterations=iterations,
                         converged=converged, breakdown=breakdown, object_bytes=storage,
                         relative_tolerance=str(tolerance),
                         relative_solution_error=f'{forward_error:.12E}',
                         relative_true_residual=f'{relative_residual:.12E}',
                         reported_residual=f'{rho:.12E}'))
    return rows


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)



def summarize(results):
    baseline = {(r['math'], r['type'], r['scalar'], r['fixture'], r['configuration']): r
                for r in results if r['epsilon'] == 'baseline'}
    summary = []
    groups = {'ordinary_full': lambda r: not r['fixture'].startswith('near_') and r['configuration'] in ('p1', 'p2'),
              'ordinary_zero': lambda r: not r['fixture'].startswith('near_') and r['configuration'] == 'p1_zero',
              'ordinary_restart': lambda r: r['configuration'] == 'restart_p2',
              'near': lambda r: r['fixture'].startswith('near_')}
    for epsilon in VALUES:
        for math in ['default', 'std']:
            for dtype in ['float', 'double']:
                for group, matches in groups.items():
                    rows = [r for r in results if r['epsilon'] == epsilon and r['math'] == math
                            and r['type'] == dtype and matches(r)]
                    improved, worsened = 0, 0
                    for row in rows:
                        original = baseline[row['math'], row['type'], row['scalar'], row['fixture'], row['configuration']]
                        actual = float(row['relative_solution_error'])
                        previous = float(original['relative_solution_error'])
                        improved += actual < previous
                        worsened += actual > previous
                    accuracy_limit = 1e-6 if dtype == 'float' else 1e-12
                    summary.append(dict(epsilon=epsilon, math=math, type=dtype, group=group,
                        cases=len(rows), converged=sum(int(r['converged']) for r in rows),
                        breakdown=sum(int(r['breakdown']) for r in rows),
                        total_iterations=sum(int(r['iterations']) for r in rows),
                        maximum_solution_error=max(float(r['relative_solution_error']) for r in rows),
                        maximum_true_residual=max(float(r['relative_true_residual']) for r in rows),
                        solution_error_improved=improved, solution_error_worsened=worsened,
                        solution_error_unchanged=len(rows)-improved-worsened,
                        accuracy_limit=accuracy_limit,
                        accurate_cases=sum(float(r['relative_solution_error']) <= accuracy_limit for r in rows)))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results/epsilon_parameter_sweep')
    parser.add_argument('--compiler', default='g++')
    parser.add_argument('--revision', default=DEFAULT_REVISION,
                        help='Library commit, including its pinned math/utility submodules')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results, raw = [], []
    cache = {}
    flags = ['-std=c++11', '-O2', '-Wall', '-Wextra', '-Werror']
    with tempfile.TemporaryDirectory(prefix='epsilon_gmres_') as directory:
        temporary = Path(directory)
        baseline = temporary / 'baseline'
        commit, dependencies = snapshot(baseline, args.revision)
        header_hashes = {name: hashlib.sha256((baseline / 'base_matrix' / name).read_bytes()).hexdigest()
                         for name in ['base_matrix_gmres.hpp', 'base_matrix_eigen_solver.hpp', 'base_matrix_numerical.hpp']}
        dependency_hashes = {str(path.relative_to(baseline)): hashlib.sha256(path.read_bytes()).hexdigest()
                             for dependency in DEPENDENCIES
                             for path in sorted((baseline / dependency).rglob('*.hpp'))}
        fixed = temporary / 'fixed'
        shutil.copytree(baseline / 'base_matrix', fixed / 'base_matrix')
        for name in ['base_matrix_eigen_solver.hpp', 'base_matrix_gmres.hpp']:
            path = fixed / 'base_matrix' / name
            original = path.read_text()
            text = original.replace('std::numeric_limits<T>::epsilon()',
                                    'static_cast<T>(MATRIX_TEST_FIXED_EPSILON)')
            text = text.replace('std::numeric_limits<Real>::epsilon()',
                                'static_cast<Real>(MATRIX_TEST_FIXED_EPSILON)')
            if text == original:
                raise ValueError('Requested revision has no expected epsilon call in ' + name)
            path.write_text(text)
        with localcontext() as context:
            context.prec = 80
            for epsilon in VALUES:
                tree = temporary / ('baseline' if epsilon == 'baseline' else 'fixed')
                for math in ['default', 'std']:
                    name = 'gmres_' + epsilon + '_' + math
                    binary = temporary / name
                    command = [args.compiler] + flags
                    if epsilon != 'baseline': command += ['-DMATRIX_TEST_FIXED_EPSILON=' + epsilon]
                    if math == 'std': command += ['-DBASE_MATH_USE_STD_MATH_']
                    command += ['-I' + str(tree / 'base_matrix')]
                    command += ['-I' + str(baseline / path) for path in INCLUDES]
                    command += [str(SOURCE), '-o', str(binary)]
                    compiled = subprocess.run(command, text=True, capture_output=True)
                    if compiled.returncode:
                        raise RuntimeError(compiled.stdout + compiled.stderr)
                    text = subprocess.check_output([str(binary)], text=True)
                    raw.append(dict(epsilon=epsilon, math=math, output=text))
                    results.extend(evaluate(text, epsilon, math, cache))
                    print(name, 'cases=', len(text.splitlines()), flush=True)
    write_csv(output / 'gmres_cases.csv', results)
    write_csv(output / 'gmres_summary.csv', summarize(results))
    (output / 'gmres_exact_inputs.json').write_text(json.dumps(raw, indent=2))
    metadata = dict(source_commit=commit, requested_revision=args.revision,
                    submodule_revisions=dependencies,
                    compiler=subprocess.check_output([args.compiler, '--version'], text=True).splitlines()[0],
                    flags=flags, epsilon_values=VALUES, precision=80, verification_precision=120,
                    case_count=len(results), oracle='Exact hexadecimal stored input; Decimal Gaussian elimination with partial pivoting. Complex inputs use an equivalent real block system.',
                    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    header_sha256=header_hashes, dependency_header_sha256=dependency_hashes,
                    note='No production headers changed. min() remains type dependent for reciprocal representability. Full-K cases cap at N, restart cases at K*cycles. Tight per-type relative tolerances are separate from the swept breakdown parameter; *_zero uses exact-zero stopping.')
    (output / 'gmres_metadata.json').write_text(json.dumps(metadata, indent=2))
    print('Wrote', len(results), 'cases to', output)

if __name__ == '__main__':
    main()
