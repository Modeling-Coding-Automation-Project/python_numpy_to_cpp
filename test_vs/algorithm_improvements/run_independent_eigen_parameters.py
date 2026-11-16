#!/usr/bin/env python3
"""Measure independently configured final eigen/GMRES threshold coefficients.

All public parameters are assigned after construction. Only the hardcoded GMRES
breakdown coefficient is replaced in an immutable, isolated source snapshot.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import tempfile

import numpy as np

from run_epsilon_eigen_sweep import (ROOT, BASELINE_COMMIT, DEPENDENCIES,
                                     command, evaluate, snapshot_headers)

SOURCE = Path(__file__).with_name('epsilon_independent_eigen_probe.cpp')
BREAKDOWNS = ['8e-7', '8e-12', '8e-16']
PROFILES = list(itertools.product([8e-7, 8e-12], [3.2e-6, 3.2e-12],
                                 [1e-3, 1e-6, 1e-9], [5e-3, 1e-6, 1e-12]))
ADDITIONAL_PROFILES = [(8e-7, 3.2e-6, 3e-4, 5e-3), (8e-7, 3.2e-6, 3e-4, 1e-6),
                       (8e-9, 3.2e-6, 3e-4, 5e-3), (8e-9, 3.2e-6, 1e-6, 5e-3),
                       (8e-12, 3.2e-10, 1e-9, 5e-3), (8e-12, 3.2e-10, 1e-9, 1e-6),
                       (8e-12, 3.2e-10, 1e-11, 5e-3), (8e-9, 3.2e-6, 1e-4, 5e-3)]
PROFILES += ADDITIONAL_PROFILES


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quality_limit(math_mode, dtype):
    return 1e-10 if math_mode == 'std' and dtype == 'double' else 2e-6


def summarize(records):
    groups = {}
    for row in records:
        groups.setdefault((row['math'], row['type'], row['profile']), []).append(row)
    summaries = []
    for (math_mode, dtype, profile), rows in groups.items():
        general = [row for row in rows if row['algorithm'] != 'jacobi']
        limit = quality_limit(math_mode, dtype)
        quality = lambda row: (float(row['eigenvalue_error']) <= limit and
                               float(row['eigenvector_residual']) <= limit)
        usable = lambda row: (quality(row) and int(row['values_converged']) == 1 and
                              (row['algorithm'] == 'jacobi' or int(row['vectors_converged']) == 1))
        first = general[0]
        summaries.append(dict(math=math_mode, type=dtype, profile=profile,
            breakdown=first['breakdown'], requested_relative_tolerance=first['requested_relative_tolerance'],
            requested_vector_tolerance=first['requested_vector_tolerance'],
            requested_shift=first['requested_shift'], requested_inner_tolerance=first['requested_inner_tolerance'],
            cases=len(rows), values_converged=sum(int(row['values_converged']) for row in rows),
            general_cases=len(general), vectors_converged=sum(int(row['vectors_converged']) for row in general),
            inner_failed=sum(int(row['inner_failed']) for row in general), quality_limit=limit,
            quality_cases=sum(quality(row) for row in rows), usable_cases=sum(usable(row) for row in rows),
            max_eigenvalue_error=max(float(row['eigenvalue_error']) for row in rows),
            max_eigenvector_residual=max(float(row['eigenvector_residual']) for row in rows),
            qr_sweeps=sum(int(row['qr_sweeps']) for row in general)))
    return summaries


def baseline_references(revision):
    directory = SOURCE.parent / 'results/epsilon_parameter_sweep'
    metadata = json.loads((directory / 'eigen_metadata.json').read_text())
    if metadata['commit'] != revision:
        raise RuntimeError('Previous-sweep references must use the selected revision')
    path = directory / 'eigen_metrics.csv'
    rows = []
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if row['epsilon'] not in ['machine', '1e-7']:
                continue
            baseline = row['epsilon'] == 'machine'
            base = np.finfo(np.float32 if row['type'] == 'float' else np.float64).eps if baseline else 1e-7
            row.update(profile='baseline_machine' if baseline else 'common_1e-7',
                       breakdown=8 * float(base),
                       requested_relative_tolerance=8 * float(base),
                       requested_vector_tolerance=32 * float(base),
                       requested_shift=max(1e-6, math.sqrt(float(base))),
                       requested_inner_tolerance=16 * math.sqrt(float(base)))
            rows.append(row)
    return summarize(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'test_vs/algorithm_improvements/results/independent_numerical_parameters')
    parser.add_argument('--revision', default=BASELINE_COMMIT)
    parser.add_argument('--configuration', choices=['both', 'default', 'std'], default='both')
    parser.add_argument('--extras-only', action='store_true',
                        help='Append the eight follow-up profiles to saved initial-grid results')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configurations = ['default', 'std'] if args.configuration == 'both' else [args.configuration]
    revision = command(['git', '-C', ROOT, 'rev-parse', args.revision]).strip()
    references = baseline_references(revision)
    records = []
    if args.extras_only:
        previous = json.loads((output / 'eigen_metadata.json').read_text())
        if previous['commit'] != revision:
            raise RuntimeError('Follow-up profiles must use the initial-grid revision')
        with (output / 'eigen_metrics.csv').open() as stream:
            records = [row for row in csv.DictReader(stream)
                       if row['math'] not in configurations or
                       int(row['profile'].rsplit('_p', 1)[1]) < 36]
    flags = ['-std=c++11', '-O2', '-Wall', '-Wextra', '-Werror']
    with tempfile.TemporaryDirectory(prefix='independent_eigen_parameters_') as directory:
        work = Path(directory)
        headers = snapshot_headers(args.revision, work)
        gmres = headers / 'base_matrix_gmres.hpp'
        source, count = re.subn(
            r'static_cast<Real>\(GMRES_BREAKDOWN_EPSILON_MULTIPLIER\)\s*\*\s*std::numeric_limits<Real>::epsilon\(\)',
            'static_cast<Real>(TEST_BREAKDOWN)', gmres.read_text())
        if count != 1:
            raise RuntimeError('Expected exactly one hardcoded GMRES breakdown coefficient')
        gmres.write_text(source)
        includes = [headers] + [work / path for path in [
            'external_libraries/python_math_to_cpp/base_math',
            'external_libraries/python_math_to_cpp/python_math',
            'external_libraries/base_utility_cpp/base_utility']]
        for math_mode in configurations:
            for breakdown in BREAKDOWNS:
                label = 'eigen_' + math_mode + '_beta' + breakdown
                if args.extras_only:
                    label += '_extra'
                binary = work / label
                defines = ['-DTEST_BREAKDOWN=' + breakdown]
                if args.extras_only:
                    defines.append('-DTEST_EXTRA_ONLY')
                if math_mode == 'std':
                    defines.append('-DBASE_MATH_USE_STD_MATH_')
                command(['g++', *flags, *defines, *('-I' + str(path) for path in includes),
                         SOURCE, '-o', binary], work / (label + '.build.log'))
                raw = command([binary], output / (label + '.raw.csv'))
                grouped = {}
                for fields in csv.reader(raw.splitlines()):
                    profile = int(fields[0])
                    assert float(fields[1]) == float(breakdown)
                    assert float(fields[2]) == PROFILES[profile][3]
                    assert all(math.isfinite(float.fromhex(field)) for field in fields[15:])
                    grouped.setdefault(profile, []).append(','.join(fields[3:]))
                assert len(grouped) == (len(ADDITIONAL_PROFILES) if args.extras_only else len(PROFILES))
                for profile, lines in grouped.items():
                    identifier = 'beta' + breakdown + '_p' + str(profile)
                    rows = evaluate('\n'.join(lines), identifier, math_mode)
                    assert len(rows) == 44
                    relative, vector, shift, inner = PROFILES[profile]
                    for row in rows:
                        row.pop('epsilon')
                        row.update(profile=identifier, breakdown=float(breakdown),
                                   requested_relative_tolerance=relative, requested_vector_tolerance=vector,
                                   requested_shift=shift, requested_inner_tolerance=inner)
                        assert int(row['qr_sweeps']) <= 128
                        assert math.isfinite(row['eigenvalue_error'])
                        assert math.isfinite(row['eigenvector_residual'])
                        if row['algorithm'] != 'jacobi':
                            conversion = np.float32 if row['type'] == 'float' else np.float64
                            assert row['relative_tolerance'] == float(conversion(relative))
                            assert row['vector_tolerance'] == float(conversion(vector))
                            assert row['shift'] == float(conversion(shift))
                    records.extend(rows)
                print(label, 'profiles=', len(grouped), 'rows=', len(grouped) * 44, flush=True)
    keys = {(row['math'], row['type'], row['algorithm'], row['fixture'], row['profile'])
            for row in records}
    assert len(keys) == len(records)
    write_csv(output / 'eigen_metrics.csv', records)
    write_csv(output / 'eigen_summary.csv', summarize(records))
    write_csv(output / 'eigen_reference_summary.csv', references)
    metadata = dict(commit=revision,
        submodules={path: command(['git', '-C', ROOT, 'ls-tree', args.revision, path]).split()[2] for path in DEPENDENCIES},
        compiler=command(['g++', '--version']).splitlines()[0], flags=flags, numpy=np.__version__,
        profiles_per_math=len(BREAKDOWNS) * len(PROFILES), case_count=len(records),
        configurations=sorted({row['math'] for row in records}),
        grid=dict(breakdown=BREAKDOWNS, relative_tolerance=[8e-7, 8e-12],
                  vector_tolerance=[3.2e-6, 3.2e-12], shift=[1e-3, 1e-6, 1e-9],
                  inner_tolerance=[5e-3, 1e-6, 1e-12]),
        additional_profiles=ADDITIONAL_PROFILES,
        changes='Four public parameters assigned after construction; only breakdown coefficient patched in isolated headers.',
        budgets='N=3; general QR 128, inverse iteration 30, inner GMRES K=3/cycles=1/passes=1; Jacobi sweeps=10.',
        quality_gate='Both eigenvalue error and eigenvector residual <= 2e-6; std double <= 1e-10. Independent of requested convergence tolerance.',
        reference_note='Previous sweep baseline/common-1e-7 summaries re-evaluated with the same fixed quality gate; requested shift/inner values are nominal analytic values.',
        source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in [SOURCE, Path(__file__)]},
        oracle='NumPy binary64 on exact stored inputs, matching prior sweep; normalized by Frobenius norm of A.')
    (output / 'eigen_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Saved', len(records), 'cases', flush=True)


if __name__ == '__main__':
    main()
