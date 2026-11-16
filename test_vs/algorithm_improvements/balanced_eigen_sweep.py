#!/usr/bin/env python3
"""Balanced fixed Eigen defaults: all former training/holdout cases now select candidates."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil

import numpy as np

from eigen_tolerance_sweep import REVISION, QUALITY_LIMIT, matched_maximum_error, verify_snapshot, write_csv
from run_epsilon_eigen_sweep import ROOT, DEPENDENCIES, command, snapshot_headers

SOURCE = Path(__file__).with_name('balanced_eigen_probe.cpp')
RESULTS = SOURCE.parent / 'results/balanced_tolerance_selection/eigen'
SEVERE_LIMIT = 1e-3
DEFAULT_PROFILES = []
for relative, vector, inner in itertools.product([8e-9, 1e-7, 3e-7, 1e-6], [3.2e-6, 4e-6, 4.8e-6], [5e-3, 1e-3]):
    name = f'r{relative:g}_v{vector:g}_i{inner:g}'
    if (relative, vector, inner) == (8e-9, 3.2e-6, 5e-3): name = 'current_a'
    if (relative, vector, inner) == (1e-6, 4.8e-6, 1e-3): name = 'previous_candidate'
    DEFAULT_PROFILES.append(dict(name=name, relative=relative, vector=vector, inner=inner))
DEFAULT_PROFILES += [dict(name='old_tolerances_fixed_beta_shift', relative=0, vector=0, inner=0, mode=1),
                     dict(name='old_all_parameters', relative=0, vector=0, inner=0, mode=2)]


def evaluate(raw, profile, references):
    records = []
    for fields in csv.reader(raw.splitlines()):
        dtype, algorithm, dimension, fixture = fields[:4]
        dimension = int(dimension)
        values_converged, vectors_converged, inner_failed, status, sweeps, size = map(int, fields[4:10])
        relative, vector, shift, inner, internal_residual = map(float, fields[10:15])
        raw_numbers = np.array([float.fromhex(value) for value in fields[15:]])
        square = dimension * dimension
        assert len(raw_numbers) == 3 * square + 2 * dimension
        matrix = raw_numbers[:square].reshape(dimension, dimension)
        parts = raw_numbers[square:square + 2 * dimension].reshape(dimension, 2)
        values = parts[:, 0] + 1j * parts[:, 1]
        parts = raw_numbers[square + 2 * dimension:].reshape(dimension, dimension, 2)
        vectors = parts[:, :, 0] + 1j * parts[:, :, 1]
        key = (dtype, dimension, fixture, algorithm == 'jacobi')
        if key not in references:
            reference = np.linalg.eigvalsh(matrix) if algorithm == 'jacobi' else np.linalg.eigvals(matrix)
            references[key] = (matrix.copy(), reference)
        reference_matrix, reference = references[key]
        assert np.array_equal(matrix, reference_matrix)
        finite = bool(np.isfinite(raw_numbers).all())
        scale = np.linalg.norm(matrix, 'fro')
        value_error = matched_maximum_error(values, reference) / scale if finite else float('inf')
        residuals = []
        for index in range(dimension):
            length = np.linalg.norm(vectors[:, index])
            residuals.append(float(np.linalg.norm(matrix @ vectors[:, index] -
                                                   values[index] * vectors[:, index]) /
                                   (scale * length)) if finite and length > 0 else float('inf'))
        residual = max(residuals)
        quality = value_error <= QUALITY_LIMIT and residual <= QUALITY_LIMIT
        reported = values_converged == 1 and (algorithm == 'jacobi' or vectors_converged == 1)
        if algorithm != 'jacobi':
            conversion = np.float32 if dtype == 'float' else np.float64
            if profile.get('mode', 0) == 0:
                assert relative == float(conversion(profile['relative']))
                assert vector == float(conversion(profile['vector']))
                assert inner == float(conversion(profile['inner']))
            else:
                epsilon = float(np.finfo(conversion).eps)
                assert relative == float(conversion(8 * epsilon))
                assert vector == float(conversion(32 * epsilon))
                assert abs(inner / (16 * math.sqrt(epsilon)) - 1) < 1e-5
            if profile.get('mode', 0) == 2:
                assert abs(shift / max(1e-6, math.sqrt(float(np.finfo(conversion).eps))) - 1) < 1e-5
            else:
                assert shift == float(conversion(3e-4))
            assert 0 <= sweeps <= (128 if dimension == 3 else 64)
        records.append(dict(profile=profile['name'], type=dtype, algorithm=algorithm,
            n=dimension, fixture=fixture, requested_relative=profile['relative'],
            requested_vector=profile['vector'], requested_inner=profile['inner'],
            reference_mode=profile.get('mode', 0),
            relative_tolerance=relative, vector_tolerance=vector, shift=shift, inner_tolerance=inner,
            value_error=value_error, vector_residual=residual, internal_residual=internal_residual,
            values_converged=values_converged, vectors_converged=vectors_converged,
            inner_failed=inner_failed, status=status, qr_sweeps=sweeps, object_bytes=size,
            finite_output=int(finite), quality_pass=int(quality), reported_success=int(reported),
            usable=int(quality and reported)))
    return records


def summarize(records):
    groups = {}
    for row in records:
        for dtype in [row['type'], 'both']:
            groups.setdefault((row['profile'], dtype), []).append(row)
    summaries = []
    for (profile, dtype), rows in groups.items():
        general = [row for row in rows if row['algorithm'] != 'jacobi']
        unique = {}
        for row in rows:
            unique.setdefault((row['type'], row['n'], row['fixture']), []).append(row)
        severe = lambda row: max(float(row['value_error']), float(row['vector_residual'])) > SEVERE_LIMIT
        summaries.append(dict(profile=profile, type=dtype, cases=len(rows),
            relative=rows[0]['requested_relative'], vector=rows[0]['requested_vector'], inner=rows[0]['requested_inner'],
            reference_mode=rows[0]['reference_mode'], unique_inputs=len(unique),
            unique_quality=sum(all(int(row['quality_pass']) for row in values) for values in unique.values()),
            unique_usable=sum(all(int(row['usable']) for row in values) for values in unique.values()),
            severe_unique=sum(any(severe(row) for row in values) for values in unique.values()),
            quality_cases=sum(int(row['quality_pass']) for row in rows),
            usable_cases=sum(int(row['usable']) for row in rows),
            false_success=sum(int(row['reported_success']) and not int(row['quality_pass']) for row in rows),
            severe_cases=sum(severe(row) for row in rows),
            values_converged=sum(int(row['values_converged']) for row in rows),
            vectors_converged=sum(int(row['vectors_converged']) for row in general),
            inner_failed=sum(int(row['inner_failed']) for row in general),
            qr_sweeps=sum(int(row['qr_sweeps']) for row in general),
            max_value_error=max(float(row['value_error']) for row in rows),
            max_vector_residual=max(float(row['vector_residual']) for row in rows)))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=Path('/tmp/balanced_eigen_6b1d627'))
    parser.add_argument('--output', type=Path, default=RESULTS)
    parser.add_argument('--profiles', type=Path)
    parser.add_argument('--label', default='grid')
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--reuse-build', action='store_true')
    args = parser.parse_args()
    work, output = args.work.resolve(), args.output.resolve()
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if not (work / 'base_matrix').exists():
        snapshot_headers(REVISION, work)
    snapshot_digest = verify_snapshot(work)
    profiles = json.loads(args.profiles.read_text()) if args.profiles else DEFAULT_PROFILES
    assert len({profile['name'] for profile in profiles}) == len(profiles)
    flags = ['-std=c++11', '-O2', '-Wall', '-Wextra', '-Werror']
    if args.validation: flags.append('-DBALANCED_EIGEN_VALIDATION')
    stem = 'balanced_eigen_' + ('validation' if args.validation else 'selection')
    binary = work / stem
    old_binary = work / (stem + '_old_beta')
    includes = [work / name for name in ['base_matrix', 'external_libraries/python_math_to_cpp/base_math',
                'external_libraries/python_math_to_cpp/python_math', 'external_libraries/base_utility_cpp/base_utility']]
    build = ['g++', *flags, *('-I' + str(path) for path in includes), SOURCE]
    expected = dict(revision=REVISION, snapshot_digest=snapshot_digest,
                    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), flags=flags)
    if args.reuse_build:
        assert json.loads(binary.with_suffix('.json').read_text()) == expected
    else:
        command([*build, '-o', binary], output / ('balanced_eigen_' + args.label + '.build.log'))
        binary.with_suffix('.json').write_text(json.dumps(expected, indent=2) + '\n')
    patch = None
    if any(profile.get('mode', 0) == 2 for profile in profiles):
        header = work / 'base_matrix/base_matrix_gmres.hpp'
        original = header.read_bytes()
        pattern = b'static_cast<Real>(GMRES_BREAKDOWN_RELATIVE_TOLERANCE)'
        replacement = b'(static_cast<Real>(8) * std::numeric_limits<Real>::epsilon())'
        assert original.count(pattern) == 1
        patched = original.replace(pattern, replacement)
        patch = dict(original_sha256=hashlib.sha256(original).hexdigest(),
                     patched_sha256=hashlib.sha256(patched).hexdigest(),
                     original=pattern.decode(), replacement=replacement.decode())
        if not args.reuse_build:
            try:
                header.write_bytes(patched)
                command([*build, '-include', 'limits', '-o', old_binary],
                        output / ('balanced_eigen_' + args.label + '_old_beta.build.log'))
            finally:
                header.write_bytes(original)
        assert old_binary.exists()
        assert verify_snapshot(work) == snapshot_digest
    raw_directory = output / ('balanced_eigen_' + args.label + '_raw')
    raw_directory.mkdir(exist_ok=True)
    references, records = {}, []
    for index, profile in enumerate(profiles):
        mode = profile.get('mode', 0)
        chosen = old_binary if mode == 2 else binary
        raw = command([chosen, profile['relative'], profile['vector'], profile['inner'], mode],
                      raw_directory / (str(index) + '_' + profile['name'] + '.csv'))
        rows = evaluate(raw, profile, references)
        assert len(rows) == (48 if args.validation else 140)
        records.extend(rows)
        summary = next(row for row in summarize(rows) if row['type'] == 'both')
        print(profile['name'], 'quality=', summary['quality_cases'], 'usable=', summary['usable_cases'],
              'unique=', summary['unique_usable'], 'severe_unique=', summary['severe_unique'], flush=True)
    prefix = 'balanced_eigen_' + args.label
    write_csv(output / (prefix + '_metrics.csv'), records)
    write_csv(output / (prefix + '_summary.csv'), summarize(records))
    metadata = dict(**expected, binary=str(binary), compiler=command(['g++', '--version']).splitlines()[0],
                    numpy=np.__version__, validation=args.validation, profiles=profiles, records=len(records),
                    old_beta_compile_only_patch=patch,
                    binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    old_binary_sha256=hashlib.sha256(old_binary.read_bytes()).hexdigest() if patch else None,
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    dependencies={name: command(['git', '-C', ROOT, 'ls-tree', REVISION, name]).split()[2]
                                  for name in DEPENDENCIES},
                    selection_scope='All former 92 search + 48 holdout cases now select the balanced candidate; no claim they remain unseen.',
                    unique_input_rule='A type/dimension/fixture counts once; all applicable real/complex/Jacobi APIs must pass for unique_usable.',
                    quality_limit=QUALITY_LIMIT, severe_limit=SEVERE_LIMIT,
                    fixed='Except reference mode2: beta=8e-12, shift=3e-4; division_min=1e-20; N3 QR128, others64; inverse30; K=N,C=1,P=1; Jacobi10')
    (output / (prefix + '_metadata.json')).write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()


