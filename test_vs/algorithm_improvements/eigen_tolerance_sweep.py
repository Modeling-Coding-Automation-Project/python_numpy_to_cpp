#!/usr/bin/env python3
"""Isolated, quality-only Eigen convergence-tolerance study with fixed resources."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np

from run_epsilon_eigen_sweep import ROOT, DEPENDENCIES, command, snapshot_headers

SOURCE = Path(__file__).with_name('eigen_tolerance_probe.cpp')
REVISION = '6b1d627edf2de6e3e56deca59b83b37e942a97aa'
QUALITY_LIMIT = 2e-6
RELATIVE_TOLERANCES = [1e-5, 3e-6, 1e-6, 9.5367431640625e-7, 8e-7,
                       3e-7, 1e-7, 3e-8, 1e-9, 1e-10, 1e-11, 1e-12]
DEFAULT_PROFILES = [dict(name='current_a', relative=8e-9, vector=3.2e-6, inner=5e-3)] + [
    dict(name='rtol_' + format(value, '.12g'), relative=value, vector=3.2e-6, inner=5e-3)
    for value in RELATIVE_TOLERANCES]


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_snapshot(work, revision=REVISION):
    """Check every archived file against its immutable Git blob, including extras."""
    specifications = [(ROOT, revision, 'base_matrix', work)]
    for name in DEPENDENCIES:
        commit = command(['git', '-C', ROOT, 'ls-tree', revision, name]).split()[2]
        specifications.append((ROOT / name, commit, '', work / name))
    records = []
    for repository, commit, prefix, destination in specifications:
        arguments = ['git', '-C', repository, 'ls-tree', '-r', '-z', commit]
        if prefix:
            arguments.append(prefix)
        entries = command(arguments).split('\0')
        expected_paths = set()
        for entry in entries:
            if not entry:
                continue
            description, name = entry.split('\t', 1)
            mode, kind, expected_hash = description.split()
            if kind == 'commit':
                # git archive leaves nested gitlinks empty; include roots use
                # the separately verified top-level dependency snapshots.
                records.append((str((destination / name).relative_to(work)),
                                'gitlink:' + expected_hash))
                continue
            if kind != 'blob':
                raise RuntimeError('Unexpected non-file snapshot entry: ' + entry)
            path = destination / name
            contents = path.read_bytes()
            actual_hash = hashlib.sha1(b'blob ' + str(len(contents)).encode() + b'\0' + contents).hexdigest()
            if actual_hash != expected_hash:
                raise RuntimeError('Snapshot file differs from pinned Git tree: ' + str(path))
            expected_paths.add(path)
            records.append((str(path.relative_to(work)), expected_hash))
        scanned = destination / prefix if prefix else destination
        actual_paths = {path for path in scanned.rglob('*') if path.is_file()}
        if actual_paths != expected_paths:
            raise RuntimeError('Snapshot contains missing or extra files under ' + str(scanned))
    return hashlib.sha256(json.dumps(sorted(records)).encode()).hexdigest()


def matched_maximum_error(values, reference):
    """Minimum maximum eigenvalue mismatch, using finite bipartite matching."""
    distance = np.abs(values[:, None] - reference[None, :])
    bounds = sorted(set(float(value) for value in distance.ravel()))
    count = len(values)

    def perfect_match(threshold):
        owner = [-1] * count

        def visit(value, seen):
            for target in range(count):
                if not seen[target] and distance[value, target] <= threshold:
                    seen[target] = True
                    if owner[target] < 0 or visit(owner[target], seen):
                        owner[target] = value
                        return True
            return False

        return all(visit(value, [False] * count) for value in range(count))

    low, high = 0, len(bounds) - 1
    for unused in range(int(math.ceil(math.log2(len(bounds)))) + 1):
        if low < high:
            middle = (low + high) // 2
            if perfect_match(bounds[middle]):
                high = middle
            else:
                low = middle + 1
    return bounds[low]


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
            assert relative == float(conversion(profile['relative']))
            assert vector == float(conversion(profile['vector']))
            assert shift == float(conversion(3e-4))
            assert inner == float(conversion(profile['inner']))
            assert 0 <= sweeps <= (128 if dimension == 3 else 64)
        records.append(dict(profile=profile['name'], type=dtype, algorithm=algorithm,
            n=dimension, fixture=fixture, requested_relative=profile['relative'],
            requested_vector=profile['vector'], requested_inner=profile['inner'],
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
        for corpus in ['all', 'n3' if int(row['n']) == 3 else 'larger']:
            groups.setdefault((row['profile'], row['type'], corpus), []).append(row)
    result = []
    for (profile, dtype, corpus), rows in groups.items():
        general = [row for row in rows if row['algorithm'] != 'jacobi']
        result.append(dict(profile=profile, type=dtype, corpus=corpus, cases=len(rows),
            general_cases=len(general), relative=rows[0]['requested_relative'],
            vector=rows[0]['requested_vector'], inner=rows[0]['requested_inner'],
            quality_cases=sum(int(row['quality_pass']) for row in rows),
            reported_success=sum(int(row['reported_success']) for row in rows),
            usable_cases=sum(int(row['usable']) for row in rows),
            values_converged=sum(int(row['values_converged']) for row in rows),
            vectors_converged=sum(int(row['vectors_converged']) for row in general),
            inner_failed=sum(int(row['inner_failed']) for row in general),
            qr_sweeps=sum(int(row['qr_sweeps']) for row in general),
            max_value_error=max(float(row['value_error']) for row in rows),
            max_vector_residual=max(float(row['vector_residual']) for row in rows)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=SOURCE.parent / 'results/convergence_tolerance_tuning/eigen')
    parser.add_argument('--work', type=Path, help='Persistent isolated build directory, reusable between stages')
    parser.add_argument('--profiles', type=Path, help='JSON list of name/relative/vector/inner dictionaries')
    parser.add_argument('--label', default='rtol')
    parser.add_argument('--holdout', action='store_true', help='Use unseen N5/7/12 corpus only after candidate selection')
    parser.add_argument('--reuse-build', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = args.work.resolve() if args.work else Path(tempfile.mkdtemp(prefix='eigen_tolerance_'))
    work.mkdir(parents=True, exist_ok=True)
    profiles = json.loads(args.profiles.read_text()) if args.profiles else DEFAULT_PROFILES
    assert len({profile['name'] for profile in profiles}) == len(profiles)
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    flags = ['-std=c++11', '-O2', '-Wall', '-Wextra', '-Werror']
    if args.holdout:
        flags.append('-DEIGEN_TUNING_HOLDOUT')
    binary = work / ('eigen_tolerance_holdout' if args.holdout else 'eigen_tolerance_search')
    build_metadata = binary.with_suffix('.json')
    if not (work / 'base_matrix').exists():
        snapshot_headers(REVISION, work)
    snapshot_digest = verify_snapshot(work)
    expected = dict(revision=REVISION, source_sha256=source_hash, flags=flags,
                    snapshot_digest=snapshot_digest)
    if args.reuse_build:
        assert json.loads(build_metadata.read_text()) == expected
    else:
        includes = [work / name for name in [
            'base_matrix', 'external_libraries/python_math_to_cpp/base_math',
            'external_libraries/python_math_to_cpp/python_math',
            'external_libraries/base_utility_cpp/base_utility']]
        command(['g++', *flags, *('-I' + str(path) for path in includes), SOURCE, '-o', binary],
                output / ('eigen_tolerance_' + args.label + '.build.log'))
        build_metadata.write_text(json.dumps(expected, indent=2) + '\n')
    references, records = {}, []
    raw_directory = output / ('eigen_tolerance_' + args.label + '_raw')
    raw_directory.mkdir(exist_ok=True)
    for index, profile in enumerate(profiles):
        raw = command([binary, profile['relative'], profile['vector'], profile['inner']],
                      raw_directory / (str(index) + '_' + profile['name'] + '.csv'))
        rows = evaluate(raw, profile, references)
        assert len(rows) == (48 if args.holdout else 92)
        records.extend(rows)
        print(profile['name'], 'cases=', len(rows), 'quality=', sum(row['quality_pass'] for row in rows),
              'usable=', sum(row['usable'] for row in rows), flush=True)
    stem = 'eigen_tolerance_' + args.label
    write_csv(output / (stem + '_metrics.csv'), records)
    write_csv(output / (stem + '_summary.csv'), summarize(records))
    metadata = dict(**expected, binary=str(binary), compiler=command(['g++', '--version']).splitlines()[0],
        numpy=np.__version__, holdout=args.holdout, profiles=profiles, records=len(records),
        dependencies={name: command(['git', '-C', ROOT, 'ls-tree', REVISION, name]).split()[2] for name in DEPENDENCIES},
        fixed='beta=8e-12; shift=3e-4; division_min=1e-20; QR=128 for N3, 64 otherwise; inverse=30; K=N,C=1,P=1; Jacobi=10',
        quality='Eigenvalue error and independent normalized vector residual both <=2e-6, separate from convergence flags.',
        oracle='NumPy binary64 on exact exported inputs; minimum maximum eigenvalue matching; Frobenius normalization.',
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output / (stem + '_metadata.json')).write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
