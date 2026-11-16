#!/usr/bin/env python3
"""Tune only GMRES relative tolerance against fixed independent quality limits."""
import argparse
import csv
from decimal import Decimal, localcontext
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

import gmres_oracle as oracle
from run_epsilon_gmres_sweep import snapshot, INCLUDES, write_csv

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'gmres_tolerance_probe.cpp'
REVISIONS = {
    'last_timing': '879a31981d4cb99173939bd4a12df96bf8ccdf1a',
    'current': '6b1d627edf2de6e3e56deca59b83b37e942a97aa',
}
TAUS = ['0', '1e-12', '1e-10', '1e-9', '1e-8', '3e-8', '1e-7',
        '3e-7', '1e-6', '3e-6']
IDENTITY = ['type', 'n', 'fixture', 'k', 'cycles', 'passes', 'precondition', 'phase']


def decimal_text(value):
    return '0' if value == 0 else f'{value:.15E}'


def build(args, output):
    build_root = args.build_dir.resolve()
    build_root.mkdir(parents=True, exist_ok=True)
    flags = ['-std=c++11', '-O2', '-Wall', '-Wextra', '-Werror']
    metadata = dict(flags=flags, source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        compiler=subprocess.check_output([args.compiler, '--version'], text=True),
        precision=80, verification_precision=120, taus=TAUS, variants=[],
        fixed_quality_profiles={'typed': {'float': '1e-6', 'double': '1e-12'},
                                'engineering': {'float': '1e-6', 'double': '1e-6'}},
        quality_sequence='Two RHS states; all 1000 alternating updates verified by probe',
        timing_sequence='1000 alternating RHS updates and solves, zero initial x, volatile sink',
        controls='Only public relative_tolerance is changed. K/cycles/passes/preconditioner/floor remain fixed.')
    for label, revision in REVISIONS.items():
        tree = build_root / label
        commit, dependencies = snapshot(tree, revision)
        binary = tree / 'gmres_tolerance_probe'
        command = [args.compiler, *flags, '-I' + str(tree / 'base_matrix')]
        command += ['-I' + str(tree / name) for name in INCLUDES]
        command += [str(SOURCE), '-o', str(binary)]
        print('Building', label, commit, flush=True)
        subprocess.run(command, check=True)
        header = tree / 'base_matrix/base_matrix_gmres.hpp'
        metadata['variants'].append(dict(label=label, commit=commit,
            submodules=dependencies, binary=str(binary), command=command,
            gmres_sha256=hashlib.sha256(header.read_bytes()).hexdigest()))
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


def collect(output):
    metadata = json.loads((output / 'metadata.json').read_text())
    for variant in metadata['variants']:
        for tau in TAUS if variant['label'] == 'current' else ['0']:
            command = [variant['binary'], 'quality', tau]
            raw = subprocess.check_output(command, text=True)
            if len(raw.splitlines()) != 222:
                raise ValueError('Expected 108 configurations x two states + six near cases')
            (output / (variant['label'] + '_' + tau + '.exact.csv')).write_text(raw)
            print(variant['label'], tau, '222 cases', flush=True)


def evaluate(raw, revision, tau, cache):
    result = []
    for fields in csv.reader(raw.splitlines()):
        dtype, dimension, fixture = fields[:3]
        n = int(dimension)
        k, cycles, passes, precondition, phase, iterations, converged, breakdown, storage = map(int, fields[3:12])
        tolerance, reported = map(oracle.number, fields[12:14])
        values = list(map(oracle.number, fields[14:]))
        matrix_values = values[:n * n]
        matrix = [matrix_values[i * n:(i + 1) * n] for i in range(n)]
        rhs, solution = values[n * n:n * n + n], values[n * n + n:]
        if len(solution) != n or iterations > k * cycles:
            raise ValueError('Malformed output or exceeded deterministic iteration cap')
        key = (tuple(matrix_values), tuple(rhs))
        if key not in cache:
            reference = oracle.solve(matrix, rhs)
            with localcontext() as precise:
                precise.prec = 120
                verification = oracle.solve(matrix, rhs)
                agreement = oracle.ratio(oracle.norm([a - b for a, b in zip(reference, verification)]),
                                         oracle.norm(verification))
            if agreement > Decimal('1e-50'):
                raise ValueError('80/120-digit oracle disagreement')
            if oracle.ratio(oracle.norm(oracle.residual(matrix, rhs, reference)), oracle.norm(rhs)) > Decimal('1e-60'):
                raise ValueError('Oracle residual exceeds fixed validation limit')
            cache[key] = reference, agreement
        reference, agreement = cache[key]
        forward = oracle.ratio(oracle.norm([a - b for a, b in zip(solution, reference)]), oracle.norm(reference))
        relative_residual = oracle.ratio(oracle.norm(oracle.residual(matrix, rhs, solution)), oracle.norm(rhs))
        typed_limit = Decimal('1e-6' if dtype == 'float' else '1e-12')
        result.append(dict(revision=revision, tau=tau, type=dtype, n=n, fixture=fixture,
            k=k, cycles=cycles, passes=passes, precondition=precondition, phase=phase,
            iterations=iterations, converged=converged, breakdown=breakdown,
            object_bytes=storage, relative_tolerance=str(tolerance),
            relative_forward_error=decimal_text(forward), relative_true_residual=decimal_text(relative_residual),
            reported_absolute_residual=decimal_text(reported),
            typed_quality_limit=str(typed_limit), typed_quality_pass=int(forward <= typed_limit),
            engineering_quality_limit='1e-6', engineering_quality_pass=int(forward <= Decimal('1e-6')),
            independent_residual_meets_tau=int(relative_residual <= tolerance),
            oracle_80_120_agreement=decimal_text(agreement)))
    return result


def quality(output):
    cache, rows = {}, []
    with localcontext() as context:
        context.prec = 80
        for revision in REVISIONS:
            for tau in TAUS if revision == 'current' else ['0']:
                raw = (output / (revision + '_' + tau + '.exact.csv')).read_text()
                rows.extend(evaluate(raw, revision, tau, cache))
    reference = {(r['revision'], tuple(r[key] for key in IDENTITY)): r
                 for r in rows if r['tau'] == '0'}
    for row in rows:
        for prefix, revision in [('current_zero', 'current'), ('last_zero', 'last_timing')]:
            before = reference[revision, tuple(row[key] for key in IDENTITY)]
            row[prefix + '_iterations_saved'] = before['iterations'] - row['iterations']
            row[prefix + '_typed_quality_lost'] = int(before['typed_quality_pass'] and not row['typed_quality_pass'])
            row[prefix + '_engineering_quality_lost'] = int(before['engineering_quality_pass'] and not row['engineering_quality_pass'])
            row[prefix + '_forward_error'] = before['relative_forward_error']
    write_csv(output / 'quality_cases.csv', rows)
    summary = []
    for tau in TAUS:
        for dtype in ['float', 'double']:
            for group in ['all_benchmark', 'well', 'scaled', 'hilbert', 'near']:
                selected = [r for r in rows if r['revision'] == 'current' and r['tau'] == tau and r['type'] == dtype
                    and ((group == 'all_benchmark' and not r['fixture'].startswith('near_')) or
                         (group == 'near' and r['fixture'].startswith('near_')) or r['fixture'] == group)]
                summary.append(dict(tau=tau, type=dtype, group=group, states=len(selected),
                    iterations=sum(r['iterations'] for r in selected),
                    current_zero_iterations_saved=sum(r['current_zero_iterations_saved'] for r in selected),
                    last_zero_iterations_saved=sum(r['last_zero_iterations_saved'] for r in selected),
                    converged=sum(r['converged'] for r in selected), breakdown=sum(r['breakdown'] for r in selected),
                    typed_quality_pass=sum(r['typed_quality_pass'] for r in selected),
                    engineering_quality_pass=sum(r['engineering_quality_pass'] for r in selected),
                    current_zero_typed_quality_lost=sum(r['current_zero_typed_quality_lost'] for r in selected),
                    current_zero_engineering_quality_lost=sum(r['current_zero_engineering_quality_lost'] for r in selected),
                    last_zero_typed_quality_lost=sum(r['last_zero_typed_quality_lost'] for r in selected),
                    last_zero_engineering_quality_lost=sum(r['last_zero_engineering_quality_lost'] for r in selected),
                    maximum_forward_error=max(float(r['relative_forward_error']) for r in selected),
                    maximum_true_residual=max(float(r['relative_true_residual']) for r in selected)))
    write_csv(output / 'quality_summary.csv', summary)
    storage = {}
    for row in rows:
        storage.setdefault(tuple(row[key] for key in IDENTITY), set()).add(row['object_bytes'])
    if not all(len(sizes) == 1 for sizes in storage.values()):
        raise ValueError('Object size changed')
    (output / 'quality_validation.json').write_text(json.dumps(dict(
        cases=len(rows), exact_input_systems=len(cache), finite_outputs=True,
        reference_precision=80, verification_precision=120, fixed_storage_unchanged=True,
        all_iterations_within_k_times_cycles=True,
        note='Convergence status uses working precision; the independent Decimal residual is evaluated separately.'), indent=2) + '\n')


def verify_timing(output):
    expected = {}
    for path in output.glob('*.exact.csv'):
        label = path.name.removesuffix('.exact.csv')
        revision, tau = label.rsplit('_', 1)
        for fields in csv.reader(path.read_text().splitlines()):
            dtype, n, fixture = fields[:3]
            if fixture.startswith('near_') or fields[7] != '1': continue
            identity = tuple(fields[:7])
            expected[revision, tau, identity] = (fields[8], fields[9], fields[10], fields[-int(n)])
    with (output / 'timing/samples.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        identity = tuple(row[key] for key in IDENTITY[:-1])
        actual = (row['last_iterations'], row['last_converged'], row['last_breakdown'], row['last_x0'])
        if actual != expected[row['revision'], row['tau'], identity]:
            raise ValueError('1000-solve timing result differs from two-state quality probe')
    (output / 'timing/validation.json').write_text(json.dumps(dict(
        samples=len(rows), all_last_iterations_flags_x0_match_quality=True,
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2) + '\n')


def summarize_candidates(output):
    with (output / 'quality_cases.csv').open() as stream:
        quality_rows = list(csv.DictReader(stream))
    with (output / 'timing/summary.csv').open() as stream:
        timing_rows = list(csv.DictReader(stream))
    keys = IDENTITY[:-1]
    quality_by_key = {}
    for row in quality_rows:
        quality_by_key.setdefault((row['revision'], row['tau'], tuple(row[key] for key in keys)), []).append(row)
    result = []
    for tau in dict.fromkeys(row['tau'] for row in timing_rows):
        for dtype in ['float', 'double']:
            for fixture in ['well', 'scaled', 'hilbert']:
                selected = [row for row in timing_rows if row['tau'] == tau and row['type'] == dtype
                            and row['fixture'] == fixture and row['compared_with'] == 'current_tau0']
                qualified, successful, lost = [], [], 0
                for row in selected:
                    identity = tuple(row[key] for key in keys)
                    after = quality_by_key['current', tau, identity]
                    before = quality_by_key['current', '0', identity]
                    passes = all(state['typed_quality_pass'] == '1' for state in after)
                    lost += all(state['typed_quality_pass'] == '1' for state in before) and not passes
                    if passes:
                        qualified.append(row)
                        if all(state['converged'] == '1' and state['breakdown'] == '0' and
                               state['independent_residual_meets_tau'] == '1' for state in after):
                            successful.append(row)
                for label, chosen in [('accurate', qualified), ('accurate_converged', successful)]:
                    selected_keys = {tuple(row[key] for key in keys) for row in chosen}
                    errors = [float(state['relative_forward_error']) for identity in selected_keys
                              for state in quality_by_key['current', tau, identity]]
                    previous_ratios = [float(row['paired_ratio_median']) for row in timing_rows
                        if row['tau'] == tau and row['compared_with'] == 'last_timing_tau0'
                        and tuple(row[key] for key in keys) in selected_keys]
                    result.append(dict(tau=tau, type=dtype, fixture=fixture, selection=label,
                        total_configs=len(selected), selected_configs=len(chosen),
                        current_zero_qualified_configs_lost=lost,
                        maximum_forward_error=max(errors) if errors else '',
                        median_time_ratio_to_current_zero=statistics.median(float(row['paired_ratio_median']) for row in chosen) if chosen else '',
                        median_time_ratio_to_last_zero=statistics.median(previous_ratios) if chosen else ''))
    write_csv(output / 'candidate_summary.csv', result)


def timing(args, output):
    metadata = json.loads((output / 'metadata.json').read_text())
    schedule = [(v, tau) for v in metadata['variants'] for tau in
                (args.taus if v['label'] == 'current' else ['0'])]
    cpu = min(os.sched_getaffinity(0)) if args.cpu is None else args.cpu
    records, order = [], []
    directory = output / 'timing'
    directory.mkdir(exist_ok=True)
    for repeat in range(args.repeats):
        current = schedule[repeat % len(schedule):] + schedule[:repeat % len(schedule)]
        for variant, tau in current:
            command = ['taskset', '-c', str(cpu), variant['binary'], 'time', tau]
            raw = subprocess.check_output(command, text=True)
            filename = variant['label'] + '_' + tau + '.' + str(repeat) + '.csv'
            (directory / filename).write_text(raw)
            order.append(dict(repeat=repeat, revision=variant['label'], tau=tau, file=filename))
            for fields in csv.reader(raw.splitlines()):
                dtype, n, fixture, k, cycles, passes, precondition = fields[:7]
                records.append(dict(revision=variant['label'], tau=tau, repeat=repeat,
                    type=dtype, n=int(n), fixture=['well', 'hilbert', 'scaled'][int(fixture)],
                    k=int(k), cycles=int(cycles), passes=int(passes), precondition=int(precondition),
                    ns=float(fields[7]), last_iterations=int(fields[8]),
                    last_converged=int(fields[9]), last_breakdown=int(fields[10]), last_x0=fields[11]))
            print(filename, flush=True)
    write_csv(directory / 'samples.csv', records)
    keys = ['type', 'n', 'fixture', 'k', 'cycles', 'passes', 'precondition']
    grouped = {}
    for row in records:
        grouped.setdefault((row['revision'], row['tau'], tuple(row[key] for key in keys)), []).append(row)
    summary = []
    for (revision, tau, identity), samples in grouped.items():
        if revision != 'current': continue
        for before_revision in ['last_timing', 'current']:
            before = {r['repeat']: r for r in grouped[before_revision, '0', identity]}
            ratios = [r['ns'] / before[r['repeat']]['ns'] for r in samples]
            summary.append(dict(tau=tau, compared_with=before_revision + '_tau0',
                **dict(zip(keys, identity)),
                median_ns=statistics.median(r['ns'] for r in samples),
                before_median_ns=statistics.median(r['ns'] for r in before.values()),
                paired_ratio_median=statistics.median(ratios), ratio_min=min(ratios), ratio_max=max(ratios)))
    write_csv(directory / 'summary.csv', summary)
    (directory / 'metadata.json').write_text(json.dumps(dict(cpu=cpu, repeats=args.repeats,
        taus=args.taus, process_order=order), indent=2) + '\n')
    verify_timing(output)
    summarize_candidates(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'results/convergence_tolerance_tuning/gmres')
    parser.add_argument('--build-dir', type=Path, default=Path('/tmp/gmres_tolerance_tuning'))
    parser.add_argument('--compiler', default='g++')
    parser.add_argument('--phase', choices=['explore', 'build', 'collect', 'quality', 'timing', 'verify_timing', 'summarize'], default='explore')
    parser.add_argument('--taus', nargs='+', default=['0', '1e-12', '1e-7'])
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--cpu', type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.phase in ['explore', 'build']: build(args, output)
    if args.phase in ['explore', 'collect']: collect(output)
    if args.phase in ['explore', 'quality']: quality(output)
    if args.phase == 'timing': timing(args, output)
    if args.phase == 'verify_timing': verify_timing(output)
    if args.phase == 'summarize': summarize_candidates(output)


if __name__ == '__main__':
    main()
