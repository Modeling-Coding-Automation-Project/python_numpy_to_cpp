#!/usr/bin/env python3
"""Compare balanced Eigen candidates in one binary with fixed-CPU serial timing."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

from balanced_eigen_sweep import RESULTS
from eigen_tolerance_sweep import REVISION, verify_snapshot, write_csv
from eigen_tolerance_timing import identity, STATE_FIELDS, KEYS
from run_epsilon_eigen_sweep import command

SOURCE = Path(__file__).with_name('balanced_eigen_benchmark.cpp')
PROFILES = ['current_a', 'candidate_inner_1e-4', 'candidate_inner_1e-3', 'candidate_inner_2e-3']


def quality_rows():
    result = {}
    mappings = [('grid', 'current_a', PROFILES[0]),
                ('nearby', 'r1e-06_v4e-06_i0.0001', PROFILES[1]),
                ('grid', 'r1e-06_v4e-06_i0.001', PROFILES[2]),
                ('nearby', 'r1e-06_v4e-06_i0.002', PROFILES[3])]
    for phase, stored, profile in mappings:
        with (RESULTS / ('balanced_eigen_' + phase + '_metrics.csv')).open() as stream:
            for row in csv.DictReader(stream):
                if row['profile'] == stored:
                    result[profile, identity(row)] = row
    assert len(result) == 560
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=Path('/tmp/balanced_eigen_6b1d627'))
    parser.add_argument('--output', type=Path, default=RESULTS / 'timing')
    parser.add_argument('--process-repeats', type=int, default=7)
    parser.add_argument('--cpu', type=int)
    args = parser.parse_args()
    work, output = args.work.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = output / 'raw'
    raw.mkdir(exist_ok=True)
    snapshot_digest = verify_snapshot(work)
    quality = quality_rows()
    flags = ['-std=c++11', '-O2', '-Wall', '-Wextra', '-Werror', '-fstack-usage']
    includes = [work / name for name in ['base_matrix', 'external_libraries/python_math_to_cpp/base_math',
                'external_libraries/python_math_to_cpp/python_math', 'external_libraries/base_utility_cpp/base_utility']]
    binary = work / 'balanced_eigen_benchmark'
    build = ['g++', *flags, *('-I' + str(path) for path in includes), SOURCE, '-o', binary]
    command(build, output / 'balanced_eigen_build.log')
    shutil.copy2(binary.with_suffix('.su'), output / 'balanced_eigen_benchmark.su')
    cpu = args.cpu if args.cpu is not None else min(os.sched_getaffinity(0))
    observations, order = {}, []
    for repeat in range(args.process_repeats):
        sequence = [(repeat + offset) % len(PROFILES) for offset in range(len(PROFILES))]
        if repeat % 2:
            sequence.reverse()
        for parameter in sequence:
            profile = PROFILES[parameter]
            begin = time.monotonic()
            execution = subprocess.run([str(binary), str(parameter)], check=True, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                preexec_fn=lambda: os.sched_setaffinity(0, {cpu}))
            duration = time.monotonic() - begin
            assert not execution.stderr
            (raw / (profile + '.' + str(repeat) + '.csv')).write_text(execution.stdout)
            rows = list(csv.DictReader(execution.stdout.splitlines()))
            assert len(rows) == 140 and len({identity(row) for row in rows}) == 140
            for row in rows:
                assert row['profile'] == profile
                expected = quality[profile, identity(row)]
                assert all(row[field] == expected[field] for field in STATE_FIELDS)
                observations.setdefault(identity(row), {}).setdefault(profile, {})[repeat] = row
            order.append(dict(repeat=repeat, profile=profile, process_seconds=duration))
            print(repeat, profile, round(duration, 3), flush=True)
    summaries = []
    for key, profiles in observations.items():
        baseline = profiles[PROFILES[0]]
        for profile in PROFILES[1:]:
            candidate = profiles[profile]
            ratios = [float(candidate[index]['ns']) / float(baseline[index]['ns'])
                      for index in range(args.process_repeats)]
            row = dict(zip(KEYS, key))
            row.update(profile=profile,
                before_median_ns=statistics.median(float(item['ns']) for item in baseline.values()),
                after_median_ns=statistics.median(float(item['ns']) for item in candidate.values()),
                paired_ratio_median=statistics.median(ratios), paired_ratio_min=min(ratios), paired_ratio_max=max(ratios),
                faster_pairs=sum(ratio < 1 for ratio in ratios), slower_pairs=sum(ratio > 1 for ratio in ratios),
                pairs=len(ratios), material_speedup=statistics.median(ratios) < 0.90 and max(ratios) < 0.95,
                material_slowdown=statistics.median(ratios) > 1.10 and min(ratios) > 1.05)
            for prefix, name in [('before', PROFILES[0]), ('after', profile)]:
                for field in ['quality_pass', 'usable', 'value_error', 'vector_residual', 'vectors_converged',
                              'inner_failed', 'qr_sweeps', 'object_bytes']:
                    row[prefix + '_' + field] = quality[name, key][field]
            assert row['before_object_bytes'] == row['after_object_bytes']
            summaries.append(row)
    write_csv(output / 'balanced_eigen_timing_summary.csv', summaries)
    metadata = dict(revision=REVISION, snapshot_digest=snapshot_digest, cpu=cpu, flags=flags,
                    build_command=list(map(str, build)), process_repeats=args.process_repeats, process_order=order,
                    profiles=PROFILES, compiler=command(['g++', '--version']).splitlines()[0],
                    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    same_binary=True, constructor_included=True,
                    note='All 140 cases now select the balanced profile; frame layout identical for runtime parameters. No MCU/WCET or dynamic peak stack measurement.')
    (output / 'balanced_eigen_timing_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
