#!/usr/bin/env python3
"""Time frozen A/candidate profiles in one binary, then join independent quality."""
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

from eigen_tolerance_sweep import REVISION, SOURCE as QUALITY_SOURCE, verify_snapshot, write_csv
from run_epsilon_eigen_sweep import command

SOURCE = Path(__file__).with_name('eigen_tolerance_benchmark.cpp')
RESULTS = SOURCE.parent / 'results/convergence_tolerance_tuning/eigen'
KEYS = ['type', 'algorithm', 'n', 'fixture']
STATE_FIELDS = ['values_converged', 'vectors_converged', 'inner_failed', 'status',
                'qr_sweeps', 'object_bytes']


def identity(row):
    return tuple(str(row[key]) for key in KEYS)


def quality_rows(directory):
    result = {}
    specifications = [('rtol', 'current_a', 'current_a'),
                      ('joint', 'r1e-6_v4.8e-06_i0.001', 'candidate_common'),
                      ('holdout', 'current_a', 'current_a'),
                      ('holdout', 'candidate_common', 'candidate_common')]
    for label, stored_name, profile in specifications:
        with (directory / ('eigen_tolerance_' + label + '_metrics.csv')).open() as stream:
            for row in csv.DictReader(stream):
                if row['profile'] == stored_name:
                    row['profile'] = profile
                    key = (profile, identity(row))
                    assert key not in result
                    result[key] = row
    assert len(result) == 280
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=Path('/tmp/eigen_tolerance_tuning_6b1d627'))
    parser.add_argument('--quality', type=Path, default=RESULTS)
    parser.add_argument('--output', type=Path, default=RESULTS / 'timing')
    parser.add_argument('--process-repeats', type=int, default=7)
    parser.add_argument('--cpu', type=int)
    args = parser.parse_args()
    assert args.process_repeats > 0
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_directory = output / 'raw'
    raw_directory.mkdir(exist_ok=True)
    work = args.work.resolve()
    snapshot_digest = verify_snapshot(work)
    # Timing fixtures were copied unchanged. Check that invariant mechanically.
    quality_source, timing_source = QUALITY_SOURCE.read_text(), SOURCE.read_text()
    begin = 'template <typename T> inline Matrix<T, 3, 3> small_fixture'
    quality_fixture = quality_source[quality_source.index(begin):quality_source.index(
        'template <typename T, typename S, std::size_t N, typename Values>\ninline void output')]
    timing_fixture = timing_source[timing_source.index(begin):timing_source.index(
        'std::size_t repetitions_override')]
    assert quality_fixture.strip() == timing_fixture.strip()
    quality = quality_rows(args.quality.resolve())
    flags = ['-std=c++11', '-O2', '-Wall', '-Wextra', '-Werror', '-fstack-usage']
    includes = [work / name for name in [
        'base_matrix', 'external_libraries/python_math_to_cpp/base_math',
        'external_libraries/python_math_to_cpp/python_math',
        'external_libraries/base_utility_cpp/base_utility']]
    binary = work / 'eigen_tolerance_benchmark'
    build = ['g++', *flags, *('-I' + str(path) for path in includes), SOURCE, '-o', binary]
    command(build, output / 'eigen_tolerance_build.log')
    shutil.copy2(binary.with_suffix('.su'), output / 'eigen_tolerance_benchmark.su')
    cpu = args.cpu if args.cpu is not None else min(os.sched_getaffinity(0))
    observations, order = {}, []
    for repeat in range(args.process_repeats):
        profiles = [('current_a', '0'), ('candidate_common', '1')]
        if repeat % 2:
            profiles.reverse()
        for profile, parameter in profiles:
            begin_time = time.monotonic()
            execution = subprocess.run([str(binary), parameter], check=True, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       preexec_fn=lambda: os.sched_setaffinity(0, {cpu}))
            elapsed = time.monotonic() - begin_time
            assert not execution.stderr
            (raw_directory / (profile + '.' + str(repeat) + '.csv')).write_text(execution.stdout)
            rows = list(csv.DictReader(execution.stdout.splitlines()))
            assert len(rows) == 140
            assert len({identity(row) for row in rows}) == 140
            for row in rows:
                assert row['profile'] == profile
                expected = quality[profile, identity(row)]
                assert all(row[field] == expected[field] for field in STATE_FIELDS)
                assert float(row['ns']) > 0
                observations.setdefault(identity(row), {}).setdefault(profile, []).append(row)
            order.append(dict(repeat=repeat, profile=profile, process_seconds=elapsed))
            print(repeat, profile, 'seconds=', round(elapsed, 3), flush=True)
    summaries = []
    for key, profiles in observations.items():
        before = profiles['current_a']
        after = profiles['candidate_common']
        ratios = [float(new['ns']) / float(old['ns']) for old, new in zip(before, after)]
        baseline_quality, candidate_quality = quality['current_a', key], quality['candidate_common', key]
        assert baseline_quality['object_bytes'] == candidate_quality['object_bytes']
        common = dict(zip(KEYS, key))
        common.update(corpus=before[0]['corpus'], before_median_ns=statistics.median(float(row['ns']) for row in before),
                      after_median_ns=statistics.median(float(row['ns']) for row in after),
                      paired_ratio_median=statistics.median(ratios), paired_ratio_min=min(ratios),
                      paired_ratio_max=max(ratios), faster_pairs=sum(ratio < 1 for ratio in ratios),
                      slower_pairs=sum(ratio > 1 for ratio in ratios), pairs=len(ratios),
                      material_speedup=statistics.median(ratios) < 0.90 and max(ratios) < 0.95,
                      material_slowdown=statistics.median(ratios) > 1.10 and min(ratios) > 1.05)
        for prefix, row in [('before', baseline_quality), ('after', candidate_quality)]:
            for field in ['quality_pass', 'reported_success', 'usable', 'value_error', 'vector_residual',
                          'vectors_converged', 'inner_failed', 'qr_sweeps', 'object_bytes']:
                common[prefix + '_' + field] = row[field]
        summaries.append(common)
    write_csv(output / 'eigen_tolerance_timing_summary.csv', summaries)
    metadata = dict(revision=REVISION, snapshot_digest=snapshot_digest, compiler=command(['g++', '--version']).splitlines()[0],
                    build_command=list(map(str, build)), flags=flags, cpu=cpu,
                    same_binary_for_profiles=True, process_repeats=args.process_repeats, process_order=order,
                    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    quality_source_sha256=hashlib.sha256(QUALITY_SOURCE.read_bytes()).hexdigest(),
                    binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    frame_file_sha256=hashlib.sha256((output / 'eigen_tolerance_benchmark.su').read_bytes()).hexdigest(),
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    note='Constructor-inclusive full solves; same compiled frames and sizeof for both runtime profiles; this does not measure dynamic peak stack.',
                    scope='Search 92 + frozen-candidate holdout 48 per profile; independent quality gates joined from separately measured raw solutions.')
    (output / 'eigen_tolerance_timing_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
