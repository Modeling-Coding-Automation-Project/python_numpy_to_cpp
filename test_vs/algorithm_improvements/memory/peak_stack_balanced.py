#!/usr/bin/env python3
"""Rebuild baseline and an isolated working-header copy for stack comparison.

The existing probe, painter, compiler flags, measurements, and call-graph
analysis are reused. No production files or previous measurements are edited.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import peak_stack_run as peak

BASELINE = '6b1d627edf2de6e3e56deca59b83b37e942a97aa'
ROOT = peak.ROOT
HEADER_SUFFIXES = {'.h', '.hpp', '.ipp', '.tpp'}


def header_hashes(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob('*'))
            if path.is_file() and path.suffix in HEADER_SUFFIXES}


def prepare_and_compile(args, output):
    immutable_snapshot = peak.snapshot
    snapshots = {}

    def capture_snapshot(destination, revision):
        # Only this runner's isolated build directory is eligible for cleanup.
        if not destination.resolve().is_relative_to(args.build_dir.resolve()):
            raise ValueError('Snapshot must remain inside the isolated build directory')
        if destination.exists():
            shutil.rmtree(destination)
        commit, dependencies = immutable_snapshot(destination, BASELINE)
        if revision == 'working':
            before = header_hashes(ROOT / 'base_matrix')
            shutil.rmtree(destination / 'base_matrix')
            shutil.copytree(ROOT / 'base_matrix', destination / 'base_matrix')
            after = header_hashes(ROOT / 'base_matrix')
            copied = header_hashes(destination / 'base_matrix')
            if before != after or before != copied:
                raise ValueError('Working headers changed during snapshot creation')
            commit = 'working-tree@' + subprocess.check_output(
                ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        elif revision != BASELINE:
            raise ValueError('Unsupported revision')
        snapshots[revision] = dict(revision=commit, dependencies=dependencies,
            all_header_sha256=header_hashes(destination))
        return commit, dependencies

    args.revisions = ['baseline_6b1=' + BASELINE, 'selected_working=working']
    peak.snapshot = capture_snapshot
    try:
        peak.compile_probes(args, output)
    finally:
        peak.snapshot = immutable_snapshot

    before = snapshots[BASELINE]['all_header_sha256']
    after = snapshots['working']['all_header_sha256']
    changed = {path: dict(before=before.get(path), after=after.get(path))
               for path in sorted(set(before) | set(after)) if before.get(path) != after.get(path)}
    if any(not path.startswith('base_matrix/') for path in changed):
        raise ValueError('Only matrix headers may differ in this comparison')
    copied_matrix = {path.removeprefix('base_matrix/'): value for path, value in after.items()
                     if path.startswith('base_matrix/')}
    if copied_matrix != header_hashes(ROOT / 'base_matrix'):
        raise ValueError('Working headers changed before compilation completed')
    manifest = dict(baseline=BASELINE, snapshots=snapshots, changed_headers=changed,
        method='Both revisions rebuilt from identical probe/worker sources and flags. '
               'Selected revision uses an isolated base_matrix working-tree copy; '
               'both use dependencies pinned by the baseline.',
        working_headers_unchanged_during_build=True,
        probe_sha256=hashlib.sha256((peak.HERE / 'peak_stack_probe.cpp').read_bytes()).hexdigest(),
        worker_sha256=hashlib.sha256((peak.HERE / 'peak_stack_worker.cpp').read_bytes()).hexdigest())
    (output / 'header_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
        default=peak.HERE.parent / 'results/balanced_tolerance_selection/memory')
    parser.add_argument('--build-dir', type=Path, default=Path('/tmp/balanced_tolerance_memory'))
    parser.add_argument('--compiler', default='g++')
    parser.add_argument('--loop-modes', default='template,for')
    parser.add_argument('--stage', choices=['all', 'compile', 'measure', 'analyze'], default='all')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.stage in ('all', 'compile'):
        prepare_and_compile(args, output)
    if args.stage in ('all', 'measure'):
        peak.measure_probes(output)
    if args.stage in ('all', 'analyze'):
        peak.analyze(output)


if __name__ == '__main__':
    main()
