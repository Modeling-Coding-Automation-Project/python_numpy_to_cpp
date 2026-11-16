#!/usr/bin/env python3
"""Rebuild historical/current headers and compare identical host benchmarks.

Use --phase build before other CPU-intensive work finishes, then --phase run
alone. All library and dependency revisions are immutable Git snapshots.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

from run_epsilon_gmres_sweep import ROOT, INCLUDES, snapshot


REVISIONS = {"last_timing": "879a31981d4cb99173939bd4a12df96bf8ccdf1a",
             "before_fixed": "ab31b0d", "current": "b599e84"}
SPECS = {
    "qr_benchmark": ("time_ns", ["type", "rows", "cols", "input", "method", "division_min"], 1),
    "lu_chol_benchmark": ("median_ns", ["type", "n", "method", "case"], 1),
    "gmres_benchmark": ("ns", ["mode", "type", "n", "fixture", "k", "cycles", "passes", "precondition"], 1),
    "gmres_rect_benchmark": ("ns", ["mode", "type", "m", "n"], 1),
    "eigen_benchmark": ("microseconds", ["algorithm", "type", "n", "budget"], 1000),
    "eigen_performance_probe": ("ns", ["algorithm", "type", "n", "fixture", "profile", "stage"], 1),
}


def execute(command, **options):
    result = subprocess.run([str(value) for value in command], text=True,
                            capture_output=True, **options)
    if result.returncode:
        raise RuntimeError("Command failed: " + str(command) + "\n" + result.stdout + result.stderr)
    return result.stdout


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(output, names, loop_mode):
    flags = ["-std=c++11", "-O2", "-Wall", "-Wextra", "-Werror",
             "-fstack-usage", "-DALGORITHM_IMPROVEMENTS"]
    if loop_mode == "for":
        flags.append("-DBASE_MATRIX_USE_FOR_LOOP_OPERATION_")
    manifest = dict(flags=flags, compiler=execute(["g++", "--version"]).splitlines()[0],
                    host=platform.platform(), cpuinfo=Path("/proc/cpuinfo").read_text(),
                    loop_mode=loop_mode, benchmarks=names, versions={}, sources={},
                    cpu=min(os.sched_getaffinity(0)))
    for name in names:
        source = Path(__file__).with_name(name + ".cpp")
        manifest["sources"][name] = hashlib.sha256(source.read_bytes()).hexdigest()
    for label, revision in REVISIONS.items():
        tree = output / "build" / label
        commit, dependencies = snapshot(tree, revision)
        manifest["versions"][label] = dict(commit=commit, dependencies=dependencies)
        includes = ["-I" + str(tree / path) for path in ["base_matrix"] + INCLUDES]
        for name in names:
            source = Path(__file__).with_name(name + ".cpp")
            binary = tree / name
            execute(["g++", *flags, *includes, source, "-o", binary])
            print("Built", label, name, flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def measure(output, repeats):
    manifest = json.loads((output / "manifest.json").read_text())
    records = []
    raw = output / "raw"
    raw.mkdir(exist_ok=True)
    labels = list(manifest["versions"])
    pin_cpu = lambda: os.sched_setaffinity(0, {manifest["cpu"]})
    for name in manifest["benchmarks"]:
        # Discard one warmup process per binary; rotate order in measured rounds.
        for label in labels:
            execute([output / "build" / label / name], preexec_fn=pin_cpu)
        for repeat in range(repeats):
            order = labels[repeat % len(labels):] + labels[:repeat % len(labels)]
            if repeat % 2:
                order = list(reversed(order))
            for label in order:
                started = time.monotonic()
                data = execute([output / "build" / label / name], preexec_fn=pin_cpu)
                (raw / (name + "_" + label + "." + str(repeat) + ".csv")).write_text(data)
                records.append(dict(benchmark=name, version=label, repeat=repeat,
                                    process_seconds=time.monotonic() - started))
            print("Measured", name, "round", repeat + 1, flush=True)
    manifest["process_repeats"] = repeats
    manifest["process_order"] = records
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def summarize(output):
    manifest = json.loads((output / "manifest.json").read_text())
    summaries, metrics = [], []
    for name in manifest["benchmarks"]:
        time_key, keys, scale = SPECS[name]
        groups, measurements = {}, {}
        for label in manifest["versions"]:
            groups[label], measurements[label] = {}, {}
            for repeat in range(manifest["process_repeats"]):
                path = output / "raw" / (name + "_" + label + "." + str(repeat) + ".csv")
                for row in csv.DictReader(io.StringIO(path.read_text())):
                    identity = ";".join(key + "=" + row[key] for key in keys)
                    timing = float(row[time_key]) * scale
                    assert math.isfinite(timing) and timing > 0
                    groups[label].setdefault(identity, []).append(timing)
                    details = {key: value for key, value in row.items() if key != time_key}
                    if identity in measurements[label]:
                        assert measurements[label][identity] == details
                    else:
                        measurements[label][identity] = details
            for identity, details in measurements[label].items():
                metrics.append(dict(benchmark=name, version=label, case=identity,
                                    metrics=json.dumps(details, sort_keys=True)))
        for baseline in ["last_timing", "before_fixed"]:
            assert set(groups[baseline]) == set(groups["current"])
            for identity, before in groups[baseline].items():
                after = groups["current"][identity]
                assert len(before) == len(after) == manifest["process_repeats"]
                ratios = [new / old for old, new in zip(before, after)]
                median_ratio = statistics.median(ratios)
                summaries.append(dict(benchmark=name, baseline=baseline, case=identity,
                    before_median_ns=statistics.median(before), after_median_ns=statistics.median(after),
                    paired_ratio_median=median_ratio, paired_ratio_min=min(ratios),
                    paired_ratio_max=max(ratios), slower_pairs=sum(ratio > 1 for ratio in ratios),
                    pairs=len(ratios),
                    material_slowdown=median_ratio > 1.10 and min(ratios) > 1.05))
    write_csv(output / "timing_summary.csv", summaries)
    write_csv(output / "quality_and_storage.csv", metrics)


def diagnostics(output):
    manifest = json.loads((output / "manifest.json").read_text())
    source = Path(__file__).with_name("gmres_performance_diagnostics.cpp")
    destination = output / "diagnostics"
    destination.mkdir(exist_ok=True)
    for label in manifest["versions"]:
        tree = output / "build" / label
        binary = tree / source.stem
        includes = ["-I" + str(tree / path) for path in ["base_matrix"] + INCLUDES]
        execute(["g++", *manifest["flags"], *includes, source, "-o", binary])
        result = execute([binary])
        rows = list(csv.DictReader(io.StringIO(result)))
        assert len(rows) == 108 and all(int(row["budget_violation_count"]) == 0 for row in rows)
        (destination / (label + ".csv")).write_text(result)
        print("Diagnosed", label, flush=True)
    manifest["sources"][source.stem] = hashlib.sha256(source.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["build", "run", "summarize", "diagnostics", "all"], default="all")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--loop-mode", choices=["template", "for"], default="template")
    parser.add_argument("--benchmarks", nargs="+", choices=list(SPECS), default=list(SPECS))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.phase in ["build", "all"]:
        build(output, args.benchmarks, args.loop_mode)
    if args.phase in ["run", "all"]:
        measure(output, args.repeats)
    if args.phase in ["run", "summarize", "all"]:
        summarize(output)
    if args.phase == "diagnostics":
        diagnostics(output)


if __name__ == "__main__":
    main()
