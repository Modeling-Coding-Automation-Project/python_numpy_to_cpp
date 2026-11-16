#!/usr/bin/env python3
"""Summarize saved balanced Eigen selection, validation, and serial timing."""
import csv
import json
from pathlib import Path
import statistics


RESULTS = Path(__file__).with_name('results') / 'balanced_tolerance_selection/eigen'
KEYS = ['type', 'algorithm', 'n', 'fixture']


def read_rows(name):
    with (RESULTS / name).open() as stream:
        return list(csv.DictReader(stream))


def identity(row):
    return tuple(row[field] for field in KEYS)


def write_rows(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_quality(phase, candidate):
    rows = read_rows('balanced_eigen_' + phase + '_metrics.csv')
    before = {identity(row): row for row in rows if row['profile'] == 'current_a'}
    after = {identity(row): row for row in rows if row['profile'] == candidate}
    assert before.keys() == after.keys()
    changes = []
    counts = dict(both_usable=0, gained_usable=0, lost_usable=0, neither_usable=0)
    for key, previous in before.items():
        current = after[key]
        first, second = int(previous['usable']), int(current['usable'])
        label = ('both_usable' if first and second else 'gained_usable' if second
                 else 'lost_usable' if first else 'neither_usable')
        counts[label] += 1
        row = dict(phase=phase, **dict(zip(KEYS, key)), comparison=label)
        for prefix, record in [('before', previous), ('after', current)]:
            for field in ['value_error', 'vector_residual', 'values_converged',
                          'vectors_converged', 'inner_failed', 'status', 'qr_sweeps',
                          'quality_pass', 'reported_success', 'usable', 'object_bytes']:
                row[prefix + '_' + field] = record[field]
        assert previous['object_bytes'] == current['object_bytes']
        changes.append(row)
    return counts, changes


def summarize_timing():
    rows = read_rows('timing/balanced_eigen_timing_summary.csv')
    predicates = [
        ('all', lambda row: True),
        ('general', lambda row: row['algorithm'] != 'jacobi'),
        ('jacobi', lambda row: row['algorithm'] == 'jacobi'),
        ('both_usable', lambda row: row['before_usable'] == row['after_usable'] == '1'),
        ('both_usable_general', lambda row: row['before_usable'] == row['after_usable'] == '1'
         and row['algorithm'] != 'jacobi'),
        ('gained_usable', lambda row: row['before_usable'] == '0' and row['after_usable'] == '1'),
        ('lost_usable', lambda row: row['before_usable'] == '1' and row['after_usable'] == '0'),
        ('neither_usable', lambda row: row['before_usable'] == row['after_usable'] == '0'),
    ]
    result = []
    for profile in sorted({row['profile'] for row in rows}):
        for group, predicate in predicates:
            chosen = [row for row in rows if row['profile'] == profile and predicate(row)]
            result.append(dict(profile=profile, group=group, cases=len(chosen),
                median_case_ratio=statistics.median(float(row['paired_ratio_median']) for row in chosen),
                sum_case_medians_ratio=sum(float(row['after_median_ns']) for row in chosen)
                / sum(float(row['before_median_ns']) for row in chosen),
                material_speedup=sum(row['material_speedup'] == 'True' for row in chosen),
                material_slowdown=sum(row['material_slowdown'] == 'True' for row in chosen)))
    return result


def compare_inner_candidates():
    raw = {}
    for path in (RESULTS / 'timing/raw').glob('*.csv'):
        profile, repeat = path.stem.rsplit('.', 1)
        for row in read_rows(str(path.relative_to(RESULTS))):
            raw.setdefault(identity(row), {}).setdefault(profile, {})[int(repeat)] = float(row['ns'])
    result = []
    baseline = 'candidate_inner_1e-3'
    for profile in ['candidate_inner_1e-4', 'candidate_inner_2e-3']:
        for group in ['all', 'general', 'jacobi']:
            chosen = [profiles for key, profiles in raw.items()
                      if group == 'all' or (key[1] != 'jacobi') == (group == 'general')]
            ratios = [[profiles[profile][index] / profiles[baseline][index] for index in range(7)]
                      for profiles in chosen]
            result.append(dict(baseline=baseline, profile=profile, group=group, cases=len(chosen),
                median_case_ratio=statistics.median(statistics.median(values) for values in ratios),
                sum_case_medians_ratio=sum(statistics.median(profiles[profile].values()) for profiles in chosen)
                / sum(statistics.median(profiles[baseline].values()) for profiles in chosen),
                material_speedup=sum(statistics.median(values) < 0.90 and max(values) < 0.95 for values in ratios),
                material_slowdown=sum(statistics.median(values) > 1.10 and min(values) > 1.05 for values in ratios)))
    return result


def main():
    counts, changes = {}, []
    for phase, candidate in [('grid', 'r1e-06_v4e-06_i0.001'), ('validation', 'balanced_recommended')]:
        counts[phase], rows = compare_quality(phase, candidate)
        changes.extend(rows)
    write_rows(RESULTS / 'balanced_eigen_selected_changes.csv', changes)
    write_rows(RESULTS / 'timing/balanced_eigen_timing_groups.csv', summarize_timing())
    write_rows(RESULTS / 'timing/balanced_eigen_inner_comparison.csv', compare_inner_candidates())
    (RESULTS / 'balanced_eigen_comparison.json').write_text(json.dumps(counts, indent=2) + '\n')
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
