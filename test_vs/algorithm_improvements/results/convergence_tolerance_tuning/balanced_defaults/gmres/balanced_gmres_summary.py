#!/usr/bin/env python3
"""Summarize existing GMRES sweeps without compiling or executing the library."""
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / 'gmres'
ROOT = HERE.parents[5]
AUDITED_COMMIT = '6b1d627edf2de6e3e56deca59b83b37e942a97aa'


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def identity(row):
    return tuple(row[key] for key in ['type', 'n', 'fixture', 'k', 'cycles',
                                      'passes', 'precondition', 'phase'])


def main():
    quality = read_csv(SOURCE / 'quality_cases.csv')
    timing = read_csv(SOURCE / 'timing/summary.csv')
    reference = {identity(row): row for row in quality
                 if row['revision'] == 'current' and row['tau'] == '0'}
    summaries = []
    for tau in ['0', '1e-12', '1e-10', '1e-7', '1e-6']:
        rows = [row for row in quality if row['revision'] == 'current'
                and row['tau'] == tau and not row['fixture'].startswith('near_')]
        near = [row for row in quality if row['revision'] == 'current'
                and row['tau'] == tau and row['fixture'].startswith('near_')]
        times = [row for row in timing if row['tau'] == tau
                 and row['compared_with'] == 'current_tau0']
        lost = [row for row in rows if reference[identity(row)]['typed_quality_pass'] == '1'
                and row['typed_quality_pass'] == '0']
        summaries.append(dict(tau=tau, benchmark_states=len(rows),
            float_typed_pass=sum(int(row['typed_quality_pass']) for row in rows if row['type'] == 'float'),
            double_typed_pass=sum(int(row['typed_quality_pass']) for row in rows if row['type'] == 'double'),
            typed_total_pass=sum(int(row['typed_quality_pass']) for row in rows),
            typed_previously_successful_states_lost=len(lost),
            engineering_total_pass=sum(int(row['engineering_quality_pass']) for row in rows),
            forward_error_median=statistics.median(float(row['relative_forward_error']) for row in rows),
            forward_error_maximum=max(float(row['relative_forward_error']) for row in rows),
            maximum_lost_forward_error=max((float(row['relative_forward_error']) for row in lost), default=0),
            maximum_lost_error_over_typed_limit=max((float(row['relative_forward_error']) /
                float(row['typed_quality_limit']) for row in lost), default=0),
            near_accurate_states=sum(int(row['typed_quality_pass']) for row in near),
            near_total_states=len(near),
            near_delta_2e10_accurate_states=sum(int(row['typed_quality_pass']) for row in near
                if row['fixture'] == 'near_2e-10'),
            total_iterations=sum(int(row['iterations']) for row in rows),
            converged_states=sum(int(row['converged']) for row in rows),
            breakdown_states=sum(int(row['breakdown']) for row in rows),
            equal_frequency_median_time_sum_ratio=sum(float(row['median_ns']) for row in times) /
                sum(float(row['before_median_ns']) for row in times),
            configuration_paired_time_ratio_median=statistics.median(float(row['paired_ratio_median']) for row in times)))
    with (HERE / 'balanced_gmres_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    inverse = subprocess.check_output(['git', '-C', str(ROOT), 'show',
        AUDITED_COMMIT + ':base_matrix/base_matrix_inverse.hpp'], text=True)
    declarations = list(re.finditer(r'GmresSolver<[^;]+> solver;', inverse))
    adapters = [dict(line=inverse.count('\n', 0, match.start()) + 1,
        explicitly_assigns_decay_rate='solver.relative_tolerance = decay_rate;'
            in inverse[match.end():match.end() + 160]) for match in declarations]
    if len(adapters) != 17 or not all(row['explicitly_assigns_decay_rate'] for row in adapters):
        raise ValueError('The audited adapter contract changed')
    metadata = dict(audited_commit=AUDITED_COMMIT,
        inverse_sha256=hashlib.sha256(inverse.encode()).hexdigest(), adapters=adapters,
        input_sha256={str(path.relative_to(SOURCE)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in [SOURCE / 'quality_cases.csv', SOURCE / 'timing/summary.csv']},
        decision='Set the standalone GMRES default relative tolerance to 1e-12; keep beta, absolute tolerance, and budgets unchanged.',
        analysis='No new compilation or timing; summaries use immutable existing sweep records.')
    (HERE / 'balanced_gmres_audit.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
