#!/usr/bin/env python3
"""Check saved trace records and compare dynamic counts without rerunning kernels."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

from count_instructions import arithmetic, write_csv

HERE = Path(__file__).resolve().parent


def main():
    rows = list(csv.DictReader((HERE / 'dynamic_counts.csv').open()))
    indexed = {(row['case'], row['variant']): row for row in rows}
    cases = sorted({row['case'] for row in rows})
    comparisons = []
    for case in cases:
        for variant in ['template', 'for']:
            expected = indexed[case, variant]
            instructions = list(csv.DictReader((HERE / (variant + '_' + case + '.executed.csv')).open()))
            recalculated = Counter()
            for instruction in instructions:
                for name, amount in arithmetic(instruction).items():
                    recalculated[name] += int(instruction['single_step_visits']) * amount
            for name in ['fp_four_instruction', 'lane_add', 'lane_sub', 'lane_mul', 'lane_div',
                         'fma_instruction', 'single_step_events', 'repeat_string_step_events']:
                if recalculated[name] != int(expected[name]):
                    raise ValueError('Count mismatch: ' + case + '/' + variant + '/' + name)
            for line in (HERE / (variant + '_' + case + '.output.txt')).read_text().splitlines():
                if line.startswith(('input,', 'output,')):
                    if not all(math.isfinite(float.fromhex(value)) for value in line.split(',')[1:]):
                        raise ValueError('Nonfinite output: ' + case)
        template, loop = indexed[case, 'template'], indexed[case, 'for']
        row = dict(case=case)
        for metric in ['fp_four_instruction', 'lane_four_total', 'executed_distinct_fp_four_sites',
                       'lane_add', 'lane_sub', 'lane_mul', 'lane_div', 'fma_instruction',
                       'branch_instruction', 'integer_address_bit_arithmetic_instruction',
                       'integer_compare_instruction', 'single_step_events', 'repeat_string_step_events']:
            row['template_' + metric] = int(template[metric])
            row['for_' + metric] = int(loop[metric])
            row['delta_' + metric] = int(template[metric]) - int(loop[metric])
        comparisons.append(row)
    validation = json.loads((HERE / 'validation.json').read_text())
    if len(validation['comparisons']) != len(cases): raise ValueError('Case count mismatch')
    if not all(row['exact_inputs_equal'] and row['bitwise_outputs_equal'] and row['status_equal']
               for row in validation['comparisons']):
        raise ValueError('Inputs, outputs or solver status differ')
    write_csv(HERE / 'comparison.csv', comparisons)
    summary = dict(cases=len(cases), traced_calls=len(rows),
        dynamic_four_increased=[r['case'] for r in comparisons if r['delta_fp_four_instruction'] > 0],
        dynamic_four_equal=[r['case'] for r in comparisons if r['delta_fp_four_instruction'] == 0],
        dynamic_four_decreased=[r['case'] for r in comparisons if r['delta_fp_four_instruction'] < 0],
        lane_count_increased=[r['case'] for r in comparisons if r['delta_lane_four_total'] > 0],
        lane_count_decreased=[r['case'] for r in comparisons if r['delta_lane_four_total'] < 0],
        all_inputs_outputs_status_bitwise_equal=True,
        raw_trace_reaggregation_matches=True, all_inputs_outputs_finite=True,
        observed_fma_instructions=sum(int(r['fma_instruction']) for r in rows),
        tracer_sha256=hashlib.sha256((HERE / 'count_instructions.py').read_bytes()).hexdigest(),
        summarizer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
