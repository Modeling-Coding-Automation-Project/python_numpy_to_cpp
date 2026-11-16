#!/usr/bin/env python3
"""Validate the adopted constants in the working tree without changing tests."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from run_validation import (ROOT, SOURCES, audit_algorithm_usage,
                            audit_square_root_usage, build, execute)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(tempfile.mkdtemp(
        prefix='balanced_parameter_validation_')))
    parser.add_argument('--legacy-configs', action='store_true')
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit_algorithm_usage()
    audit_square_root_usage()
    forbidden = output / 'forbidden_headers'
    forbidden.mkdir(exist_ok=True)
    (forbidden / 'cmath').write_text('#error "Default target math must not include cmath"\n')
    records = []

    def validate(label, source, defines, sanitize=False, extra_flags=(),
                 optimization='-O2'):
        binary = output / label
        build(source, ROOT, binary, defines=defines, sanitize=sanitize,
              extra_flags=extra_flags, optimization=optimization)
        result = execute([binary], output / (label + '.txt'))
        records.append(dict(label=label, result='passed', output=result.strip(),
                            defines=defines, sanitize=sanitize,
                            optimization='-O1' if sanitize else optimization))
        print(label, 'passed', flush=True)

    suites = ['gmres', 'gmres_operation', 'eigen']
    if (SOURCES / 'balanced_parameter_regression.cpp').exists():
        suites.append('balanced_parameter')
    for mode in ['template', 'for']:
        mode_defines = ['ALGORITHM_IMPROVEMENTS']
        if mode == 'for':
            mode_defines.append('BASE_MATRIX_USE_FOR_LOOP_OPERATION_')
        validate('constraints_' + mode, SOURCES / 'constraints_regression.cpp',
                 mode_defines, extra_flags=['-fno-exceptions', '-I' + str(forbidden)])
        for math in ['default', 'standard']:
            defines = mode_defines + (['BASE_MATH_USE_STD_MATH_'] if math == 'standard' else [])
            for suite in suites:
                validate(suite + '_' + mode + '_' + math,
                         SOURCES / (suite + '_regression.cpp'), defines)
    if args.sanitize:
        for suite in ['gmres', 'eigen']:
            validate(suite + '_sanitized', SOURCES / (suite + '_regression.cpp'),
                     ['ALGORITHM_IMPROVEMENTS'], sanitize=True)
    if args.legacy_configs:
        switches = ['BASE_MATRIX_USE_CONVENTIONAL_TEMPLATE_PATTERN_',
                    'BASE_MATRIX_USE_STD_VECTOR_', 'BASE_MATRIX_USE_FOR_LOOP_OPERATION_']
        for mask in range(8):
            defines = [switch for bit, switch in enumerate(switches) if mask & (1 << bit)]
            validate('legacy_' + str(mask), ROOT / 'test_vs/test_vs_main.cpp',
                     defines, optimization='-O0')
    sources = list((ROOT / 'base_matrix').glob('*.hpp'))
    sources += [SOURCES / (suite + '_regression.cpp') for suite in suites]
    sources += [SOURCES / 'constraints_regression.cpp', Path(__file__).resolve()]
    metadata = dict(head=execute(['git', 'rev-parse', 'HEAD']).strip(),
                    compiler=execute(['g++', '--version']).splitlines()[0],
                    flags='-std=c++11 -Wall -Wextra -Werror; O2 tests, O0 legacy, O1 sanitizers',
                    hashes={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in sources},
                    submodules=subprocess.check_output(['git', 'submodule', 'status'], cwd=ROOT, text=True),
                    results=records, count=len(records),
                    scope='Working-tree defaults; regression assertions are not relaxed.')
    (output / 'validation.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Validated', len(records), 'executions; records:', output, flush=True)


if __name__ == '__main__':
    main()
