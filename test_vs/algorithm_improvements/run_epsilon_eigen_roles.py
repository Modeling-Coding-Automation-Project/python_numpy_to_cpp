#!/usr/bin/env python3
"""One-factor eigen threshold comparisons; all other epsilon uses stay original."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from run_epsilon_eigen_sweep import ROOT, SOURCE, BASELINE_COMMIT, DEPENDENCIES, command, evaluate, snapshot_headers

ROLES = ['eigenvalue', 'eigenvector', 'shift', 'inner', 'breakdown']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'test_vs/algorithm_improvements/results/epsilon_parameter_sweep/eigen_roles')
    parser.add_argument('--work', type=Path, default=Path(tempfile.mkdtemp(prefix='eigen_epsilon_roles_')))
    parser.add_argument('--values', nargs='+', default=['1e-6', '1e-7'])
    parser.add_argument('--revision', default=BASELINE_COMMIT)
    parser.add_argument('--roles', nargs='+', choices=ROLES, default=ROLES)
    args = parser.parse_args()
    output, work = args.output.resolve(), args.work.resolve()
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    headers = snapshot_headers(args.revision, work)
    for name, roles in [('base_matrix_eigen_solver.hpp', ['eigenvalue', 'eigenvector', 'shift', 'inner', 'eigenvalue']),
                        ('base_matrix_gmres.hpp', ['breakdown'])]:
        path = headers / name
        code = path.read_text()
        found = list(re.finditer(r'std::numeric_limits<(T|Real)>::epsilon\(\)', code))
        assert len(found) == len(roles), (name, len(found))
        for match, role in reversed(list(zip(found, roles))):
            replacement = 'test_epsilon_value<' + match.group(1) + ', ' + str(ROLES.index(role)) + '>()'
            code = code[:match.start()] + replacement + code[match.end():]
        path.write_text(code)
    override = work / 'override.hpp'
    override.write_text('''#include <limits>\ntemplate <typename T, int Role> inline T test_epsilon_value() {\n  return Role == MATRIX_TEST_EPSILON_ROLE ? static_cast<T>(MATRIX_TEST_FIXED_EPSILON) : std::numeric_limits<T>::epsilon();\n}\n''')
    includes = [headers] + [work / name for name in ['external_libraries/python_math_to_cpp/base_math',
                'external_libraries/python_math_to_cpp/python_math', 'external_libraries/base_utility_cpp/base_utility']]
    records = []
    for role in args.roles:
        for mode in args.values:
            label = 'eigen_role_' + role + '_' + mode
            binary = work / label
            command(['g++', '-std=c++11', '-O2', '-Wall', '-Wextra', '-Werror',
                     '-DMATRIX_TEST_FIXED_EPSILON=' + mode,
                     '-DMATRIX_TEST_EPSILON_ROLE=' + str(ROLES.index(role)),
                     '-include', override, *('-I' + str(path) for path in includes), SOURCE, '-o', binary],
                    work / (label + '.build.log'))
            text = command([binary], output / (label + '.raw.csv'))
            for row in evaluate(text, mode, 'default'):
                row['role'] = role
                records.append(row)
            print(label, flush=True)
    with (output / 'eigen_roles_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    metadata = dict(commit=command(['git', '-C', ROOT, 'rev-parse', args.revision]).strip(),
                    flags='-std=c++11 -O2 -Wall -Wextra -Werror', math='default',
                    submodules={name: command(['git', '-C', ROOT, 'ls-tree', args.revision, name]).split()[2] for name in DEPENDENCIES},
                    substitutions={'eigenvalue': 'both general/symmetric eigenvalue relative tolerance',
                                   'eigenvector': 'general eigenvector residual tolerance',
                                   'shift': 'general inverse-iteration shift floor',
                                   'inner': 'inner GMRES fallback relative tolerance',
                                   'breakdown': 'GMRES Arnoldi breakdown'},
                    source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [SOURCE, Path(__file__)]})
    for name in ['base_matrix_eigen_solver.hpp', 'base_matrix_gmres.hpp']:
        contents = subprocess.check_output(['git', '-C', str(ROOT), 'show', args.revision + ':base_matrix/' + name])
        metadata['source_hashes']['base_matrix/' + name] = hashlib.sha256(contents).hexdigest()
    (output / 'eigen_roles_metadata.json').write_text(json.dumps(metadata, indent=2))
    print('Saved', len(records), 'measurements', flush=True)

if __name__ == '__main__':
    main()
