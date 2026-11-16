#!/usr/bin/env python3
"""Investigate directly specified GMRES beta and tau without production edits."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
from decimal import localcontext

from run_epsilon_gmres_sweep import DEFAULT_REVISION, INCLUDES, snapshot, evaluate, write_csv

SOURCE = Path(__file__).with_name('epsilon_independent_gmres_probe.cpp')
BETAS = ['8e-7', '8e-12', '8e-16']
TAUS = ['0', '1e-3', '1e-6', '1e-12']


def summary(rows):
    result = []
    for beta in BETAS:
        for math_config in ['default', 'std']:
            for dtype in ['float', 'double']:
                for tau in TAUS:
                    for group in ['ordinary', 'near']:
                        chosen = [r for r in rows if r['beta'] == beta and r['math'] == math_config
                                  and r['type'] == dtype and r['tau_label'] == tau
                                  and r['fixture'].startswith('near_') == (group == 'near')]
                        limit = 1e-6 if dtype == 'float' else 1e-12
                        result.append(dict(beta=beta, math=math_config, type=dtype, tau=tau,
                            group=group, cases=len(chosen),
                            converged=sum(r['converged'] for r in chosen),
                            breakdown=sum(r['breakdown'] for r in chosen),
                            accurate_cases=sum(float(r['relative_solution_error']) <= limit for r in chosen),
                            forward_error_limit=limit,
                            total_iterations=sum(r['iterations'] for r in chosen),
                            maximum_forward_error=max(float(r['relative_solution_error']) for r in chosen),
                            maximum_true_residual=max(float(r['relative_true_residual']) for r in chosen)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results/independent_numerical_parameters')
    parser.add_argument('--revision', default=DEFAULT_REVISION)
    parser.add_argument('--compiler', default='g++')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    flags = ['-std=c++11', '-O2', '-Wall', '-Wextra', '-Werror']
    rows, raw, cache = [], [], {}
    with tempfile.TemporaryDirectory(prefix='independent_gmres_') as directory:
        tree = Path(directory)
        commit, dependencies = snapshot(tree, args.revision)
        header = tree / 'base_matrix/base_matrix_gmres.hpp'
        original = header.read_text()
        original_hash = hashlib.sha256(header.read_bytes()).hexdigest()
        pattern = r'static_cast<Real>\(GMRES_BREAKDOWN_EPSILON_MULTIPLIER\)\s*\*\s*std::numeric_limits<Real>::epsilon\(\)'
        modified, count = re.subn(pattern, 'static_cast<Real>(TEST_BREAKDOWN_BETA)', original)
        if count != 1:
            raise ValueError('Expected one GMRES breakdown expression')
        header.write_text(modified)
        with localcontext() as context:
            context.prec = 80
            for beta in BETAS:
                for math_config in ['default', 'std']:
                    label = 'gmres_' + beta + '_' + math_config
                    binary = tree / label
                    command = [args.compiler] + flags + ['-DTEST_BREAKDOWN_BETA=' + beta]
                    if math_config == 'std': command += ['-DBASE_MATH_USE_STD_MATH_']
                    command += ['-I' + str(tree / 'base_matrix')]
                    command += ['-I' + str(tree / include) for include in INCLUDES]
                    command += [str(SOURCE), '-o', str(binary)]
                    build = subprocess.run(command, text=True, capture_output=True)
                    if build.returncode: raise RuntimeError(build.stdout + build.stderr)
                    text = subprocess.check_output([str(binary)], text=True)
                    raw.append(dict(beta=beta, math=math_config, output=text))
                    for row in evaluate(text, beta, math_config, cache):
                        row['beta'] = row.pop('epsilon')
                        row['tau_label'] = row['configuration'].split('_tau')[1]
                        for field in ['relative_solution_error', 'relative_true_residual', 'reported_residual']:
                            if not math.isfinite(float(row[field])):
                                raise ValueError('Nonfinite result')
                        rows.append(row)
                    print(label, 'cases=', len(text.splitlines()), flush=True)
            # Same arithmetic and decisions in the selectable for-loop path.
            loop_command = [args.compiler] + flags + ['-DTEST_BREAKDOWN_BETA=8e-16',
                '-DBASE_MATRIX_USE_FOR_LOOP_OPERATION_', '-I' + str(tree / 'base_matrix')]
            loop_command += ['-I' + str(tree / include) for include in INCLUDES]
            loop_binary = tree / 'gmres_for_loop'
            loop_command += [str(SOURCE), '-o', str(loop_binary)]
            subprocess.run(loop_command, check=True, capture_output=True, text=True)
            loop_text = subprocess.check_output([str(loop_binary)], text=True)
            reference = next(r['output'] for r in raw if r['beta'] == '8e-16' and r['math'] == 'default')
            if loop_text != reference:
                raise RuntimeError('For-loop results differ from template results')
    storage = {}
    for row in rows:
        storage.setdefault((row['type'], row['scalar'], row['n'], row['k']), set()).add(row['object_bytes'])
    assert all(len(sizes) == 1 for sizes in storage.values())
    write_csv(output / 'gmres_cases.csv', rows)
    write_csv(output / 'gmres_summary.csv', summary(rows))
    (output / 'gmres_exact_inputs.json').write_text(json.dumps(raw, indent=2))
    metadata = dict(commit=commit, submodule_revisions=dependencies,
        compiler=subprocess.check_output([args.compiler, '--version'], text=True).splitlines()[0],
        flags=flags, beta_values=BETAS, tau_values=TAUS, case_count=len(rows),
        precision=80, verification_precision=120, finite_outputs=True, iteration_cap='K=N, cycles=1',
        object_storage_unchanged=True, for_loop_equivalence='beta=8e-16/default; all112 rows byte-identical',
        gmres_header_sha256=original_hash, source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation='beta is the final Arnoldi relative coefficient. tau is the public relative_tolerance; absolute_tolerance remains zero. No epsilon factor or automatic machine-epsilon floor is applied to either.',
        oracle='Decimal solution of exact hexadecimal matrix/RHS, verified at80/120 digits. Production headers unchanged.')
    (output / 'gmres_metadata.json').write_text(json.dumps(metadata, indent=2))
    print('Wrote', len(rows), 'cases', flush=True)


if __name__ == '__main__':
    main()
