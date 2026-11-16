#!/usr/bin/env python3
"""Compare immutable revisions with painted-stack and GCC call-chain probes.

Compilation and measurement can be scheduled separately to avoid interfering
with timing runs: --stage compile, then --stage measure, then --stage analyze.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from run_epsilon_gmres_sweep import snapshot, INCLUDES

DEFAULT_REVISIONS = [
    'last_879=879a31981d4cb99173939bd4a12df96bf8ccdf1a',
    'pre_constants_ab31=ab31b0d25f5b081435864541d882f67571db74dd',
    'current_b599=b599e8461d0a4b07aa007bc6376746744767e671']


def run(command, **kwargs):
    print(' '.join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compile_probes(args, output):
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    flags = ['-std=c++11', '-O2', '-Wall', '-Wextra', '-Werror',
             '-fstack-usage', '-fcallgraph-info=su', '-pthread']
    worker_flags = ['-mno-red-zone', '-fno-shrink-wrap', '-fno-shrink-wrap-separate']
    worker = build / 'peak_stack_worker.o'
    run([args.compiler, *flags, *worker_flags, '-c',
         HERE / 'peak_stack_worker.cpp', '-o', worker])
    metadata = dict(compiler=subprocess.check_output(
        [args.compiler, '--version'], text=True), platform=platform.platform(),
        machine=platform.machine(), flags=flags,
        worker_extra_flags=worker_flags, build_dir=str(build),
        harness_hashes={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in HERE.glob('peak_stack_*') if path.is_file()},
        variants=[])
    for setting in args.revisions:
        label, revision = setting.split('=', 1)
        directory = build / label
        headers = directory / 'headers'
        commit, dependencies = snapshot(headers, revision)
        for mode in args.loop_modes.split(','):
            if mode not in ('template', 'for'):
                raise ValueError('Unknown loop mode: ' + mode)
            variant = label + '_' + mode
            object_file = directory / (variant + '.o')
            executable = directory / variant
            options = [] if mode == 'template' else [
                '-DBASE_MATRIX_USE_FOR_LOOP_OPERATION_']
            include_flags = ['-I' + str(headers / 'base_matrix')]
            include_flags += ['-I' + str(headers / path) for path in INCLUDES]
            command = [args.compiler, *flags, *options, *include_flags,
                       '-c', HERE / 'peak_stack_probe.cpp', '-o', object_file]
            run(command)
            # Bind external functions before the measurement starts. The probe
            # also calls every operation once before spawning painted threads.
            run([args.compiler, '-pthread', object_file, worker, '-Wl,-z,now',
                 '-o', executable])
            for suffix in ('.su', '.ci'):
                shutil.copyfile(object_file.with_suffix(suffix),
                                output / (variant + suffix))
            metadata['variants'].append(dict(label=label, commit=commit,
                dependencies=dependencies, mode=mode, name=variant,
                executable=str(executable), compile_command=list(map(str, command))))
    # Disassembly makes the painter's fixed frame/no-call paint loop auditable.
    with (output / 'worker_disassembly.txt').open('w') as stream:
        run(['objdump', '-dr', worker], stdout=stream)
    shutil.copyfile(worker.with_suffix('.su'), output / 'worker.su')
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


def measure_probes(output):
    metadata = json.loads((output / 'metadata.json').read_text())
    for variant in metadata['variants']:
        with (output / (variant['name'] + '.raw.csv')).open('w') as stream:
            environment = dict(os.environ, LD_BIND_NOW='1')
            run([variant['executable']], stdout=stream, env=environment)


def blocks(text, keyword):
    """Read VCG records without mistaking braces inside quoted labels."""
    pattern = re.compile(r'\b' + keyword + r':\s*\{')
    for match in pattern.finditer(text):
        depth, quoted, escaped = 1, False, False
        start = match.end()
        for index in range(start, len(text)):
            char = text[index]
            if escaped:
                escaped = False
            elif char == '\\' and quoted:
                escaped = True
            elif char == '"':
                quoted = not quoted
            elif not quoted:
                depth += (char == '{') - (char == '}')
                if depth == 0:
                    yield text[start:index]
                    break


def attributes(block):
    return {match[0]: json.loads('"' + match[1] + '"') for match in
            re.findall(r'(\w+):\s*"((?:\\.|[^"\\])*)"', block)}


def call_graph(path):
    text = path.read_text()
    nodes, edges = {}, {}
    for block in blocks(text, 'node'):
        node = attributes(block)
        label = node.get('label', '')
        usage = re.search(r'(\d+) bytes \(([^)]+)\)', label)
        nodes[node['title']] = dict(label=label,
            bytes=int(usage[1]) if usage else None,
            usage=usage[2] if usage else 'external_or_unresolved')
    for block in blocks(text, 'edge'):
        edge = attributes(block)
        edges.setdefault(edge['sourcename'], set()).add(edge['targetname'])
    return nodes, edges


def chain_from(root, nodes, edges):
    cache = {}

    def visit(name, active):
        if name in active:
            return 0, [], {name + ': recursive cycle'}, set()
        if name in cache:
            return cache[name]
        node = nodes.get(name, dict(bytes=None, usage='missing', label=name))
        unknown, unbounded = set(), set()
        if node['bytes'] is None:
            unknown.add(name)
        if 'dynamic' in node['usage'] and 'bounded' not in node['usage']:
            unbounded.add(name)
        best, route = 0, []
        for target in sorted(edges.get(name, ())):
            child, child_route, missing, dynamic = visit(target, active | {name})
            unknown.update(missing)
            unbounded.update(dynamic)
            if child > best:
                best, route = child, child_route
        result = ((node['bytes'] or 0) + best, [name] + route,
                  unknown, unbounded)
        cache[name] = result
        return result

    return visit(root, set())


def analyze(output):
    metadata = json.loads((output / 'metadata.json').read_text())
    rows, chains = [], {}
    for variant in metadata['variants']:
        nodes, edges = call_graph(output / (variant['name'] + '.ci'))
        with (output / (variant['name'] + '.raw.csv')).open() as stream:
            raw = list(csv.DictReader(stream))
        for sample in raw:
            root = sample['root']
            roots = [name for name in nodes if name == root or
                     name.endswith(':' + root)]
            if len(roots) != 1:
                raise ValueError('Missing call graph root: ' + root)
            maximum, route, unknown, dynamic = chain_from(roots[0], nodes, edges)
            # GCC's frame usage may omit a leaf's x86-64 red-zone reservation.
            # A final 128-byte allowance covers it on a finite known call path.
            known_bound = maximum + 128
            complete = not unknown and not dynamic
            context = int(sample['context_bytes'])
            row = dict(revision=variant['label'], commit=variant['commit'],
                       loop=variant['mode'], **sample,
                       known_call_chain_bytes=maximum,
                       known_chain_plus_red_zone=known_bound,
                       call_graph_complete=int(complete),
                       context_plus_known_chain=context + known_bound,
                       unknown_calls=';'.join(sorted(unknown)),
                       unbounded_frames=';'.join(sorted(dynamic)))
            rows.append(row)
            chains[variant['name'] + ':' + root] = dict(
                route=[dict(symbol=name, **nodes.get(name, {})) for name in route],
                unknown_calls=sorted(unknown), unbounded_frames=sorted(dynamic))
            if complete and int(sample['stack_written_max']) > known_bound:
                raise ValueError('Paint exceeds complete static bound: ' + root)
    write_csv(output / 'peak_stack_cases.csv', rows)
    (output / 'call_chains.json').write_text(json.dumps(chains, indent=2) + '\n')
    reference_label = metadata['variants'][0]['label']
    reference = {(row['loop'], row['root']): row for row in rows
                 if row['revision'] == reference_label}
    comparisons = []
    for row in rows:
        before = reference[row['loop'], row['root']]
        comparisons.append(dict(revision=row['revision'], loop=row['loop'],
            root=row['root'], type=row['type'], n=row['n'],
            context_delta=int(row['context_bytes']) - int(before['context_bytes']),
            solver_delta=int(row['solver_bytes']) - int(before['solver_bytes']),
            stack_written_delta=int(row['stack_written_max']) - int(before['stack_written_max']),
            known_chain_delta=row['known_call_chain_bytes'] - before['known_call_chain_bytes'],
            total_written_delta=int(row['context_plus_written_max']) - int(before['context_plus_written_max']),
            total_known_chain_delta=row['context_plus_known_chain'] - before['context_plus_known_chain'],
            unknown_calls_changed=int(row['unknown_calls'] != before['unknown_calls'])))
    write_csv(output / 'peak_stack_comparison.csv', comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
        default=HERE.parent / 'results/performance_recheck_2026_11_17/memory')
    parser.add_argument('--build-dir', type=Path, default=Path('/tmp/matrix_peak_stack_audit'))
    parser.add_argument('--compiler', default='g++')
    parser.add_argument('--revisions', nargs='+', default=DEFAULT_REVISIONS)
    parser.add_argument('--loop-modes', default='template,for')
    parser.add_argument('--stage', choices=['all', 'compile', 'measure', 'analyze'], default='all')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.stage in ('all', 'compile'):
        compile_probes(args, output)
    if args.stage in ('all', 'measure'):
        measure_probes(output)
    if args.stage in ('all', 'analyze'):
        analyze(output)


if __name__ == '__main__':
    main()
