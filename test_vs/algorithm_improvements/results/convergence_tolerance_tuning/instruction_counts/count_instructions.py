#!/usr/bin/env python3
"""Count actually executed x86-64 instructions with parent/child ptrace.

The arithmetic totals are dynamic instruction counts, not assembly line counts.
Only host instrumentation is added; both variants use one immutable Git snapshot.
"""
import argparse
from collections import Counter
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE.parents[2]))
from run_epsilon_gmres_sweep import snapshot, INCLUDES

REVISION = '6b1d627edf2de6e3e56deca59b83b37e942a97aa'
SOURCE = HERE / 'instruction_probe.cpp'
CASES = [f'{kernel}_{scalar}_{dtype}_{n}'
         for kernel in ['multiply', 'norm'] for scalar in ['real', 'complex']
         for dtype in ['float', 'double'] for n in [8, 16]]
CASES += [f'gmres_real_{dtype}_{n}' for dtype in ['float', 'double'] for n in [8, 16]]

PTRACE_PEEKTEXT, PTRACE_POKETEXT = 1, 4
PTRACE_CONT, PTRACE_SINGLESTEP = 7, 9
PTRACE_GETREGS, PTRACE_SETREGS = 12, 13
PTRACE_SETOPTIONS, PTRACE_O_EXITKILL = 0x4200, 0x00100000
LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.ptrace.restype = ctypes.c_long
LIBC.ptrace.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]


class Registers(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in
                'r15 r14 r13 r12 rbp rbx r11 r10 r9 r8 rax rcx rdx rsi rdi '
                'orig_rax rip cs eflags rsp ss fs_base gs_base ds es fs gs'.split()]


def ptrace(request, pid, address=0, data=0):
    ctypes.set_errno(0)
    result = LIBC.ptrace(request, pid, ctypes.c_void_p(address),
                        ctypes.c_void_p(data) if isinstance(data, int) else data)
    error = ctypes.get_errno()
    if result == -1 and error:
        raise OSError(error, os.strerror(error))
    return result


def registers(pid):
    result = Registers()
    ptrace(PTRACE_GETREGS, pid, data=ctypes.byref(result))
    return result


def wait_stop(pid):
    unused, status = os.waitpid(pid, 0)
    if not os.WIFSTOPPED(status):
        raise RuntimeError(f'Child exited before end marker: status={status}')
    return os.WSTOPSIG(status)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, records):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def build(directory):
    directory.mkdir(parents=True, exist_ok=True)
    tree = directory / 'headers'
    commit, dependencies = snapshot(tree, REVISION)
    flags = ['-std=c++11', '-O2', '-Wall', '-Wextra', '-Werror']
    metadata = dict(revision=commit, dependencies=dependencies, flags=flags,
        source_sha256=sha(SOURCE), runner_sha256_at_build=sha(__file__),
        compiler=subprocess.check_output(['g++', '--version'], text=True),
        objdump=subprocess.check_output(['objdump', '--version'], text=True).splitlines()[0],
        platform=platform.platform(), machine=platform.machine(), variants=[],
        scope='One apply call after one warmup, between noipa/noinline host markers; callees included.',
        parameters='Fixed 6b1d627/b599 library; GMRES relative=absolute=0, beta=8e-12, division_min=1e-25, K=N, cycles=1, passes=1.',
        caveat='Dedicated probe, not the previous watermark executable; x86-64 dynamic architectural instruction counts, not CPU micro-ops or elapsed time.',
        header_hashes={str(p.relative_to(tree)):sha(p) for p in sorted(tree.rglob('*.hpp'))})
    for variant in ['template', 'for']:
        binary = directory / ('instructions_' + variant)
        command = ['g++', *flags]
        if variant == 'for': command += ['-DBASE_MATRIX_USE_FOR_LOOP_OPERATION_']
        command += ['-I' + str(tree / name) for name in ['base_matrix', *INCLUDES]]
        command += [str(SOURCE), '-o', str(binary)]
        subprocess.run(command, check=True)
        metadata['variants'].append(dict(name=variant, binary=str(binary),
                                         command=command, binary_sha256=sha(binary)))
    (HERE / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


def disassemble(path, start=None, end=None):
    command = ['objdump', '-d', '-M', 'intel', '--no-show-raw-insn', '--wide']
    if start is not None:
        command += ['--start-address=' + hex(start), '--stop-address=' + hex(end)]
    command.append(str(path))
    text = subprocess.check_output(command, text=True)
    instructions, symbols, current = {}, {}, ''
    for line in text.splitlines():
        match = re.match(r'^([0-9a-f]+) <(.+)>:$', line)
        if match:
            current = match.group(2)
            symbols[current] = int(match.group(1), 16)
        match = re.match(r'^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$', line)
        if match:
            instructions[int(match.group(1), 16)] = dict(
                symbol=current, mnemonic=match.group(2), operands=match.group(3))
    return text, instructions, symbols


def mappings(pid):
    result = []
    for line in Path(f'/proc/{pid}/maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) < 6 or not parts[5].startswith('/'): continue
        first, last = [int(value, 16) for value in parts[0].split('-')]
        result.append((first, last, int(parts[2], 16), parts[5]))
    return result


def arithmetic(instruction):
    """Architectural scalar lanes, including unused packed lanes; not useful FLOPs."""
    mnemonic, operands = instruction['mnemonic'], instruction['operands']
    result = Counter()
    match = re.fullmatch(r'v?(add|sub|mul|div)(ss|sd|ps|pd)', mnemonic)
    if match:
        operation, suffix = match.groups()
        width = 512 if 'zmm' in operands else 256 if 'ymm' in operands else 128
        lanes = 1 if suffix in ['ss', 'sd'] else width // (32 if suffix == 'ps' else 64)
        result['fp_four_instruction'] = 1
        result['fp_' + operation + '_instruction'] = 1
        result['lane_' + operation] = lanes
        result['scalar_fp_instruction' if lanes == 1 else 'packed_fp_instruction'] = 1
    elif re.fullmatch(r'v?(hadd|hsub|addsub)(ps|pd)', mnemonic):
        suffix = mnemonic[-2:]
        width = 256 if 'ymm' in operands else 128
        lanes = width // (32 if suffix == 'ps' else 64)
        result['fp_four_instruction'] = 1
        result['packed_fp_instruction'] = 1
        if 'addsub' in mnemonic:
            result['lane_add'] = result['lane_sub'] = lanes // 2
        else:
            result['lane_add' if 'hadd' in mnemonic else 'lane_sub'] = lanes
    elif re.fullmatch(r'vf(n?madd|n?msub|maddsub|msubadd)(132|213|231)(ss|sd|ps|pd)', mnemonic):
        suffix = mnemonic[-2:]
        width = 512 if 'zmm' in operands else 256 if 'ymm' in operands else 128
        lanes = 1 if suffix in ['ss', 'sd'] else width // (32 if suffix == 'ps' else 64)
        result['fp_four_instruction'] = result['fma_instruction'] = 1
        result['lane_mul'] = lanes
        if 'maddsub' in mnemonic or 'msubadd' in mnemonic:
            result['lane_add'] = result['lane_sub'] = lanes // 2
        else:
            result['lane_sub' if 'msub' in mnemonic else 'lane_add'] = lanes
    elif re.fullmatch(r'f(add|subr?|mul|divr?)p?', mnemonic):
        operation = re.match(r'f(add|sub|mul|div)', mnemonic).group(1)
        result['fp_four_instruction'] = result['x87_four_instruction'] = 1
        result['lane_' + operation] = 1
    elif re.fullmatch(r'v?(sqrt|rsqrt|rcp)(ss|sd|ps|pd)', mnemonic):
        result['fp_sqrt_reciprocal_instruction'] = 1
    elif mnemonic.startswith(('comis', 'ucomis', 'vcomis', 'vucomis')):
        result['fp_compare_instruction'] = 1
    elif mnemonic in ['add', 'sub', 'imul', 'mul', 'div', 'idiv', 'inc', 'dec',
                      'adc', 'sbb', 'neg', 'lea', 'shl', 'shr', 'sar', 'sal',
                      'rol', 'ror', 'and', 'or', 'xor', 'not']:
        result['integer_address_bit_arithmetic_instruction'] = 1
    elif mnemonic in ['cmp', 'test']:
        result['integer_compare_instruction'] = 1
    elif mnemonic.startswith('j') or mnemonic.startswith('loop'):
        result['branch_instruction'] = 1
    elif mnemonic.startswith(('call', 'ret')):
        result['call_return_instruction'] = 1
    elif re.search(r'\b[xyz]mm\d+\b', operands):
        result['other_simd_instruction'] = 1
    else:
        result['other_instruction'] = 1
    # REP string instructions can stop once per repetition under single-step.
    # Preserve events explicitly instead of treating them as PMU retired counts.
    if mnemonic in ['rep', 'repe', 'repz', 'repne', 'repnz']:
        result['repeat_string_step_events'] = 1
    result['single_step_events'] = 1
    return result


def trace(binary, case, instructions, symbols, limit):
    process = subprocess.Popen([str(binary), 'trace', case], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    pid = process.pid
    try:
        assert wait_stop(pid) == signal.SIGSTOP
        ptrace(PTRACE_SETOPTIONS, pid, data=PTRACE_O_EXITKILL)
        maps = mappings(pid)
        # Both GCC PIE and fixed-address ELF are supported. Symbols carry the
        # ELF virtual address; PIE mappings add their load bias.
        elf_type = subprocess.check_output(['readelf', '-h', str(binary)], text=True)
        bias = min(first - offset for first, last, offset, path in maps
                   if path == str(binary)) if re.search(r'Type:\s+DYN', elf_type) else 0
        begin, end = symbols['instruction_count_begin'] + bias, symbols['instruction_count_end'] + bias
        saved = ptrace(PTRACE_PEEKTEXT, pid, begin) & ((1 << 64) - 1)
        ptrace(PTRACE_POKETEXT, pid, begin, (saved & ~255) | 0xcc)
        ptrace(PTRACE_CONT, pid)
        assert wait_stop(pid) == signal.SIGTRAP
        state = registers(pid)
        assert state.rip == begin + 1
        ptrace(PTRACE_POKETEXT, pid, begin, saved)
        state.rip = begin
        ptrace(PTRACE_SETREGS, pid, data=ctypes.byref(state))
        executed, decoded = Counter(), {}
        external = {}
        for step in range(limit):
            pc = state.rip
            if pc == end: break
            if pc not in decoded:
                relative = pc - bias
                if relative in instructions:
                    decoded[pc] = dict(instructions[relative], image=str(binary), address=hex(relative))
                else:
                    entry = next((r for r in maps if r[0] <= pc < r[1]), None)
                    if entry is None:
                        maps = mappings(pid)
                        entry = next((r for r in maps if r[0] <= pc < r[1]), None)
                    if entry is None: raise RuntimeError(f'No mapped image for PC {pc:x}')
                    first, last, offset, path = entry
                    address = pc - first + offset
                    unused, decoded_range, unused_symbols = disassemble(path, address, address + 4096)
                    for position, instruction in decoded_range.items():
                        decoded[position + first - offset] = dict(instruction, image=path, address=hex(position))
                    if pc not in decoded: raise RuntimeError(f'Cannot decode PC {pc:x}')
                    external[path] = sha(path)
            executed[pc] += 1
            ptrace(PTRACE_SINGLESTEP, pid)
            stop = wait_stop(pid)
            if stop != signal.SIGTRAP: raise RuntimeError(f'Unexpected signal {stop} at {pc:x}')
            state = registers(pid)
        else:
            raise RuntimeError(f'Instruction limit {limit} reached')
        ptrace(PTRACE_CONT, pid)
        stdout, stderr = process.communicate(timeout=30)
        if process.returncode != 0 or stderr:
            raise RuntimeError(f'Child failed: {process.returncode}: {stderr}')
        rows, totals, mnemonics = [], Counter(), Counter()
        for pc, count in sorted(executed.items()):
            instruction = decoded[pc]
            rows.append(dict(**instruction, single_step_visits=count))
            for key, value in arithmetic(instruction).items(): totals[key] += count * value
            mnemonics[instruction['mnemonic']] += count
        totals['lane_four_total'] = sum(totals['lane_' + op] for op in ['add', 'sub', 'mul', 'div'])
        totals['executed_distinct_sites'] = len(executed)
        totals['executed_distinct_fp_four_sites'] = sum(
            bool(arithmetic(decoded[pc])['fp_four_instruction']) for pc in executed)
        return stdout, rows, totals, mnemonics, external
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def collect(selected, limit):
    metadata = json.loads((HERE / 'metadata.json').read_text())
    if 'runner_sha256' in metadata:
        metadata['runner_sha256_at_build'] = metadata.pop('runner_sha256')
    metadata['runner_sha256_at_trace'] = sha(__file__)
    (HERE / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if sha(SOURCE) != metadata['source_sha256']:
        raise RuntimeError('Probe changed since build')
    records, reports = [], []
    for variant in metadata['variants']:
        binary = Path(variant['binary'])
        if sha(binary) != variant['binary_sha256']: raise RuntimeError('Binary changed since build')
        assembly, instructions, symbols = disassemble(binary)
        (HERE / (variant['name'] + '.asm')).write_text(assembly)
        for case in selected:
            native = subprocess.check_output([str(binary), 'native', case], text=True)
            stdout, rows, totals, mnemonics, external = trace(binary, case, instructions, symbols, limit)
            if stdout != native: raise RuntimeError('Trace changed numerical result: ' + case)
            stem = variant['name'] + '_' + case
            (HERE / (stem + '.output.txt')).write_text(stdout)
            write_csv(HERE / (stem + '.executed.csv'), rows)
            for key in ['fp_four_instruction', 'fp_add_instruction', 'fp_sub_instruction',
                        'fp_mul_instruction', 'fp_div_instruction', 'lane_add', 'lane_sub',
                        'lane_mul', 'lane_div', 'fma_instruction', 'scalar_fp_instruction',
                        'packed_fp_instruction', 'x87_four_instruction',
                        'fp_sqrt_reciprocal_instruction', 'fp_compare_instruction',
                        'integer_address_bit_arithmetic_instruction', 'integer_compare_instruction',
                        'branch_instruction', 'call_return_instruction', 'other_simd_instruction',
                        'other_instruction', 'repeat_string_step_events']:
                totals.setdefault(key, 0)
            records.append(dict(variant=variant['name'], case=case, **totals))
            reports.append(dict(variant=variant['name'], case=case, native_equals_trace=True,
                output_sha256=hashlib.sha256(stdout.encode()).hexdigest(),
                mnemonics=dict(sorted(mnemonics.items())), external_images=external,
                counts=dict(totals)))
            print(variant['name'], case, 'dynamic FP=', totals['fp_four_instruction'],
                  'lane operations=', totals['lane_four_total'], 'steps=', totals['single_step_events'], flush=True)
    # Keep a consistent CSV schema even when the first case has no FMA, etc.
    fields = ['variant', 'case'] + sorted(set().union(*(set(r) for r in records)) - {'variant', 'case'})
    records = [{key:row.get(key, 0) for key in fields} for row in records]
    write_csv(HERE / 'dynamic_counts.csv', records)
    comparisons = []
    for case in selected:
        first = (HERE / ('template_' + case + '.output.txt')).read_text()
        second = (HERE / ('for_' + case + '.output.txt')).read_text()
        first_lines, second_lines = first.splitlines(), second.splitlines()
        inputs_equal = first_lines[0] == second_lines[0]
        outputs_equal = first_lines[1] == second_lines[1]
        comparisons.append(dict(case=case, exact_inputs_equal=inputs_equal,
            bitwise_outputs_equal=outputs_equal, status_equal=first_lines[2:] == second_lines[2:]))
        if not inputs_equal: raise RuntimeError('Inputs differ across variants: ' + case)
    (HERE / 'trace_reports.json').write_text(json.dumps(reports, indent=2) + '\n')
    (HERE / 'validation.json').write_text(json.dumps(dict(
        cases=len(selected), traced_calls=len(records), source_sha256=sha(SOURCE),
        runner_sha256=sha(__file__), comparisons=comparisons,
        all_native_outputs_match_trace=True,
        method='PTRACE_SINGLESTEP, each stopped RIP decoded before execution, including callee instructions until end marker.'), indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['build', 'trace', 'all'], default='all')
    parser.add_argument('--build-dir', type=Path, default=Path('/tmp/matrix_instruction_counts'))
    parser.add_argument('--cases', nargs='+', choices=CASES, default=CASES)
    parser.add_argument('--instruction-limit', type=int, default=2000000)
    args = parser.parse_args()
    if platform.machine() != 'x86_64': raise RuntimeError('This ptrace register layout requires x86-64 Linux')
    if args.phase in ['build', 'all']: build(args.build_dir.resolve())
    if args.phase in ['trace', 'all']: collect(args.cases, args.instruction_limit)


if __name__ == '__main__':
    main()
