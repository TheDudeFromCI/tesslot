#!/usr/bin/env python3
"""Base-Qwen forward or converted-Kidra backward smoke; no optimizer updates.
Python 3.10+ standard library. Requires prior CUDA qualification on this machine.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent


def runtime_plan(config, parameter_budget_bytes):
    layers = config['num_hidden_layers']
    token = config['bos_token_id']
    if not isinstance(layers, int) or not 1 <= layers <= 1024 or not isinstance(token, int) or not 0 <= token < config['vocab_size']:
        raise ValueError('invalid layer count or BOS ID')
    target = dict(kind='cuda', index=0)
    return dict(root=target, layers=[target.copy() for _ in range(layers)], precision='tf32', parameter_budget_bytes=parameter_budget_bytes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--backward-export', type=Path, help='Opt into one small native trajectory backward using a converted Kidra model')
    p.add_argument('--target-prefix', action='store_true', help='Backward only: exact single-target causal-prefix graph, excluding irrelevant prediction branches')
    p.add_argument('--gpu-uuid', required=True, help='Physical GPU UUID from nvidia-smi, normally the RTX 3060')
    p.add_argument('--reserve-mib', type=int, default=1024)
    p.add_argument('--max-bytes', type=int, default=4294967296, help='FP32 source/parameter budget, not a total VRAM cap')
    args = p.parse_args()
    if args.reserve_mib < 0 or args.max_bytes <= 0:
        p.error('reserve must be nonnegative and max-bytes positive')
    if args.target_prefix and not args.backward_export:
        p.error('--target-prefix requires --backward-export')
    model = args.model_dir.expanduser().resolve()
    config = json.loads((model/'config.json').read_text())
    plan = runtime_plan(config, args.max_bytes)
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID')
    mode = 'backward' if args.backward_export else 'forward'
    result = dict(format='tesslot-pretrained-'+mode+'/1', gpu_uuid=args.gpu_uuid, reserve_mib=args.reserve_mib,
                  precision='tf32', training=False, backward_requested=bool(args.backward_export), target_prefix=args.target_prefix, pretrained_training_ready=False, samples=[],
                  memory_scope='Sampled whole-device usage; includes other processes and may miss transient peaks.')

    def sample():
        r = subprocess.run(['nvidia-smi', '--id='+args.gpu_uuid, '--query-gpu=uuid,name,memory.total,memory.used,memory.free',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15, check=True)
        rows = list(csv.reader(r.stdout.splitlines()))
        if len(rows) != 1 or len(rows[0]) != 5 or rows[0][0].strip() != args.gpu_uuid:
            raise ValueError('GPU UUID must identify exactly one physical device')
        uuid, name, total, used, free = [s.strip() for s in rows[0]]
        row = dict(time_unix=time.time(), uuid=uuid, name=name, total_mib=int(total), used_mib=int(used), free_mib=int(free))
        result['samples'].append(row)
        return row

    status = 1
    child = None
    try:
        baseline = sample()
        result['baseline'] = baseline
        if baseline['free_mib'] <= args.reserve_mib:
            raise ValueError('not enough observed free VRAM to preserve reserve')
        # Build separately so compilation time does not look like model execution time.
        with (out/'build.stdout.log').open('w') as stdout, (out/'build.stderr.log').open('w') as stderr:
            build = subprocess.run(['cargo', 'build', '-p', 'tesslot-cli', '--features', 'cuda', '--locked', '--message-format=json'],
                                   cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=1800)
        if build.returncode:
            raise RuntimeError('CUDA CLI build failed; inspect build.stderr.log')
        binary = None
        for line in (out/'build.stdout.log').read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('reason') == 'compiler-artifact' and event.get('target', {}).get('name') == 'tesslot' and event.get('executable'):
                binary = event['executable']
        if binary is None:
            raise RuntimeError('build produced no tesslot executable')
        if args.target_prefix:
            # Exercise CUDA select/scatter-add backward on a tiny model before loading
            # model-sized weights. A missing/zero-test run is not a passing gate.
            with (out/'embedding-gate.stdout.log').open('w') as stdout, (out/'embedding-gate.stderr.log').open('w') as stderr:
                gate = subprocess.run(['cargo', 'test', '-p', 'tesslot-tensor', '--features', 'cuda', '--lib', 'cuda_embedding_gather_gradients', '--locked', '--', '--ignored', '--nocapture'],
                                      cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=1800)
            if gate.returncode or '1 passed;' not in (out/'embedding-gate.stdout.log').read_text():
                result['embedding_gate'] = 'failed'
                raise RuntimeError('CUDA embedding gather/backward gate failed; inspect embedding-gate logs')
            result['embedding_gate'] = 'passed'
        (out/'runtime.json').write_text(json.dumps(plan, indent=2)+'\n')
        if args.backward_export:
            command = [binary, 'backward-remix', '--input', str(model), '--export', str(args.backward_export.expanduser().resolve()),
                       '--runtime', str(out/'runtime.json'), '--max-bytes', str(args.max_bytes)]
            if args.target_prefix:
                command.append('--target-prefix')
        else:
            command = [binary, 'forward-qwen', '--input', str(model), '--tokens', str(config['bos_token_id']),
                       '--runtime', str(out/'runtime.json'), '--max-bytes', str(args.max_bytes)]
        result['command'] = command
        start = time.monotonic()
        with (out/(mode+'.stdout.log')).open('w') as stdout, (out/(mode+'.stderr.log')).open('w') as stderr:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            while True:
                observed = sample()
                if observed['free_mib'] < args.reserve_mib:
                    raise RuntimeError('sampled free VRAM fell below reserve; stopping probe')
                if child.poll() is not None:
                    break
                if time.monotonic()-start > 1800:
                    raise RuntimeError('probe timed out after 30 minutes')
                time.sleep(0.25)
            status = child.returncode
        result['elapsed_'+mode+'_seconds'] = time.monotonic()-start
        if status == 0:
            logits = json.loads((out/(mode+'.stdout.log')).read_text())
            if args.backward_export:
                if logits['summary']['all_gradients_finite'] is not True or logits['summary']['gradient_norm'] <= 0 or logits['updates'] != 0 or logits['optimizer_allocated'] is not False or logits['sample_cursor'] != 0:
                    raise ValueError('backward report must have finite nonzero gradients and no optimizer/cursor updates')
            elif logits['logits_shape'] != [1, config['vocab_size']]:
                raise ValueError('unexpected one-token logits shape')
            result[mode] = logits
        result['status'] = 'passed' if status == 0 else 'failed'
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, RuntimeError) as error:
        result.update(status='failed', error=str(error))
        status = 1
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        phase_log = out/(mode+'.stderr.log')
        phases = []
        if phase_log.exists():
            for line in phase_log.read_text(errors='replace').splitlines():
                try:
                    event = json.loads(line)
                    if isinstance(event, dict) and 'phase' in event:
                        phases.append(event)
                except ValueError:
                    continue
        result['phases'] = phases
        result['last_phase'] = phases[-1] if phases else None
        result['process_exit_code'] = child.returncode if child is not None else None
        result['sampled_peak_device_used_mib'] = max((s['used_mib'] for s in result['samples']), default=None)
        result['sampled_min_free_mib'] = min((s['free_mib'] for s in result['samples']), default=None)
        (out/(mode+'-report.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(report=str(out/(mode+'-report.json')), status=result['status'], training=False), indent=2))
    return status


if __name__ == '__main__':
    raise SystemExit(main())
