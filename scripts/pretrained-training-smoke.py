#!/usr/bin/env python3
"""Short full-parameter SGD run, saved checkpoints and a fresh-process restart comparison.
Python 3.10+ standard library. Uses the already-qualified single-target CUDA path.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from pretrained_common import runtime_plan

ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILES = ('config.json', 'model.safetensors', 'tokenizer.json', 'conversion.json')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def workload(path):
    """Bound the diagnostic before model loading; native parsing still validates semantics."""
    with path.open('rb') as stream:
        data = stream.read(1024*1024+1)
    if len(data) > 1024*1024:
        raise ValueError('training smoke export exceeds 1 MiB')
    chunks = [json.loads(line) for line in data.splitlines() if line.strip()]
    if not chunks or not all(isinstance(c, dict) for c in chunks):
        raise ValueError('training smoke requires one complete trajectory')
    first = chunks[0]
    if (any((c['sample_id'], c['trajectory_uuid']) != (first['sample_id'], first['trajectory_uuid']) for c in chunks)
            or any(c['chunk_count'] != len(chunks) or c['chunk_index'] != i for i, c in enumerate(chunks))):
        raise ValueError('training smoke requires exactly one complete ordered trajectory')
    rows = [r for c in chunks for r in c['rows']]
    if not 1 <= len(rows) <= 32 or sum(r['loss_weight'] > 0 for r in rows) != 1:
        raise ValueError('training smoke requires at most 32 rows and exactly one effective target')
    return dict(rows=len(rows), chunks=len(chunks), window_capacity=first['max_window_slots'],
                active_windows=len(first['windows']))


def checkpoint(path, expected_step, dataset_hash, learning_rate):
    """Verify the on-disk contract independently of the trainer's success exit code."""
    manifest = json.loads((path/'manifest.json').read_text())
    state = manifest['remix']
    weight_hash = sha256(path/'model.safetensors')
    if (manifest['format'] != 'tiny-qwen-remix-sgd/1'
            or manifest['optimizer'] != 'sgd-no-momentum/1'
            or manifest['step'] != expected_step or state['completed_samples'] != expected_step
            or state['dataset_sha256'] != dataset_hash
            or not math.isclose(manifest['learning_rate'], learning_rate, rel_tol=1e-7, abs_tol=0)
            or manifest['weights_sha256'] != weight_hash
            or manifest.get('adamw') is not None or (path/'optimizer.safetensors').exists()):
        raise ValueError('checkpoint checksum, SGD configuration, step or dataset cursor mismatch')
    return manifest, weight_hash


def validate_events(events, first_step, steps):
    updates = [e for e in events if e.get('event') == 'optimizer_updated']
    saved = [e for e in events if e.get('event') == 'checkpoint_saved']
    if len(updates) != steps or len(saved) != 1:
        raise ValueError('missing optimizer updates or saved-checkpoint event')
    metrics = [e['metrics'] for e in updates]
    if any(m['step'] != first_step+i+1 or m['gradient_norm'] <= 0
           or not all(math.isfinite(m[k]) for k in ('loss', 'gradient_norm'))
           for i, m in enumerate(metrics)):
        raise ValueError('nonfinite or inconsistent SGD update metrics')
    final = saved[0]
    loss = final['final_single_target_loss']
    if (final['step'] != first_step+steps or final['state']['completed_samples'] != first_step+steps
            or final['optimizer'] != 'sgd-no-momentum/1' or final['single_target'] is not True
            or final['dataset_samples'] != 1 or not math.isfinite(loss)):
        raise ValueError('invalid single-trajectory checkpoint event or final loss')
    return dict(updates=metrics, loss_before=metrics[0]['loss'], loss_after=loss,
                loss_decreased=loss < metrics[0]['loss'], step=final['step'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--export', type=Path, default=ROOT/'fixtures/pretrained-backward-24/trajectory.jsonl')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--steps', type=int, default=4, help='Total updates in each comparison branch, 2..16')
    parser.add_argument('--split-step', type=int, default=2, help='Stop/checkpoint boundary, strictly between zero and steps')
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--reserve-mib', type=int, default=1024)
    parser.add_argument('--max-bytes', type=int, default=4294967296, help='Per-file/model load budget, not a VRAM cap')
    args = parser.parse_args()
    if not 2 <= args.steps <= 16 or not 0 < args.split_step < args.steps:
        parser.error('use 2..16 steps and 0 < split-step < steps')
    if args.reserve_mib < 0 or args.max_bytes <= 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('invalid reserve, load budget or learning rate')
    model = args.model_dir.expanduser().resolve()
    export = args.export.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID')
    report = dict(format='tesslot-pretrained-training-smoke/1', status='failed',
                  optimizer='sgd-no-momentum/1', precision='tf32', reserve_mib=args.reserve_mib,
                  steps=args.steps, split_step=args.split_step, learning_rate=args.learning_rate,
                  pretrained_training_ready=False, samples=[], stages={}, checks={},
                  memory_scope='Sampled whole-device usage; includes other processes and may miss transient peaks.')
    child = None
    source_hashes = None
    status = 1

    def save_report():
        (out/'training-report.json').write_text(json.dumps(report, indent=2)+'\n')

    def sample(stage):
        result = subprocess.run(['nvidia-smi', '--id='+args.gpu_uuid,
                                 '--query-gpu=uuid,name,memory.total,memory.used,memory.free',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15, check=True)
        rows = list(csv.reader(result.stdout.splitlines()))
        if len(rows) != 1 or len(rows[0]) != 5 or rows[0][0].strip() != args.gpu_uuid:
            raise ValueError('GPU UUID must identify exactly one physical device')
        uuid, name, total, used, free = [s.strip() for s in rows[0]]
        observed = dict(stage=stage, time_unix=time.time(), uuid=uuid, name=name,
                        total_mib=int(total), used_mib=int(used), free_mib=int(free))
        report['samples'].append(observed)
        if observed['free_mib'] < args.reserve_mib:
            raise RuntimeError('sampled free VRAM fell below reserve; stopping training smoke')
        return observed

    def run_stage(name, command):
        nonlocal child
        data = dict(command=command, status='running')
        report['stages'][name] = data
        save_report()
        start = time.monotonic()
        try:
            with (out/(name+'.stdout.log')).open('w') as stdout, (out/(name+'.stderr.log')).open('w') as stderr:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
                while True:
                    sample(name)
                    if child.poll() is not None:
                        break
                    if time.monotonic()-start > 1800:
                        raise RuntimeError(name+' timed out after 30 minutes')
                    time.sleep(0.25)
            if child.returncode:
                raise RuntimeError(name+' failed; inspect '+name+'.stderr.log')
            data['status'] = 'passed'
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            data['process_exit_code'] = child.returncode if child is not None else None
            data['elapsed_seconds'] = time.monotonic()-start
            phases = []
            for line in (out/(name+'.stderr.log')).read_text(errors='replace').splitlines():
                try:
                    event = json.loads(line)
                    if isinstance(event, dict) and 'phase' in event:
                        phases.append(event)
                except ValueError:
                    pass
            data['phases'] = phases
            data['last_phase'] = phases[-1] if phases else None
            observed = [s for s in report['samples'] if s['stage'] == name]
            data['sampled_peak_device_used_mib'] = max((s['used_mib'] for s in observed), default=None)
            data['sampled_min_free_mib'] = min((s['free_mib'] for s in observed), default=None)
            if data['status'] == 'running':
                data['status'] = 'failed'
            child = None
            save_report()

    try:
        config = json.loads((model/'config.json').read_text())
        plan = runtime_plan(config, args.max_bytes)
        (out/'runtime.json').write_text(json.dumps(plan, indent=2)+'\n')
        # Three model-sized checkpoints are retained; allow one extra checkpoint plus margin.
        required = 4*(model/'model.safetensors').stat().st_size+512*1024*1024
        free = shutil.disk_usage(out).free
        report['disk'] = dict(free_bytes=free, required_free_bytes=required)
        if free < required:
            raise ValueError('insufficient disk space for three checkpoints and serialization margin')
        source_hashes = {name: sha256(model/name) for name in SOURCE_FILES}
        dataset_hash = sha256(export)
        report['source_sha256'] = source_hashes
        report['dataset_sha256'] = dataset_hash
        report['workload'] = workload(export)
        report['baseline'] = sample('baseline')
        run_stage('build', ['cargo', 'build', '-p', 'tesslot-cli', '--features', 'cuda', '--locked', '--message-format=json'])
        binary = None
        for line in (out/'build.stdout.log').read_text().splitlines():
            event = json.loads(line)
            if event.get('reason') == 'compiler-artifact' and event.get('target', {}).get('name') == 'tesslot' and event.get('executable'):
                binary = event['executable']
        if binary is None:
            raise RuntimeError('build produced no tesslot executable')
        run_stage('embedding-gate', ['cargo', 'test', '-p', 'tesslot-tensor', '--features', 'cuda', '--lib',
                                     'cuda_embedding_gather_gradients', '--locked', '--', '--ignored', '--nocapture'])
        if '1 passed;' not in (out/'embedding-gate.stdout.log').read_text():
            raise RuntimeError('CUDA gather gate did not execute exactly one passing test')
        report['checks']['embedding_gate'] = 'passed'
        manifests = {}
        for name, steps, previous in [('continuous', args.steps, None), ('stopped', args.split_step, None),
                                      ('resumed', args.steps-args.split_step, 'stopped')]:
            command = [binary, 'train-remix-local', str(export), '--tokenizer-file', str(model/'tokenizer.json'),
                       '--output', str(out/name), '--steps', str(steps), '--optimizer', 'sgd',
                       '--learning-rate', str(args.learning_rate), '--single-target', '--runtime', str(out/'runtime.json'),
                       '--max-bytes', str(args.max_bytes)]
            command += ['--resume', str(out/previous)] if previous else ['--pretrained', str(model)]
            print(json.dumps(dict(stage=name, status='starting')), flush=True)
            run_stage(name, command)
            events = [json.loads(line) for line in (out/(name+'.stdout.log')).read_text().splitlines() if line.strip()]
            report['stages'][name]['training'] = validate_events(events, args.split_step if previous else 0, steps)
            manifest, weight_hash = checkpoint(out/name, args.split_step if name == 'stopped' else args.steps,
                                                dataset_hash, args.learning_rate)
            manifests[name] = manifest
            report['stages'][name]['weights_sha256'] = weight_hash
            report['checks'][name] = 'passed'
            save_report()
        identical = report['stages']['continuous']['weights_sha256'] == report['stages']['resumed']['weights_sha256']
        report['checks']['restart_identical_weights'] = 'passed' if identical else 'failed'
        same_state = manifests['continuous'] == manifests['resumed']
        report['checks']['restart_identical_manifest'] = 'passed' if same_state else 'failed'
        if not identical or not same_state:
            raise ValueError('continuous and resumed checkpoints differ; restart equivalence is not established')
        if sha256(export) != dataset_hash:
            raise ValueError('export changed during smoke test')
        report['checks']['dataset_unchanged'] = 'passed'
        report['status'] = 'passed'
        status = 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError, RuntimeError) as error:
        report.update(status='failed', error=str(error))
    finally:
        if source_hashes is not None:
            try:
                unchanged = source_hashes == {name: sha256(model/name) for name in SOURCE_FILES}
                report['checks']['source_files_unchanged'] = 'passed' if unchanged else 'failed'
                if not unchanged:
                    raise ValueError('source model files changed during smoke test')
            except (OSError, ValueError) as error:
                report.update(status='failed', error=str(error))
                status = 1
        report['sampled_peak_device_used_mib'] = max((s['used_mib'] for s in report['samples']), default=None)
        report['sampled_min_free_mib'] = min((s['free_mib'] for s in report['samples']), default=None)
        save_report()
    print(json.dumps(dict(report=str(out/'training-report.json'), status=report['status'], checks=report['checks']), indent=2))
    return status


if __name__ == '__main__':
    raise SystemExit(main())
