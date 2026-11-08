#!/usr/bin/env python3
"""Collect local readiness evidence; pretrained CUDA training is not implemented.
Python 3.10+ standard library only. Source weights are read, never modified.
"""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def parse_gpus(text, reserve_mib):
    """Separate total/free/reserved budgets; do not subtract OS usage twice."""
    devices = []
    for row in csv.reader(text.splitlines()):
        if not row:
            continue
        if len(row) != 8:
            raise ValueError("unexpected nvidia-smi CSV column count")
        index, uuid, name, bus, driver, total, used, free = [s.strip() for s in row]
        total, used, free = int(total), int(used), int(free)
        if min(total, used, free) < 0 or total == 0 or max(used, free) > total:
            raise ValueError("invalid GPU memory observation")
        devices.append(dict(index=index, uuid=uuid, name=name, pci_bus_id=bus,
                            driver=driver, total_mib=total, used_mib=used, free_mib=free,
                            reserve_mib=reserve_mib,
                            observed_training_budget_mib=max(0, min(free, total-reserve_mib))))
    return devices


def run(command, destination, label, timeout=1800):
    record = dict(command=[str(p) for p in command])
    try:
        result = subprocess.run(record['command'], cwd=ROOT, capture_output=True, timeout=timeout)
        stdout, stderr = result.stdout, result.stderr
        record.update(exit_code=result.returncode, status='passed' if result.returncode == 0 else 'failed')
    except (OSError, subprocess.TimeoutExpired) as error:
        stdout = getattr(error, 'stdout', b'') or b''
        stderr = str(error).encode('utf-8', errors='replace')
        record.update(exit_code=None, status='unavailable')
    out = destination / (label + '.stdout.log')
    err = destination / (label + '.stderr.log')
    out.write_bytes(stdout)
    err.write_bytes(stderr)
    record.update(stdout_log=out.name, stderr_log=err.name)
    return record, stdout.decode('utf-8', errors='replace')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path, help='New directory for local evidence; never overwritten')
    parser.add_argument('--model-dir', type=Path, help='Optional local HF SafeTensors directory, single-file or indexed shards')
    parser.add_argument('--tesslot-bin', type=Path, help='Use an existing binary instead of building with Cargo')
    parser.add_argument('--reserve-mib', type=int, default=1024, help='VRAM left outside training per GPU (default: 1 GiB)')
    parser.add_argument('--export', type=Path, help='Optional native export to validate, not train')
    parser.add_argument('--tokenizer-identity', type=Path, help='Trusted native identity JSON for --export')
    parser.add_argument('--cpu-smoke', action='store_true', help='Run tiny fixture training, checkpoint stop/resume and compare weights')
    parser.add_argument('--cuda-probes', action='store_true', help='Build/run optional small CUDA numerical gates; not Qwen training')
    args = parser.parse_args()
    if args.reserve_mib < 0 or bool(args.export) != bool(args.tokenizer_identity):
        parser.error('reserve must be nonnegative; export and tokenizer-identity must be supplied together')
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    report = dict(format='tesslot-local-readiness/1', observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  host=dict(system=platform.system(), machine=platform.machine(), cpu_logical_count=os.cpu_count()),
                  reserve_mib=args.reserve_mib, observations={'output_disk_free_bytes': shutil.disk_usage(destination).free}, checks={}, pretrained_training_ready=False,
                  blockers=['pretrained Qwen execution/Kidra conversion is unimplemented',
                            'model-level CUDA/precision and large-model checkpoint gates remain open'])
    gpu, text = run(['nvidia-smi', '--query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total,memory.used,memory.free',
                     '--format=csv,noheader,nounits'], destination, 'gpu', timeout=30)
    report['observations']['gpu_discovery'] = gpu
    try:
        report['observations']['gpus'] = parse_gpus(text, args.reserve_mib) if gpu['status'] == 'passed' else []
    except ValueError as error:
        gpu.update(status='invalid_output', error=str(error))
        report['observations']['gpus'] = []
    # Device UUID/bus ID distinguish physical devices without assuming CUDA ordinals.
    report['observations']['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')
    provenance, text = run(['git', 'rev-parse', 'HEAD'], destination, 'git-head', timeout=30)
    report['observations']['git_head'] = text.strip() if provenance['status'] == 'passed' else None
    provenance, _ = run(['git', 'status', '--short'], destination, 'git-status', timeout=30)
    report['observations']['git_status'] = provenance
    binary = args.tesslot_bin.resolve() if args.tesslot_bin else None
    if binary is None:
        build, text = run(['cargo', 'build', '-p', 'tesslot-cli', '--locked', '--message-format=json'], destination, 'build')
        report['checks']['build'] = build
        if build['status'] == 'passed':
            for line in text.splitlines():
                try:
                    artifact = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if artifact.get('reason') == 'compiler-artifact' and artifact.get('target', {}).get('name') == 'tesslot' and artifact.get('executable'):
                    binary = Path(artifact['executable'])
    if binary and binary.is_file():
        check, _ = run([binary, '--version'], destination, 'binary-version', timeout=30)
        report['checks']['binary_version'] = check
        if args.model_dir:
            check, _ = run([binary, 'inspect-model-directory', args.model_dir.resolve()], destination, 'source-inventory')
            report['checks']['source_headers'] = check
        if args.export:
            check, _ = run([binary, 'validate', args.export.resolve(), '--tokenizer', args.tokenizer_identity.resolve()], destination, 'dataset')
            report['checks']['dataset_validation'] = check
        for name in ['preflight-0.6b-3060.json', 'preflight-0.8b-provisional-3060.json']:
            check, _ = run([binary, 'preflight-training', ROOT/'examples/qwen3-0.6b.config.json', '--scenario', ROOT/'examples'/name], destination, name)
            try:
                preflight = json.loads((destination/(name+'.stdout.log')).read_text())
                if check['status'] == 'failed' and preflight['memory']['budgets_sufficient_for_known_bytes'] is False:
                    check['status'] = 'insufficient_budget'
            except (ValueError, KeyError, TypeError):
                if check['status'] == 'passed':
                    check['status'] = 'invalid_output'
            # Static shortfall is a report outcome; actual command failures stay failures.
            report['checks'][name] = dict(**check, purpose='static scenario accounting; not measured admission')
        if args.cpu_smoke:
            common = [binary, 'train', ROOT/'fixtures/autograd.jsonl', '--tokenizer', ROOT/'fixtures/tokenizer-identity.json',
                      '--training-config', ROOT/'examples/train-tiny.json']
            steps = [('cpu-continuous', ['--output', destination/'continuous.safetensors']),
                     ('cpu-stop', ['--output', destination/'resumed.safetensors', '--checkpoint-dir', destination/'checkpoints', '--stop-after-epochs', '1']),
                     ('cpu-resume', ['--output', destination/'resumed.safetensors', '--checkpoint-dir', destination/'checkpoints', '--resume'])]
            for label, options in steps:
                check, _ = run(common + options, destination, label)
                report['checks'][label] = check
                if check['status'] != 'passed':
                    break
            first, second = destination/'continuous.safetensors', destination/'resumed.safetensors'
            report['checks']['cpu_restart_identical_weights'] = dict(status='passed' if first.exists() and second.exists() and first.read_bytes() == second.read_bytes() else 'failed',
                                                                    scope='tiny CPU fixture only; no pretrained weights or GPU')
    else:
        report['checks']['binary'] = dict(status='unavailable', error='No usable tesslot binary; inspect the build log or supply --tesslot-bin')
    if args.cuda_probes:
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if report['observations']['gpus'] and visible not in ('', '-1'):
            check, text = run(['cargo', 'test', '-p', 'tesslot-backend-probe', '--features', 'cuda', '--test', 'cuda', '--locked', '--', '--ignored', '--nocapture'], destination, 'cuda-probes')
            if check['status'] == 'passed' and 'SKIP' in text + (destination/'cuda-probes.stderr.log').read_text(errors='replace'):
                check['status'] = 'partial_or_skipped'
            report['checks']['cuda_probes'] = check
        else:
            report['checks']['cuda_probes'] = dict(status='skipped', reason='NVIDIA discovery unavailable or CUDA_VISIBLE_DEVICES disables CUDA')
    report['scope'] = 'Source metadata, static estimates and explicitly requested small probes only. This does not start pretrained training.'
    (destination/'readiness.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(report=str(destination/'readiness.json'), pretrained_training_ready=False, checks={name: check['status'] for name, check in report['checks'].items()}), indent=2))
    # Static insufficiency and unimplemented pretrained execution are report outcomes;
    # actual requested validation/probe/smoke failures must produce a failing exit.
    failures = [check for name, check in report['checks'].items() if check['status'] in ('failed', 'unavailable', 'invalid_output')]
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
