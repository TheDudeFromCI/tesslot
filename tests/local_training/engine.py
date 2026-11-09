#!/usr/bin/env python3
"""Shared execution and verification for opt-in pretrained regression tests."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import tarfile

from .process import ProbeRunner
from .placement import make_plan, parameter_layout, layout_admission, cuda_model_gates, visible_plan

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = ('config.json', 'model.safetensors', 'tokenizer.json', 'conversion.json')
LIMIT_STATUSES = {'reserve_exceeded', 'out_of_memory', 'insufficient_budget', 'unsupported_by_runtime'}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parameter_bytes(path):
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('truncated SafeTensors length')
        length = struct.unpack('<Q', prefix)[0]
        if not 2 <= length <= 16*1024*1024:
            raise ValueError('invalid SafeTensors header length')
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise ValueError('SafeTensors header must be an object')
    total = 0
    for name, tensor in header.items():
        if name == '__metadata__':
            continue
        if not isinstance(tensor, dict) or tensor['dtype'] != 'F32' or not isinstance(tensor['shape'], list) or not all(type(d) is int and d > 0 for d in tensor['shape']):
            raise ValueError('sweep requires the converted FP32 artifact')
        total += 4*math.prod(tensor['shape'])
    if total <= 0:
        raise ValueError('empty converted weights')
    return total


def verify_fixtures(directory, conversion):
    manifest = json.loads((directory/'manifest.json').read_text())
    if (manifest['format'] != 'tesslot-training-sweep-fixtures/1'
            or manifest['revision'] != 'd2c8083536d56a33d876e2650ae855722d491566'
            or manifest['tokenizer'] != conversion['tokenizer_conversion']['target']
            or manifest['row_limit'] != 512 or conversion['windows'] < 24):
        raise ValueError('fixture compiler, tokenizer, window capacity or row bound mismatch')
    for name, expected in manifest['files'].items():
        path = directory/name
        if path.resolve().parent != directory.resolve() and path.resolve().parent != (directory/'sources').resolve():
            raise ValueError('fixture path must stay in its declared directory')
        if sha256(path) != expected:
            raise ValueError('fixture checksum mismatch: '+name)
    return manifest


def events(output, name):
    result = [json.loads(line) for line in (output/(name+'.stdout.log')).read_text().splitlines() if line.strip()]
    if not all(isinstance(e, dict) for e in result):
        raise ValueError('native stdout must contain JSON objects')
    return result


def validate_training(output, name, start, steps, optimizer, saved, expected_samples):
    data = events(output, name)
    updates = [e for e in data if e.get('event') == 'optimizer_updated']
    final = [e for e in data if e.get('event') == ('checkpoint_saved' if saved else 'training_completed')]
    if len(updates) != steps or len(final) != 1:
        raise ValueError('missing updates or completion event')
    for i, event in enumerate(updates):
        metric = event['metrics']
        if metric['step'] != start+i+1 or not all(math.isfinite(metric[k]) for k in ('loss', 'gradient_norm')) or metric['gradient_norm'] <= 0:
            raise ValueError('nonfinite, zero-gradient or inconsistent update')
    last = final[0]
    if (last['step'] != start+steps or last['state']['completed_samples'] != start+steps
            or last['optimizer'] != optimizer or last['trained_weights_saved'] is not saved
            or last['dataset_samples'] != expected_samples or last['recomputation'] is not True):
        raise ValueError('optimizer, cursor, sample count or recomputation mismatch')
    validation = [e for e in data if e.get('event') == 'validation_evaluated']
    for value in validation:
        if not math.isfinite(value['loss']) or value['samples'] <= 0 or value['effective_weight'] <= 0:
            raise ValueError('nonfinite or empty held-out evaluation')
    return dict(updates=updates, validation=validation, completed_step=last['step'])


def checkpoint(output, name, step, dataset_hash, optimizer):
    directory = output/name
    manifest = json.loads((directory/'manifest.json').read_text())
    digest = sha256(directory/'model.safetensors')
    if (manifest['weights_sha256'] != digest or manifest['step'] != step
            or manifest['remix']['completed_samples'] != step
            or manifest['remix']['dataset_sha256'] != dataset_hash or manifest['optimizer'] != optimizer):
        raise ValueError('checkpoint checksum, optimizer or dataset cursor mismatch')
    if optimizer == 'sgd-no-momentum/1' and (manifest.get('adamw') is not None or (directory/'optimizer.safetensors').exists()):
        raise ValueError('unexpected optimizer moments for SGD')
    return manifest


def finish_status(report):
    stages = list(report['capacity'].values())+list(report['pilot'].values())+list(report.get('diagnostics', {}).values())+[report['adamw']]
    if report.get('error') or any(s.get('status') in {'failed', 'timed_out'} for s in stages):
        return 'failed', 1
    if not report['checks'].get('pilot_restart_identical') or any(s.get('status') in LIMIT_STATUSES or s.get('status') == 'not_run' for s in stages):
        return 'completed_with_limits', 2
    return 'passed', 0


def archive_logs(output, destination):
    with tarfile.open(destination, 'x:gz') as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and not path.is_symlink() and path.suffix in {'.json', '.jsonl', '.log'}:
                archive.add(path, arcname=output.name+'/'+str(path.relative_to(output)), recursive=False)


def collect(args, output):
    model, fixtures = args.model_dir.expanduser().resolve(), args.fixtures_dir.expanduser().resolve()
    build_cuda = bool(args.gpu_uuid)
    report = dict(format='tesslot-pretrained-tests/1', status='running',
                  backend='cpu' if args.cpu else 'cuda', reserve_mib=args.reserve_mib,
                  host_reserve_mib=args.host_reserve_mib,
                  capacity_policy='require_selected' if getattr(args,'strict_capacity',False) or args.capacity_case else 'discover_limits',
                  pretrained_training_ready=False, capacity={}, pilot={}, adamw=dict(status='not_run'),
                  checks={}, setup={}, diagnostics={},
                  memory_scope='Sampled whole-device GPU and host availability; transient peaks may be missed.')
    source_hashes = None
    runner = ProbeRunner(ROOT, output, args.gpu_uuid, args.reserve_mib, args.timeout_seconds, args.host_reserve_mib)
    def save():
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    def run(name, command):
        print(json.dumps(dict(stage=name, status='starting')), flush=True)
        report['current_stage'] = name
        save()
        result = runner.run(name, command)
        report.setdefault('processes', {})[name] = result
        save()
        return result
    try:
        config = json.loads((model/'config.json').read_text())
        conversion = json.loads((model/'conversion.json').read_text())
        manifest = verify_fixtures(fixtures, conversion)
        cases = {c['sample_id']: c for c in manifest['capacity']}
        selected = args.capacity_case or list(cases)
        if len(set(selected)) != len(selected) or any(c not in cases for c in selected):
            raise ValueError('unknown or duplicate capacity case')
        report['capacity'] = {name: dict(status='not_run', workload=cases[name]) for name in selected}
        report['pilot'] = {name: dict(status='not_run') for name in ['pilot-continuous', 'pilot-stopped', 'pilot-resumed']}
        weights = parameter_bytes(model/'model.safetensors')
        free_disk, required_disk = shutil.disk_usage(output).free, 4*(model/'model.safetensors').stat().st_size+512*1024*1024
        report['disk'] = dict(free_bytes=free_disk, required_free_bytes=required_disk)
        if free_disk < required_disk:
            raise ValueError('insufficient disk space for three pilot checkpoints and serialization margin')
        source_hashes = {name: sha256(model/name) for name in SOURCE_FILES}
        report['source_sha256'] = source_hashes
        report['parameters'] = weights//4
        plan, unsupported = make_plan(config, args)
        report['requested_runtime_plan'] = plan
        report['requested_gpu_uuids'] = args.gpu_uuid or []
        if not unsupported:
            plan, args.gpu_uuid, mapping = visible_plan(plan, args.gpu_uuid or [])
            report['visible_gpu_uuids'] = args.gpu_uuid
            report['cuda_ordinal_mapping'] = mapping
        report['runtime_plan'] = plan
        layout = parameter_layout(model/'model.safetensors', plan)
        report['parameter_bytes_per_owner'] = layout
        report['profile_status'] = 'unsupported' if unsupported else 'supported'
        if unsupported:
            report['profile_reason'] = unsupported
            return report
        args.cpu = plan['root']['kind'] == 'cpu'
        report['backend'] = 'cpu' if args.cpu else 'cuda'
        runner = ProbeRunner(ROOT, output, args.gpu_uuid, args.reserve_mib, args.timeout_seconds, args.host_reserve_mib)
        report['baseline'] = runner.observe()
        report['sgd_state_admission'] = layout_admission(layout, report['baseline'], args.gpu_uuid or [], args.reserve_mib, args.host_reserve_mib, 'sgd')
        host_free = report['baseline'].get('host_available_mib')
        host_import_available = max(0, host_free-args.host_reserve_mib)*1024*1024 if host_free is not None else None
        report['host_import_lower_bound_bytes'] = 2*weights
        if report['sgd_state_admission']['known_oversized'] or (host_import_available is not None and 2*weights > host_import_available):
            report.update(profile_status='insufficient_budget', profile_reason='Known full-training state or dense host-import storage exceeds a device/host budget; this training profile is not executed.')
            return report
        (output/'runtime.json').write_text(json.dumps(plan, indent=2)+'\n')
        if args.tesslot_bin:
            binary = str(args.tesslot_bin.expanduser().resolve())
        else:
            command = ['cargo', 'build', '-p', 'tesslot-cli', '--locked', '--message-format=json']
            if build_cuda:
                command += ['--features', 'cuda']
            result = run('build', command)
            report['setup']['build'] = result
            if result['status'] != 'passed':
                raise RuntimeError('build failed; training stages were not launched')
            binaries = [e['executable'] for e in events(output, 'build') if e.get('reason') == 'compiler-artifact' and e.get('target', {}).get('name') == 'tesslot' and e.get('executable')]
            if not binaries:
                raise ValueError('build produced no executable')
            binary = binaries[-1]
        report['binary_sha256'] = sha256(Path(binary))
        report['binary_path'] = binary
        if not args.cpu:
            backend = run('backend-gates', ['cargo','test','-p','tesslot-backend-probe','--features','cuda','--test','cuda','--locked','--','--ignored','--nocapture','--test-threads=1'])
            report['setup']['backend_gates'] = backend
            if backend['status'] != 'passed':
                raise RuntimeError('CUDA backend regression gates failed')
            for name in cuda_model_gates(plan['precision'], len(args.gpu_uuid)):
                gate = run(name, ['cargo','test','-p','tesslot-tensor','--features','cuda','--test','qwen_cuda',name,'--locked','--','--exact','--ignored','--nocapture'])
                report['setup'][name] = gate
                if gate['status'] != 'passed' or '1 passed;' not in (output/(name+'.stdout.log')).read_text():
                    raise RuntimeError('configured precision/placement gate failed: '+name)
            gate = run('embedding-gate', ['cargo', 'test', '-p', 'tesslot-tensor', '--features', 'cuda', '--lib',
                                          'cuda_embedding_gather_gradients', '--locked', '--', '--ignored', '--nocapture'])
            report['setup']['embedding_gate'] = gate
            if gate['status'] != 'passed' or '1 passed;' not in (output/'embedding-gate.stdout.log').read_text():
                raise RuntimeError('CUDA gather/backward gate failed; training stages were not launched')
            name = 'activation-gate'
            gate = run(name, ['cargo','test','-p','tesslot-tensor','--features','cuda','--lib',
                              'cuda_silu_large_magnitude_gradients','--locked','--','--ignored','--nocapture'])
            report['setup'][name] = gate
            if gate['status'] != 'passed' or '1 passed;' not in (output/(name+'.stdout.log')).read_text():
                raise RuntimeError('CUDA activation gradient gate failed; model-sized stages were not launched')
        if args.base_model_dir:
            base = args.base_model_dir.expanduser().resolve()
            for name, cmd in [('source-headers', [binary, 'inspect-model-directory', str(base)]),
                              ('base-forward', [binary, 'forward-qwen', '--input', str(base),
                                                '--tokens', str(json.loads((base/'config.json').read_text())['bos_token_id']),
                                                '--runtime', str(output/'runtime.json'), '--max-bytes', str(args.max_bytes)])]:
                report['diagnostics'][name] = run(name, cmd)
        else:
            report['diagnostics']['base-forward'] = dict(status='not_supplied', reason='Original HF directory not supplied with --base-model-dir')
        backward_fixture = fixtures/manifest['diagnostic']['path']
        common = [binary, 'backward-remix', '--input', str(model), '--export', str(backward_fixture),
                  '--target-prefix', '--runtime', str(output/'runtime.json'), '--max-bytes', str(args.max_bytes)]
        for name, cmd in [('converted-backward', common), ('single-sgd-update', common+['--sgd-step', '--learning-rate', str(args.learning_rate)])]:
            result = run(name, cmd)
            if result['status'] == 'passed':
                try:
                    value = json.loads((output/(name+'.stdout.log')).read_text())
                    if name == 'converted-backward':
                        summary = value['summary']
                        if not summary['all_gradients_finite'] or not math.isfinite(summary['gradient_norm']) or summary['gradient_norm'] <= 0 or value['updates'] != 0 or value['sample_cursor'] != 0:
                            raise ValueError('backward diagnostic must have finite nonzero gradients and unchanged counters')
                    else:
                        probe = value['probe']
                        if probe['updates'] != 1 or probe['sample_cursor'] != 1 or not probe['all_updates_finite'] or not probe['all_gradients_finite'] or not all(math.isfinite(probe[k]) for k in ['loss_before','loss_after','gradient_norm']) or value['trained_weights_saved'] is not False:
                            raise ValueError('SGD diagnostic must complete one finite ephemeral update')
                except (KeyError, ValueError, TypeError) as error:
                    result.update(status='failed', error=str(error))
            report['diagnostics'][name] = result
        # A full 24-window context exceeds the current explicit native row admission.
        requested = 24*config['max_position_embeddings']
        report['capacity']['full-context-admission'] = dict(status='unsupported_by_runtime' if requested > manifest['row_limit'] else 'not_run',
            requested_windows=24, requested_tokens_per_window=config['max_position_embeddings'],
            requested_live_tokens=requested, runtime_row_limit=manifest['row_limit'],
            reason='Full per-window context exceeds the 512-row trajectory admission; no GPU allocation attempted.'
                if requested > manifest['row_limit'] else 'Full configured context is not represented by the compiled matrix; not qualified.')

        def command(export, steps, optimizer='sgd', destination=None, previous=None, validation=None):
            result = [binary, 'train-remix-local', str(export), '--tokenizer-file', str(model/'tokenizer.json'),
                      '--steps', str(steps), '--optimizer', optimizer, '--learning-rate', str(args.learning_rate),
                      '--recompute', '--runtime', str(output/'runtime.json'), '--max-bytes', str(args.max_bytes)]
            result += ['--output', str(output/destination)] if destination else ['--no-checkpoint']
            result += ['--resume', str(output/previous)] if previous else ['--pretrained', str(model)]
            if validation:
                result += ['--validation-export', str(validation)]
            return result

        for name in selected:
            case = cases[name]
            if case['max_position'] >= config['max_position_embeddings']:
                report['capacity'][name].update(status='unsupported_by_runtime', reason='case exceeds model context')
                continue
            result = run(name, command(fixtures/case['path'], 1))
            result['workload'] = case
            if result['status'] == 'passed':
                try:
                    result['training'] = validate_training(output, name, 0, 1, 'sgd-no-momentum/1', False, 1)
                except (ValueError, KeyError, TypeError) as error:
                    result.update(status='failed', error=str(error))
            report['capacity'][name] = result
            save()
        # The pilot remains independent of larger capacity failures and uses its own small workloads.
        train = fixtures/manifest['pilot']['train']['path']
        validation = fixtures/manifest['pilot']['validation']['path']
        split = args.pilot_steps//2
        checkpoint_manifests = {}
        for name, steps, previous, start in [('pilot-continuous', args.pilot_steps, None, 0),
                                             ('pilot-stopped', split, None, 0),
                                             ('pilot-resumed', args.pilot_steps-split, 'pilot-stopped', split)]:
            if previous and report['pilot'][previous]['status'] != 'passed':
                report['pilot'][name].update(reason='stopped checkpoint was not validated')
                continue
            result = run(name, command(train, steps, destination=name, previous=previous, validation=validation))
            if result['status'] == 'passed':
                try:
                    result['training'] = validate_training(output, name, start, steps, 'sgd-no-momentum/1', True, 4)
                    evaluation = result['training']['validation']
                    if len(evaluation) != 2 or [v['when'] for v in evaluation] != ['before_training', 'after_training'] or any(v['samples'] != 4 or v['dataset_sha256'] != sha256(validation) for v in evaluation):
                        raise ValueError('missing or incorrect disjoint held-out evaluations')
                    checkpoint_manifests[name] = checkpoint(output, name, start+steps, sha256(train), 'sgd-no-momentum/1')
                    result['weights_sha256'] = checkpoint_manifests[name]['weights_sha256']
                except (ValueError, KeyError, TypeError, OSError) as error:
                    result.update(status='failed', error=str(error))
            report['pilot'][name] = result
            save()
        if all(s['status'] == 'passed' for s in report['pilot'].values()):
            continuous, stopped, resumed = [report['pilot'][name]['training'] for name in ['pilot-continuous', 'pilot-stopped', 'pilot-resumed']]
            equal = (checkpoint_manifests['pilot-continuous'] == checkpoint_manifests['pilot-resumed']
                     and continuous['updates'] == stopped['updates']+resumed['updates']
                     and continuous['validation'][0]['loss'] == stopped['validation'][0]['loss']
                     and stopped['validation'][1]['loss'] == resumed['validation'][0]['loss']
                     and continuous['validation'][1]['loss'] == resumed['validation'][1]['loss'])
            report['checks']['pilot_restart_identical'] = equal
            before, after = continuous['validation'][0]['loss'], continuous['validation'][1]['loss']
            report['held_out'] = dict(loss_before=before, loss_after=after, improved=after < before,
                                      purpose='Small synthetic copy/select/update/wait pilot; not broad behavioral validation')
            if not equal:
                report['error'] = 'pilot checkpoint, sample sequence, update metrics or validation changed across restart'
        # Transactional FP32 AdamW needs old/new moment and parameter maps live together.
        observed = runner.observe()
        admission = layout_admission(layout, observed, args.gpu_uuid or [], args.reserve_mib, args.host_reserve_mib)
        if not admission['admitted']:
            report['adamw'] = dict(status='insufficient_budget', executed=False, admission=admission,
                                   reason='Known transactional FP32 state exceeds a per-device budget after reserve, or available memory is unknown')
        else:
            name = 'adamw-two-step'
            smallest = min(cases.values(), key=lambda c: c['rows'])
            result = run(name, command(fixtures/smallest['path'], 2, optimizer='adamw'))
            result.update(executed=True, admission=admission)
            if result['status'] == 'passed':
                try:
                    result['training'] = validate_training(output, name, 0, 2, 'adamw-f32/1', False, 1)
                except (ValueError, KeyError, TypeError) as error:
                    result.update(status='failed', error=str(error))
            report['adamw'] = result
        # Recheck every fixture, including the held-out set, and all source artifact files.
        verify_fixtures(fixtures, conversion)
        report['checks']['fixtures_unchanged'] = True
    except Exception as error:
        # Preserve a failed report/archive even for an unexpected orchestration error.
        report['error'] = type(error).__name__+': '+str(error)
    finally:
        if source_hashes is not None:
            try:
                after = {name: sha256(model/name) for name in SOURCE_FILES}
                report['source_sha256_after'] = after
                changed = [name for name in SOURCE_FILES if source_hashes[name] != after[name]]
                report['source_files_changed'] = changed
                same = not changed
                report['checks']['source_files_unchanged'] = same
                if not same:
                    report['error'] = 'source artifact files changed'
            except OSError as error:
                report['error'] = str(error)
        report['sgd_pilot_qualified'] = all(report['checks'].get(k) is True for k in
            ['pilot_restart_identical', 'source_files_unchanged', 'fixtures_unchanged'])
        report['cuda_pilot_qualified'] = report['sgd_pilot_qualified'] and not args.cpu
        integrity = all(report['checks'].get(k) is True for k in ['source_files_unchanged', 'fixtures_unchanged'])
        report['adamw_update_qualified'] = report['adamw']['status'] == 'passed' and not args.cpu and integrity
        report['capacity_passed'] = [name for name, value in report['capacity'].items() if value['status'] == 'passed']
        report['limits'] = {name: dict(status=value['status'], reason=value.get('reason', value.get('error')))
                            for name, value in list(report['capacity'].items())+[('adamw', report['adamw'])]
                            if value['status'] in LIMIT_STATUSES}
        report['status'], status = finish_status(report)
        save()
    return report
