"""Hardware-independent placement validation and per-owner state accounting."""
import json
import copy
import math
import struct

from .process import runtime_plan


def owner_key(target):
    if target.get('kind') == 'cpu' and set(target) == {'kind'}:
        return 'cpu'
    if target.get('kind') == 'cuda' and set(target) == {'kind', 'index'} and type(target['index']) is int and target['index'] >= 0:
        return 'cuda:'+str(target['index'])
    raise ValueError('unsupported or malformed runtime target')


def visible_plan(plan, gpu_uuids):
    """Keep physical ownership while hiding devices unused by this particular profile."""
    plan = copy.deepcopy(plan)
    used = sorted({t['index'] for t in [plan['root']]+plan['layers'] if t['kind']=='cuda'})
    mapping = {old:new for new,old in enumerate(used)}
    for target in [plan['root']]+plan['layers']:
        if target['kind']=='cuda':
            target['index'] = mapping[target['index']]
    return plan, [gpu_uuids[i] for i in used], mapping


def make_plan(config, args):
    if args.runtime_plan:
        plan = json.loads(args.runtime_plan.expanduser().read_text())
        if getattr(args, 'precision', None) and args.precision != plan['precision']:
            raise ValueError('requested precision differs from the explicit runtime plan')
    else:
        plan = runtime_plan(config, args.max_bytes)
        plan['precision'] = getattr(args, 'precision', None) or 'f32'
        if args.cpu:
            plan.update(root=dict(kind='cpu'), layers=[dict(kind='cpu') for _ in plan['layers']])
        elif len(args.gpu_uuid or []) > 1:
            count = len(plan['layers'])
            plan['layers'] = [dict(kind='cuda', index=i*len(args.gpu_uuid)//count) for i in range(count)]
    if len(plan['layers']) != config['num_hidden_layers'] or type(plan['parameter_budget_bytes']) is not int or plan['parameter_budget_bytes'] <= 0:
        raise ValueError('runtime requires one owner per layer and a positive per-owner parameter budget')
    if plan['precision'] not in {'f32', 'tf32', 'bf16', 'f16'}:
        raise ValueError('unsupported runtime precision')
    owners = {owner_key(t) for t in [plan['root']]+plan['layers']}
    for owner in owners:
        if owner.startswith('cuda:') and int(owner.split(':')[1]) >= len(args.gpu_uuid or []):
            raise ValueError('runtime CUDA ordinal has no supplied physical GPU UUID')
    if 'cpu' in owners and len(owners) > 1:
        return plan, 'Mixed CPU/CUDA layer offload is rejected by the current runtime; no execution attempted.'
    if 'cpu' in owners and plan['precision'] != 'f32':
        raise ValueError('CPU execution requires f32 precision')
    return plan, None


def parameter_layout(path, plan):
    with path.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        if not 2 <= length <= 16*1024*1024:
            raise ValueError('invalid SafeTensors header length')
        header = json.loads(stream.read(length))
    usage = {}
    for name, tensor in header.items():
        if name == '__metadata__':
            continue
        target = plan['root']
        if name.startswith('model.layers.'):
            layer = int(name.split('.')[2])
            target = plan['layers'][layer]
        key = owner_key(target)
        usage[key] = usage.get(key, 0)+4*math.prod(tensor['shape'])
    return usage


def layout_admission(layout, observation, gpu_uuids, reserve_mib, host_reserve_mib, optimizer='adamw'):
    if optimizer not in {'adamw','sgd'}:
        raise ValueError('unknown optimizer accounting')
    copies, persistent = (7,4) if optimizer == 'adamw' else (3,2)
    devices = {d['uuid']: d for d in observation.get('devices', [])}
    records = {}
    for owner, weights in layout.items():
        if owner == 'cpu':
            free, reserve = observation.get('host_available_mib'), host_reserve_mib
        else:
            uuid = gpu_uuids[int(owner.split(':')[1])]
            free, reserve = devices[uuid]['free_mib'], reserve_mib
        available = max(0, (free-reserve)*1024*1024) if free is not None else None
        records[owner] = dict(parameter_bytes=weights, persistent_bytes=persistent*weights,
                              transactional_step_bytes=copies*weights, available_after_reserve_bytes=available)
    return dict(devices=records,
                admitted=all(d['available_after_reserve_bytes'] is not None and d['transactional_step_bytes'] <= d['available_after_reserve_bytes'] for d in records.values()),
                known_oversized=any(d['available_after_reserve_bytes'] is not None and d['transactional_step_bytes'] > d['available_after_reserve_bytes'] for d in records.values()),
                excludes='activations, recomputation accumulator, workspace, allocator, transfers and checkpoint/import staging')


def cuda_model_gates(precision, device_count):
    """Only request dtype gates for the configured profile, never assume tensor-core features."""
    names = ['qwen_cuda_'+precision]
    if device_count > 1:
        names.append('qwen_dual_cuda' if precision == 'f32' else 'qwen_dual_cuda_'+precision)
    return names
