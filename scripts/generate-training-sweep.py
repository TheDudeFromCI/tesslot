#!/usr/bin/env python3
"""Development-only: compile bounded capacity cases and a disjoint mechanical pilot.
Uses the pinned real Trajectory Remix compiler; workstation sweep needs no Python packages.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

REVISION = 'd2c8083536d56a33d876e2650ae855722d491566'
CAPACITY = [(2, 1), (2, 16), (2, 64), (2, 128), (8, 1), (8, 16), (8, 32), (24, 1), (24, 16)]


def source(name, replay):
    return {'schema-version': '1.0', 'id': name,
            'uuid': str(uuid.uuid5(uuid.NAMESPACE_URL, 'https://tesslot.local/training-sweep/'+name)),
            'tick-policy': 'external-first',
            'defaults': {'window-headers': False, 'implicit-noop-loss-weight': 0.0}, 'replay': replay}


def opened(name, kind):
    return dict(op='open', window=name, **{'window-type': kind})


def append(name, content):
    return dict(op='append', window=name, source='user', content=content, loss=dict(mask='exclude'))


def generate(name, content, weight=1.0):
    return dict(op='generate', window=name, content=content, loss=dict(weight=weight))


def close(name, supervised=False):
    return dict(op='generate', window=name, content='<|kidra_eos|>',
                loss=dict(weight=0.25) if supervised else dict(mask='exclude'))


def capacity_source(tokenizer, windows, tokens, all_generating=False):
    names = [f'w{i:02d}' for i in range(windows)]
    encoded = tokenizer.encode('A.'*(tokens+1), add_special_tokens=False).ids[:tokens]
    content = tokenizer.decode(encoded, skip_special_tokens=False)
    if tokenizer.encode(content, add_special_tokens=False).ids != encoded or len(encoded) != tokens:
        raise ValueError('capacity payload must roundtrip to exactly the requested token count')
    initial = [opened(n, 'reasoning' if i < 2 else 'document') for i, n in enumerate(names)]
    initial += [append(n, content) for n in names]
    name = f'capacity-w{windows:02d}-t{tokens:03d}'+('-all-generating' if all_generating else '')
    predictions = [generate(n, 'A' if i % 2 == 0 else 'B') for i, n in enumerate(names[:windows if all_generating else 2])]
    return source(name, {
        '0': initial, '1': predictions,
        '2': [close(n) for n in names]})


def pilot_source(tokenizer, split, kind, letters):
    first, second = letters
    begin = [opened('request', 'chat'), opened('document', 'document'), opened('answer', 'chat')]
    instructions = {'copy': 'Copy the value.', 'select': 'Copy the second value.',
                    'update': 'Copy the updated value.', 'wait': 'Wait for the value, then copy it.'}
    documents = {'copy': f'Value: {first}.', 'select': f'First: {first}. Second: {second}.',
                 'update': f'Value: {first}.', 'wait': 'Pending.'}
    begin += [append('request', instructions[kind]), append('document', documents[kind])]
    replay = {'0': begin}
    tick = 1
    answer = first if kind in ('copy', 'wait') else second
    if kind == 'update':
        replay['1'] = [append('document', f' Updated: {second}.')]
        tick = 2
    elif kind == 'wait':
        replay['1'] = [generate('answer', '<|kidra_noop|>', 0.25)]
        replay['2'] = [append('document', f'Value: {first}.')]
        tick = 3
    text = answer+'.'
    replay[str(tick)] = [generate('answer', text)]
    end = tick+len(tokenizer.encode(text, add_special_tokens=False).ids)
    replay[str(end)] = [close('answer', True), close('document'), close('request')]
    return source('pilot-'+split+'-'+kind, replay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=upstream, text=True).strip() != REVISION:
        parser.error('upstream must be at pinned revision '+REVISION)
    sys.path.insert(0, str(upstream))
    from tokenizers import Tokenizer
    from trajectory_remix.compiler import Compiler, chunks, visible_to
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    compiler = Compiler(tokenizer, bos='<|kidra_bos|>', eos='<|kidra_eos|>', noop='<|kidra_noop|>', max_window_slots=24)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'sources').mkdir()
    manifest = dict(format='tesslot-training-sweep-fixtures/1',
                    generator='scripts/generate-training-sweep.py',
                    upstream='https://github.com/TheDudeFromCI/trajectory-remix', revision=REVISION,
                    tokenizer=dict(sha256=compiler.fingerprint, vocab_size=compiler.tokenizer.get_vocab_size(), special_tokens=compiler.ids),
                    row_limit=512, capacity=[], pilot={}, files={},
                    purpose='Synthetic capacity and copy/select/update/wait mechanics; not broad competency or tool-use qualification')

    def compile_one(authored, single_target=False):
        data = (json.dumps(authored, indent=2)+'\n').encode()
        name = 'sources/'+authored['id']+'.json'
        (out/name).write_bytes(data)
        manifest['files'][name] = hashlib.sha256(data).hexdigest()
        plan = compiler.compile(authored)
        assert len(plan['rows']) <= 512
        assert all(w['close'] is not None for w in plan['windows'])
        targets = [r for r in plan['rows'] if r['loss_weight'] > 0]
        assert len(targets) == 1 if single_target else len(targets) >= 2
        assert all(not visible_to(r, r, plan['windows']) for r in targets)
        if authored['id'].startswith('capacity-'):
            prefix = [r for r in plan['rows'] if r['durable'] and r['tick'] == 0]
            assert all(visible_to(targets[0], r, plan['windows']) for r in prefix)
            assert targets[0]['tick'] == targets[1]['tick'] == 1
            if authored['id'].endswith('-all-generating'):
                assert len(targets) == 24 and all(r['tick'] == 1 for r in targets)
                assert all(not visible_to(q, k, plan['windows']) for q in targets for k in targets)
        if authored['id'].endswith('-wait'):
            noop = next(r for r in targets if r['token_id'] == compiler.ids['noop'])
            future = next(r for r in plan['rows'] if r['phase'] == 'B' and r['tick'] == 2)
            assert not noop['durable'] and not visible_to(noop, future, plan['windows'])
        native = ''.join(json.dumps(c, sort_keys=True)+'\n' for c in chunks(plan, 1)).encode()
        summary = dict(sample_id=authored['id'], trajectory_uuid=authored['uuid'], rows=len(plan['rows']),
                       targets=len(targets), active_windows=len(plan['windows']),
                       max_position=max(r['position'] for r in plan['rows']))
        return native, summary

    for windows, tokens, all_generating in [(w, t, False) for w, t in CAPACITY]+[(24, 16, True)]:
        authored = capacity_source(tokenizer, windows, tokens, all_generating)
        native, summary = compile_one(authored)
        path = authored['id']+'.jsonl'
        (out/path).write_bytes(native)
        manifest['files'][path] = hashlib.sha256(native).hexdigest()
        manifest['capacity'].append(dict(summary, path=path, context_tokens_per_window=tokens,
                                         generating_windows=windows if all_generating else 2,
                                         live_durable_tokens_at_prediction=windows*(tokens+1)))
    all_ids = set()
    for split, pairs in [('train', ['AB', 'CD', 'EF', 'GH']), ('validation', ['IJ', 'KL', 'MN', 'OP'])]:
        payload = bytearray()
        summaries = []
        for kind, pair in zip(['copy', 'select', 'update', 'wait'], pairs):
            native, summary = compile_one(pilot_source(tokenizer, split, kind, pair))
            assert summary['trajectory_uuid'] not in all_ids
            all_ids.add(summary['trajectory_uuid'])
            payload.extend(native)
            summaries.append(summary)
        path = 'pilot-'+split+'.jsonl'
        (out/path).write_bytes(payload)
        manifest['files'][path] = hashlib.sha256(payload).hexdigest()
        manifest['pilot'][split] = dict(path=path, samples=summaries)
    diagnostic_source = json.loads((Path(__file__).resolve().parent.parent/'fixtures/pretrained-backward.source.json').read_text())
    native, summary = compile_one(diagnostic_source, single_target=True)
    path = 'minimal-backward.jsonl'
    (out/path).write_bytes(native)
    manifest['files'][path] = hashlib.sha256(native).hexdigest()
    manifest['diagnostic'] = dict(summary, path=path)
    identity_bytes = (json.dumps(manifest['tokenizer'], indent=2)+'\n').encode()
    (out/'tokenizer-identity.json').write_bytes(identity_bytes)
    manifest['files']['tokenizer-identity.json'] = hashlib.sha256(identity_bytes).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(output=str(out), capacity_cases=len(manifest['capacity']), pilot_samples=8, tokenizer=compiler.fingerprint)))


if __name__ == '__main__':
    main()
