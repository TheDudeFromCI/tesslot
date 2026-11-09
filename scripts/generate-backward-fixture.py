#!/usr/bin/env python3
"""Compile the curated two-window smoke source with pinned Trajectory Remix.
Development-only dependencies: the pinned upstream checkout and tokenizers 0.23.1.
The output directory must be new. No weights or user tokenizer are copied.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REVISION = 'd2c8083536d56a33d876e2650ae855722d491566'
ROOT = Path(__file__).resolve().parent.parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--upstream', required=True, type=Path)
p.add_argument('--tokenizer', required=True, type=Path)
p.add_argument('--output-dir', required=True, type=Path)
a = p.parse_args()
upstream = a.upstream.resolve()
if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=upstream, text=True).strip() != REVISION:
    p.error('upstream must be at the pinned compiler revision '+REVISION)
sys.path.insert(0, str(upstream))
from tokenizers import Tokenizer
from trajectory_remix.compiler import Compiler, chunks, visible_to
source_bytes = (ROOT/'fixtures/pretrained-backward.source.json').read_bytes()
compiler = Compiler(Tokenizer.from_file(str(a.tokenizer)), bos='<|kidra_bos|>', eos='<|kidra_eos|>', noop='<|kidra_noop|>', max_window_slots=24)
plan = compiler.compile(json.loads(source_bytes))
# Inspect the compiled plan rather than assuming the authored event order proves causality.
targets = [r for r in plan['rows'] if r['loss_weight'] > 0]
assert len(targets) == 1 and targets[0]['window'] == 'answer'
context = next(r for r in plan['rows'] if r['phase'] == 'B' and r['window'] == 'context')
assert context['loss_weight'] == 0
assert visible_to(targets[0], context, plan['windows'])
assert not visible_to(targets[0], targets[0], plan['windows'])
assert all(w['close'] == 2 for w in plan['windows'])

identity = dict(sha256=compiler.fingerprint, vocab_size=compiler.tokenizer.get_vocab_size(), special_tokens=compiler.ids)
data = ''.join(json.dumps(c, sort_keys=True)+'\n' for c in chunks(plan, 1)).encode()
manifest = dict(format='tesslot-backward-fixture/1', upstream='https://github.com/TheDudeFromCI/trajectory-remix', revision=REVISION,
                generator='scripts/generate-backward-fixture.py', source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                export_sha256=hashlib.sha256(data).hexdigest(), tokenizer=identity, window_capacity=24, active_windows=2,
                purpose='Curated synthetic cross-window copying gradient/memory smoke; not behavioral training quality')
a.output_dir.mkdir(parents=True, exist_ok=False)
(a.output_dir/'trajectory.jsonl').write_bytes(data)
(a.output_dir/'tokenizer-identity.json').write_text(json.dumps(identity, indent=2)+'\n')
(a.output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(manifest, indent=2))
