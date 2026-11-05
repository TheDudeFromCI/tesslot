#!/usr/bin/env python3
"""Development-only generator; requires an unmodified pinned Remix checkout.

Run: python scripts/generate-fixtures.py /path/to/trajectory-remix
Install tokenizers, jsonschema, simpleeval and cloudpickle in a disposable venv.
This script is never used by Tesslot at runtime or by ordinary CI.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

REVISION = "d2c8083536d56a33d876e2650ae855722d491566"
upstream = pathlib.Path(sys.argv[1]).resolve()
assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip() == REVISION
sys.path.insert(0, str(upstream))
from tokenizers import Tokenizer
from trajectory_remix.compiler import Compiler, chunks, visible_to
from trajectory_remix.tools import default_registry

root = pathlib.Path(__file__).resolve().parents[1] / "fixtures"
root.mkdir(exist_ok=True)
tokenizer = Tokenizer.from_file(str(upstream / "examples/demo-tokenizer.json"))
compiler = Compiler(tokenizer, max_window_slots=7)
source = {
    "schema-version": "1.0", "id": "canonical-multi-window",
    "uuid": "12345678-1234-5678-1234-567812345678", "tick-policy": "external-first",
    "defaults": {"implicit-noop-loss-weight": 0.1},
    "replay": {
        "0": [{"op": "open", "window": "a", "window-type": "reasoning"},
              {"op": "open", "window": "b"},
              {"op": "append", "source": "user", "window": "a", "content": "hello"}],
        "1": [{"op": "generate", "window": "a", "segments": [{"text": "x", "loss": {"weight": 0.25}}, {"text": "y", "loss": {"mask": "exclude"}}]},
              {"op": "generate", "window": "b", "content": "ab"}],
        "3": [{"op": "append", "source": "tool", "window": "a", "content": "ok", "replace-content": True,
               "loss": {"mask": "include", "weight": 0.5}}],
        "4": [{"op": "generate", "window": "b", "content": "<|eos|>"}],
        "5": [{"op": "open", "window": "late"},
              {"op": "append", "source": "user", "window": "late", "content": "z"}],
        "6": [{"op": "tool-call", "tool": "calculator", "window": "calc", "arguments": {"expression": "1+2"}}],
        "45": [{"op": "generate", "window": "a", "content": "<|noop|>"}],
        "46": [{"op": "generate", "window": "a", "content": "<|eos|>"}],
    },
}
plan = compiler.compile(source)
def write_json(name, data):
    (root / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
write_json("canonical.source.json", source)
(root / "demo-tokenizer.json").write_text((upstream / "examples/demo-tokenizer.json").read_text())
for size, name in [(3, "canonical.jsonl"), (1000, "canonical-single.jsonl")]:
    (root / name).write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in chunks(plan, size)))
def mask_hex(query):
    bits = bytearray((len(plan["rows"]) + 7) // 8)
    for j, key in enumerate(plan["rows"]):
        if visible_to(query, key, plan["windows"]):
            bits[j // 8] |= 1 << (j % 8)
    return bits.hex()
write_json("visibility.json", {"row_count": len(plan["rows"]), "allowed_lsb0_hex": [mask_hex(q) for q in plan["rows"]]})
write_json("tokenizer-identity.json", {"sha256": compiler.fingerprint, "vocab_size": tokenizer.get_vocab_size(), "special_tokens": compiler.ids})
# Complete author schemas are retained for review, never registered as production tools.
write_json("tool-schemas.json", default_registry().schemas(invisible=True))
long_source = {
    "schema-version": "1.0", "id": "long-idle",
    "uuid": "abcdefab-1234-5678-1234-567812345678", "tick-policy": "external-first",
    "defaults": {"window-headers": False, "implicit-noop-loss-weight": 0.1},
    "replay": {
        "0": [{"op": "open", "window": "a"}, {"op": "open", "window": "b"}],
        "1": [{"op": "generate", "window": "a", "content": "x"}],
        "2048": [{"op": "append", "window": "b", "source": "tool", "content": "ok"}],
        "2049": [{"op": "generate", "window": "a", "content": "<|eos|>"},
                 {"op": "generate", "window": "b", "content": "<|eos|>"}],
    },
}
write_json("long.source.json", long_source)
long_plan = compiler.compile(long_source)
(root / "long.jsonl").write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in chunks(long_plan, 64)))
tiny_source = {
    "schema-version": "1.0", "id": "autograd-tiny",
    "uuid": "11223344-1234-5678-1234-567812345678", "tick-policy": "external-first",
    "defaults": {"window-headers": False, "implicit-noop-loss-weight": 0.1},
    "replay": {
        "0": [{"op": "open", "window": "a"}, {"op": "open", "window": "b"},
              {"op": "append", "window": "a", "source": "user", "content": "x"}],
        "1": [{"op": "generate", "window": "a", "content": "y"}, {"op": "generate", "window": "b", "content": "z"}],
        "2": [{"op": "append", "window": "a", "source": "tool", "content": "q", "replace-content": True, "loss": {"mask": "include", "weight": 0.25}}],
        "3": [{"op": "generate", "window": "b", "content": "<|eos|>"}],
        "4": [{"op": "generate", "window": "a", "content": "<|noop|>"}],
        "5": [{"op": "generate", "window": "a", "content": "<|eos|>"}],
    },
}
write_json("autograd.source.json", tiny_source)
tiny_plan = compiler.compile(tiny_source)
(root / "autograd.jsonl").write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in chunks(tiny_plan, 2)))
files = [p for p in root.iterdir() if p.name != "manifest.json"]
write_json("manifest.json", {"upstream": "https://github.com/TheDudeFromCI/trajectory-remix", "revision": REVISION,
    "generator": "scripts/generate-fixtures.py", "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}})
print(f"Generated {len(plan['rows'])} rows using real compiler/exporter at {REVISION}")
