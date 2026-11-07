#!/usr/bin/env python3
"""Dev-only: real Remix export with the Rust converter's exact extended tokenizer.
Usage: python scripts/generate-pretrained-fixture.py UPSTREAM_CHECKOUT CONVERSION_DIR
Generate CONVERSION_DIR using convert-pretrained on fixtures/tiny-qwen plus the
base fixtures/demo-tokenizer.json copied as tokenizer.json. Never modifies upstream.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

revision = "d2c8083536d56a33d876e2650ae855722d491566"
upstream = pathlib.Path(sys.argv[1]).resolve()
converted = pathlib.Path(sys.argv[2]).resolve()
assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip() == revision
sys.path.insert(0, str(upstream))
from tokenizers import Tokenizer
from trajectory_remix.compiler import Compiler, chunks

root = pathlib.Path(__file__).resolve().parents[1] / "fixtures"
tokenizer = Tokenizer.from_file(str(converted / "tokenizer.json"))
compiler = Compiler(tokenizer, bos="<|kidra_bos|>", eos="<|kidra_eos|>", noop="<|kidra_noop|>", max_window_slots=7)
source = json.loads((root / "autograd.source.json").read_text().replace("<|eos|>", "<|kidra_eos|>").replace("<|noop|>", "<|kidra_noop|>"))
source["id"] = "pretrained-autograd-tiny"
source["uuid"] = "33445566-1234-5678-1234-567812345678"
plan = compiler.compile(source)
(root / "pretrained-autograd.jsonl").write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in chunks(plan, 2)))
identity = {"sha256": compiler.fingerprint, "vocab_size": tokenizer.get_vocab_size(), "special_tokens": compiler.ids}
assert identity == json.loads((converted / "conversion.json").read_text())["tokenizer_conversion"]["target"]
manifest = {"upstream": "https://github.com/TheDudeFromCI/trajectory-remix", "revision": revision, "generator": "scripts/generate-pretrained-fixture.py", "tokenizer": identity, "sha256": hashlib.sha256((root / "pretrained-autograd.jsonl").read_bytes()).hexdigest()}
(root / "pretrained-fixture.json").write_text(json.dumps(manifest, indent=2) + "\n")
