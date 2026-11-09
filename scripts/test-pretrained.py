#!/usr/bin/env python3
"""Single entry point for opt-in local pretrained regression tests."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.local_training.runner import main

if __name__ == '__main__':
    raise SystemExit(main())
