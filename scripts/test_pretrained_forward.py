"""Plan construction and failure-reporting checks; no CUDA execution is simulated as passing."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).with_name('pretrained-forward-smoke.py')
spec = importlib.util.spec_from_file_location('forward_smoke', SCRIPT)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class Forward(unittest.TestCase):
    def test_plan_uses_visible_zero_and_actual_config_layer_count(self):
        c = dict(num_hidden_layers=28, bos_token_id=151643, vocab_size=151936)
        p = smoke.runtime_plan(c, 4294967296)
        self.assertEqual(len(p['layers']), 28)
        self.assertEqual(p['root'], dict(kind='cuda', index=0))
        self.assertTrue(all(t == p['root'] for t in p['layers']))
        self.assertEqual(p['precision'], 'tf32')
        with self.assertRaises(ValueError):
            smoke.runtime_plan(dict(c, num_hidden_layers=0), 4294967296)

    def test_missing_gpu_writes_failure_report_without_building_or_forward(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            model = root/'model'
            model.mkdir()
            (model/'config.json').write_text(json.dumps(dict(num_hidden_layers=2, bos_token_id=1, vocab_size=128)))
            out = root/'evidence'
            args = [str(SCRIPT), '--model-dir', str(model), '--output-dir', str(out), '--gpu-uuid', 'GPU-test']
            for backward in [False, True]:
                destination = out/('backward' if backward else 'forward')
                command_args = [str(SCRIPT), '--model-dir', str(model), '--output-dir', str(destination), '--gpu-uuid', 'GPU-test']
                if backward:
                    command_args += ['--backward-export', str(root/'trajectory.jsonl')]
                with mock.patch.object(sys, 'argv', command_args), mock.patch.object(subprocess, 'run', side_effect=FileNotFoundError('nvidia-smi unavailable')) as command:
                    self.assertEqual(smoke.main(), 1)
                    self.assertEqual(command.call_count, 1)
                mode = 'backward' if backward else 'forward'
                r = json.loads((destination/(mode+'-report.json')).read_text())
                self.assertEqual(r['status'], 'failed')
                self.assertFalse(r['training'])
                self.assertEqual(r['backward_requested'], backward)
                self.assertNotIn(mode, r)


if __name__ == '__main__':
    unittest.main()
