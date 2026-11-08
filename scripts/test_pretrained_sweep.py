"""Sweep integrity/status/archive and process supervision tests; CUDA is never qualified here."""
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

from pretrained_common import ProbeRunner

SCRIPT = Path(__file__).with_name('pretrained-readiness-sweep.py')
spec = importlib.util.spec_from_file_location('readiness_sweep', SCRIPT)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


class Sweep(unittest.TestCase):
    def test_real_child_is_terminated_on_reserve_breach_and_next_stage_can_run(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            runner = ProbeRunner(directory, directory, 'GPU-test', 1024, 5)
            with mock.patch.object(runner, 'observe', side_effect=[dict(free_mib=5000, used_mib=100), dict(free_mib=900, used_mib=11000)]):
                first = runner.run('limited', [sys.executable, '-c', 'import time; time.sleep(30)'])
            self.assertEqual(first['status'], 'reserve_exceeded')
            self.assertIsNotNone(first['process_exit_code'])
            self.assertEqual(first['sampled_min_free_mib'], 900)
            with mock.patch.object(runner, 'observe', return_value=dict(free_mib=5000, used_mib=100)):
                second = runner.run('next', [sys.executable, '-c', 'print("completed CPU child")'])
            self.assertEqual(second['status'], 'passed')
            self.assertIn('completed CPU child', (directory/'next.stdout.log').read_text())

    def test_timeout_and_allocator_failure_are_not_passing_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            runner = ProbeRunner(directory, directory, None, 1024, 0)
            result = runner.run('timeout', [sys.executable, '-c', 'import time; time.sleep(30)'])
            self.assertEqual(result['status'], 'timed_out')
            runner.timeout_seconds = 5
            result = runner.run('oom', [sys.executable, '-c', 'import sys; sys.stderr.write("CUDA_ERROR_OUT_OF_MEMORY"); sys.exit(1)'])
            self.assertEqual(result['status'], 'out_of_memory')
            result = runner.run('bad', [sys.executable, '-c', 'import sys; sys.stderr.write("nonfinite gradient"); sys.exit(1)'])
            self.assertEqual(result['status'], 'failed')

    def test_adamw_uses_transactional_state_and_reserve_not_pooled_memory(self):
        record = sweep.adamw_admission(600299520*4, 11800, 1024)
        self.assertEqual(record['transactional_step_bytes'], 600299520*28)
        self.assertGreater(record['transactional_step_bytes'], record['available_after_reserve_bytes'])
        self.assertEqual(sweep.adamw_admission(16, 500, 1024)['available_after_reserve_bytes'], 0)

    def test_limits_and_unexpected_failures_have_different_exit_codes(self):
        report = dict(checks=dict(pilot_restart_identical=True), capacity={}, pilot={}, adamw=dict(status='passed'))
        self.assertEqual(sweep.finish_status(report), ('passed', 0))
        report['capacity']['large'] = dict(status='reserve_exceeded')
        self.assertEqual(sweep.finish_status(report), ('completed_with_limits', 2))
        report['pilot']['run'] = dict(status='failed')
        self.assertEqual(sweep.finish_status(report), ('failed', 1))

    def test_archive_excludes_all_weight_payloads_and_retains_manifests_and_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            output = directory/'output'
            (output/'checkpoint').mkdir(parents=True)
            (output/'checkpoint/model.safetensors').write_bytes(b'private weights')
            (output/'checkpoint/optimizer.safetensors').write_bytes(b'private moments')
            (output/'checkpoint/manifest.json').write_text('{}')
            (output/'stage.memory.jsonl').write_text('{}\n')
            (output/'stage.stdout.log').write_text('log')
            archive = directory/'logs.tar.gz'
            sweep.archive_logs(output, archive)
            with tarfile.open(archive) as data:
                names = data.getnames()
            self.assertFalse(any(n.endswith('.safetensors') for n in names))
            self.assertIn('output/checkpoint/manifest.json', names)
            self.assertIn('output/stage.memory.jsonl', names)

    def test_compiled_fixtures_match_their_manifest_and_tokenizer(self):
        for name in ['pretrained-sweep-24', 'pretrained-sweep-tiny']:
            directory = sweep.ROOT/'fixtures'/name
            manifest = json.loads((directory/'manifest.json').read_text())
            conversion = dict(windows=24, tokenizer_conversion=dict(target=manifest['tokenizer']))
            self.assertEqual(len(sweep.verify_fixtures(directory, conversion)['capacity']), 10)
            conversion['windows'] = 4
            with self.assertRaises(ValueError):
                sweep.verify_fixtures(directory, conversion)


if __name__ == '__main__':
    unittest.main()
