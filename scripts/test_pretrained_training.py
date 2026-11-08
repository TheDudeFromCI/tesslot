"""Checkpoint-contract and reserve-abort tests; no CUDA success is simulated."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).with_name('pretrained-training-smoke.py')
spec = importlib.util.spec_from_file_location('training_smoke', SCRIPT)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class Training(unittest.TestCase):
    def test_checkpoint_rejects_corruption_cursor_and_optimizer_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            weights = b'checkpoint-payload'
            (directory/'model.safetensors').write_bytes(weights)
            manifest = dict(format='tiny-qwen-remix-sgd/1', optimizer='sgd-no-momentum/1', step=4,
                            learning_rate=0.0001, weights_sha256=hashlib.sha256(weights).hexdigest(),
                            remix=dict(completed_samples=4, dataset_sha256='dataset'))
            def write(value):
                (directory/'manifest.json').write_text(json.dumps(value))
            write(manifest)
            self.assertEqual(smoke.checkpoint(directory, 4, 'dataset', 0.0001)[0], manifest)
            for key, value in [('step', 3), ('optimizer', 'adamw-f32/1'), ('learning_rate', 0.001),
                               ('weights_sha256', 'wrong'), ('remix', dict(completed_samples=3, dataset_sha256='dataset'))]:
                write(dict(manifest, **{key: value}))
                with self.assertRaises(ValueError):
                    smoke.checkpoint(directory, 4, 'dataset', 0.0001)
            write(manifest)
            (directory/'model.safetensors').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                smoke.checkpoint(directory, 4, 'dataset', 0.0001)

    def test_reserve_abort_terminates_child_and_retains_failure_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            model = directory/'model'
            model.mkdir()
            for name in smoke.SOURCE_FILES:
                (model/name).write_text('{}')
            (model/'config.json').write_text(json.dumps(dict(num_hidden_layers=2, bos_token_id=1, vocab_size=128)))
            export = directory/'trajectory.jsonl'
            export.write_bytes((smoke.ROOT/'fixtures/pretrained-backward-tiny/trajectory.jsonl').read_bytes())
            out = directory/'evidence'
            args = [str(SCRIPT), '--model-dir', str(model), '--export', str(export),
                    '--output-dir', str(out), '--gpu-uuid', 'GPU-test']
            process = mock.Mock(returncode=-15)
            process.poll.return_value = None
            readings = [subprocess.CompletedProcess([], 0, 'GPU-test, Test GPU, 12288, 100, 11800\n'),
                        subprocess.CompletedProcess([], 0, 'GPU-test, Test GPU, 12288, 11000, 900\n')]
            with mock.patch.object(sys, 'argv', args), mock.patch.object(subprocess, 'run', side_effect=readings), \
                    mock.patch.object(subprocess, 'Popen', return_value=process):
                self.assertEqual(smoke.main(), 1)
            process.terminate.assert_called_once()
            process.wait.assert_called_once()
            report = json.loads((out/'training-report.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertIn('reserve', report['error'])
            self.assertEqual(report['sampled_min_free_mib'], 900)
            self.assertEqual(report['stages']['build']['status'], 'failed')
            self.assertEqual(report['stages']['build']['process_exit_code'], -15)
            self.assertNotIn('continuous', report['stages'])
            self.assertEqual(report['checks']['source_files_unchanged'], 'passed')

    def test_only_one_bounded_single_target_trajectory_is_accepted(self):
        fixture = smoke.ROOT/'fixtures/pretrained-backward-tiny/trajectory.jsonl'
        self.assertEqual(smoke.workload(fixture), dict(rows=8, chunks=3, window_capacity=24, active_windows=2))
        with tempfile.TemporaryDirectory() as directory:
            export = Path(directory)/'export.jsonl'
            export.write_bytes(fixture.read_bytes()*2)
            with self.assertRaises(ValueError):
                smoke.workload(export)
            chunks = [json.loads(line) for line in fixture.read_text().splitlines()]
            chunks[0]['rows'][0]['loss_weight'] = 1
            export.write_text('\n'.join(json.dumps(c) for c in chunks))
            with self.assertRaises(ValueError):
                smoke.workload(export)

    def test_missing_update_nonfinite_metrics_or_wrong_step_cannot_pass(self):
        events = [dict(event='optimizer_updated', metrics=dict(step=3, loss=5.0, gradient_norm=2.0)),
                  dict(event='optimizer_updated', metrics=dict(step=4, loss=4.5, gradient_norm=1.0)),
                  dict(event='checkpoint_saved', step=4, state=dict(completed_samples=4),
                       optimizer='sgd-no-momentum/1', single_target=True, dataset_samples=1, final_single_target_loss=4.0)]
        self.assertEqual(smoke.validate_events(events, 2, 2)['step'], 4)
        with self.assertRaises(ValueError):
            smoke.validate_events(events[1:], 2, 2)
        events[0]['metrics']['loss'] = float('nan')
        with self.assertRaises(ValueError):
            smoke.validate_events(events, 2, 2)
        events[0]['metrics']['loss'] = 5.0
        events[-1]['state']['completed_samples'] = 3
        with self.assertRaises(ValueError):
            smoke.validate_events(events, 2, 2)


if __name__ == '__main__':
    unittest.main()
