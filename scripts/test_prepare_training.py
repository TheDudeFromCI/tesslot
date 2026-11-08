"""Unit checks for hardware evidence parsing, not simulated CUDA certification."""
import importlib.util
from pathlib import Path
import unittest
import tempfile
spec = importlib.util.spec_from_file_location('prepare', Path(__file__).with_name('prepare-training-run.py'))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class Budgets(unittest.TestCase):
    def test_reserve_and_existing_usage_are_not_double_subtracted(self):
        rows = '0, GPU-a, RTX 3060, 0000:01:00.0, 999.1, 12288, 512, 11776\n1, GPU-b, RTX 5060, 0000:02:00.0, 999.1, 8192, 2048, 6144\n'
        gpus = prepare.parse_gpus(rows, 1024)
        self.assertEqual([g['observed_training_budget_mib'] for g in gpus], [11264, 6144])
        self.assertEqual([g['uuid'] for g in gpus], ['GPU-a', 'GPU-b'])

    def test_empty_invalid_and_exhausted_devices(self):
        self.assertEqual(prepare.parse_gpus('', 1024), [])
        for row in ['missing, columns', '0,a,name,bus,driver,N/A,0,0', '0,a,name,bus,driver,8,0,9']:
            with self.assertRaises(ValueError):
                prepare.parse_gpus(row, 1024)
        gpus = prepare.parse_gpus('0,a,name,bus,driver,512,0,512', 1024)
        self.assertEqual(gpus[0]['observed_training_budget_mib'], 0)


class ModelSources(unittest.TestCase):
    def test_standalone_file_with_spaces_does_not_require_a_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            weights = Path(folder)/'qwen base.safetensors'
            weights.touch()
            cmd, kind, scope = prepare.source_check(weights)
            self.assertEqual(cmd, ['inspect-weights', weights])
            self.assertEqual(kind, 'standalone_file')
            self.assertEqual(scope, 'safetensors_metadata_only')
            (Path(folder)/'config.json').write_text('{}')
            cmd, _, scope = prepare.source_check(weights)
            self.assertEqual(cmd[-2:], ['--model-config', Path(folder)/'config.json'])
            self.assertEqual(scope, 'config_and_tensor_shapes')

    def test_explicit_config_and_directory_routing(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            weights = folder/'weights.safetensors'
            config = folder/'actual-config.json'
            weights.touch()
            config.touch()
            cmd, _, scope = prepare.source_check(weights, config)
            self.assertEqual(cmd[-2:], ['--model-config', config])
            self.assertEqual(scope, 'config_and_tensor_shapes')
            self.assertEqual(prepare.source_check(folder)[0], ['inspect-model-directory', folder])
            for source, override in [(folder, config), (weights, folder/'missing.json'), (folder/'missing.safetensors', None)]:
                with self.assertRaises(ValueError):
                    prepare.source_check(source, override)


if __name__ == '__main__':
    unittest.main()
