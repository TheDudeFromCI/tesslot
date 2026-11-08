"""Unit checks for hardware evidence parsing, not simulated CUDA certification."""
import importlib.util
from pathlib import Path
import unittest
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


if __name__ == '__main__':
    unittest.main()
