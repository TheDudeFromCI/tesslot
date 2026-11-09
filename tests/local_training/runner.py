"""Opt-in unittest adapter and temporary artifact lifecycle for local model regressions."""
import argparse
import copy
import json
import math
from pathlib import Path
import tempfile
import traceback
import unittest
import uuid

from . import engine


class LocalTrainingTests(unittest.TestCase):
    """Normal discovery skips these tests; the explicit runner supplies a shared session."""
    session = None

    def setUp(self):
        if self.session is None:
            self.skipTest('opt in with python scripts/test-pretrained.py')
        self.report = self.session.report()
        if self.report.get('profile_status') in {'unsupported','insufficient_budget'}:
            if self.report['profile_status']=='insufficient_budget' and getattr(self.session.config,'strict_capacity',False):
                self.fail(self.report['profile_reason'])
            self.skipTest(self.report['profile_reason'])

    def test_setup_and_source_integrity(self):
        self.assertNotIn('error', self.report, self.report.get('error'))
        for name in ['source_files_unchanged', 'fixtures_unchanged']:
            self.assertIs(self.report['checks'].get(name), True, name)

    def test_capacity(self):
        self.assertTrue(self.report['capacity'], 'capacity stages did not run')
        for name, stage in self.report['capacity'].items():
            with self.subTest(case=name):
                if stage['status'] == 'unsupported_by_runtime' or (name == 'full-context-admission' and stage['status'] == 'not_run'):
                    self.skipTest(name+': '+stage.get('reason', stage.get('error', stage['status'])))
                if stage['status'] in {'reserve_exceeded','out_of_memory'} and not getattr(self.session.config,'capacity_case',None) and not getattr(self.session.config,'strict_capacity',False):
                    self.skipTest(name+': discovered resource limit; use --capacity-case to require a known-fit regression workload')
                self.assertEqual(stage['status'], 'passed', stage.get('error', name))

    def test_forward_backward_and_single_update(self):
        self.assertTrue(self.report['diagnostics'], 'diagnostics did not run')
        for name, stage in self.report['diagnostics'].items():
            with self.subTest(case=name):
                if stage['status'] == 'not_supplied':
                    self.skipTest(stage['reason'])
                self.assertEqual(stage['status'], 'passed', stage.get('error', name))

    def test_pilot_held_out_and_restart(self):
        self.assertTrue(self.report['pilot'], 'pilot did not run')
        for name, stage in self.report['pilot'].items():
            with self.subTest(case=name):
                self.assertEqual(stage['status'], 'passed', stage.get('error', name))
        self.assertIs(self.report['checks'].get('pilot_restart_identical'), True)

    def test_adamw(self):
        stage = self.report['adamw']
        if stage['status'] in {'insufficient_budget', 'unsupported_by_runtime'}:
            self.skipTest(stage.get('reason', stage.get('error', stage['status'])))
        self.assertEqual(stage['status'], 'passed', stage.get('error'))


class Session:
    def __init__(self, config, directory):
        self.config, self.directory, self._report = config, directory, None

    def report(self):
        if self._report is None:
            try:
                self._report = engine.collect(self.config, self.directory)
            except Exception as error:
                (self.directory/'engine-error.log').write_text(traceback.format_exc())
                self._report = dict(status='failed', error=str(error), capacity={}, diagnostics={},
                                    checks={}, pilot={}, adamw=dict(status='not_run'), capacity_passed=[])
        return self._report


def run(config, stream=None):
    """Always remove owned scratch data; export one diagnostics archive only on failure."""
    with tempfile.TemporaryDirectory(prefix='tesslot-pretrained-', dir=config.tmp_dir) as temporary:
        directory = Path(temporary)
        requested = getattr(config, 'runtime_plan', None)
        plans = requested if isinstance(requested, list) else [requested]
        profiles, successful, cached_binary = {}, True, getattr(config, 'tesslot_bin', None)
        for number, plan_path in enumerate(plans or [None], 1):
            name = 'profile-'+str(number).zfill(2)
            scratch = directory/name
            scratch.mkdir()
            profile_config = copy.copy(config)
            profile_config.runtime_plan = plan_path
            if cached_binary:
                profile_config.tesslot_bin = Path(cached_binary)
            session = Session(profile_config, scratch)
            suite_class = type('ConfiguredProfile'+str(number), (LocalTrainingTests,), {'session': session})
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(suite_class)
            result = unittest.TextTestRunner(verbosity=2, stream=stream).run(suite)
            report = session.report()
            report['test_status'] = 'passed' if result.wasSuccessful() else 'failed'
            report['runtime_plan_source'] = str(plan_path) if plan_path else None
            (scratch/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            profiles[name] = report
            successful = successful and result.wasSuccessful()
            cached_binary = report.get('binary_path', cached_binary)
            # Finished native processes no longer need checkpoints. Reclaim their space
            # before another profile; diagnostics never include these payloads.
            for payload in scratch.rglob('*.safetensors'):
                payload.unlink()
        if len(profiles) == 1:
            report = next(iter(profiles.values()))
        else:
            report = dict(format='tesslot-pretrained-test-matrix/1', profiles=profiles,
                          status='passed' if successful else 'failed',
                          test_status='passed' if successful else 'failed',
                          capacity_passed=[name+'/'+case for name, value in profiles.items() for case in value['capacity_passed']],
                          validated_profiles=[name for name, value in profiles.items() if value['test_status']=='passed' and value.get('profile_status') not in {'unsupported','insufficient_budget'}],
                          unavailable_profiles={name:value['profile_reason'] for name,value in profiles.items() if value.get('profile_status') in {'unsupported','insufficient_budget'}})
        (directory/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        archive = None
        if not successful:
            config.failure_dir.mkdir(parents=True, exist_ok=True)
            archive = config.failure_dir/('pretrained-'+uuid.uuid4().hex[:12]+'.tar.gz')
            engine.archive_logs(directory, archive)
        return report, archive, successful, str(directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run opt-in pretrained regression tests; scratch lives in the OS temporary directory and only failures export logs.')
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--base-model-dir', type=Path, help='Optional original HF directory for source inspection and base forward')
    device = parser.add_mutually_exclusive_group(required=True)
    device.add_argument('--gpu-uuid', action='append', help='Physical GPU UUID; repeat in visible ordinal order for multiple GPUs')
    device.add_argument('--cpu', action='store_true', help='Explicit CPU FP32 execution; never qualifies CUDA')
    parser.add_argument('--runtime-plan', type=Path, action='append', help='Explicit root/layer owners and precision; repeat to test several profiles in one invocation')
    parser.add_argument('--precision', choices=['f32','tf32','bf16','f16'], help='Default f32; explicit runtime plans define their own precision')
    parser.add_argument('--fixtures-dir', type=Path, default=engine.ROOT/'fixtures/pretrained-sweep-24')
    parser.add_argument('--tesslot-bin', type=Path)
    parser.add_argument('--capacity-case', action='append')
    parser.add_argument('--strict-capacity', action='store_true', help='Require selected workloads and known state budgets to fit; use for a qualified regression baseline')
    parser.add_argument('--pilot-steps', type=int, default=12)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--reserve-mib', type=int, default=1024)
    parser.add_argument('--host-reserve-mib', type=int, default=1024)
    parser.add_argument('--max-bytes', type=int, default=4294967296)
    parser.add_argument('--timeout-seconds', type=int, default=7200)
    parser.add_argument('--tmp-dir', type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument('--failure-dir', type=Path, default=Path('test-failures'))
    parser.add_argument('--json', action='store_true', help='Print full machine-readable results to stdout; does not save success logs')
    args = parser.parse_args(argv)
    if not 4 <= args.pilot_steps <= 64 or args.reserve_mib < 0 or args.host_reserve_mib < 0 or args.max_bytes <= 0 or args.timeout_seconds < 1:
        parser.error('invalid pilot steps, reserve, byte budget or timeout')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('learning rate must be positive and finite')
    if len(set(args.gpu_uuid or [])) != len(args.gpu_uuid or []):
        parser.error('physical GPU UUIDs must be unique')
    args.tmp_dir = args.tmp_dir.expanduser().resolve()
    args.failure_dir = args.failure_dir.expanduser().resolve()
    if not args.tmp_dir.is_dir():
        parser.error('temporary parent directory must exist')
    # A failure archive must outlive this session's temporary directory.
    report, archive, successful, temporary = run(args)
    if args.json:
        print(json.dumps(dict(report=report, archive=str(archive) if archive else None, temporary_directory=temporary, successful=successful)))
    else:
        print('Tests: '+report['test_status']+'; collection: '+report['status']+'; '+str(len(report['capacity_passed']))+' capacity cases passed.')
        if report.get('profile_reason'):
            print('Unsupported profile: '+report['profile_reason'])
        if report.get('held_out'):
            print('Held-out loss: {loss_before:.6f} -> {loss_after:.6f}'.format(**report['held_out']))
        print('Failure diagnostics: '+str(archive) if archive else 'Temporary logs and checkpoints removed; no diagnostics exported.')
    return 0 if successful else 1
