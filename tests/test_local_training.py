"""Sweep integrity/status/archive and process supervision tests; CUDA is never qualified here."""
import json
import copy
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
from tests.local_training.process import ProbeRunner
from tests.local_training import engine as sweep
from tests.local_training import runner as suite_runner
from tests.local_training.process import runtime_plan
from types import SimpleNamespace
from tests.local_training.placement import make_plan, parameter_layout, layout_admission, cuda_model_gates, visible_plan


class Sweep(unittest.TestCase):
    def test_strict_capacity_turns_a_known_budget_shortfall_into_a_failure(self):
        for strict in [False,True]:
            with self.subTest(strict=strict), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                config = SimpleNamespace(tmp_dir=directory, failure_dir=directory/'failures', strict_capacity=strict)
                report = dict(status='completed_with_limits',profile_status='insufficient_budget',profile_reason='known state exceeds owner budget',capacity_passed=[])
                with mock.patch.object(sweep,'collect',return_value=report):
                    _, archive, successful, temporary = suite_runner.run(config,stream=io.StringIO())
                self.assertEqual(successful, not strict)
                self.assertEqual(archive is not None, strict)
                self.assertFalse(Path(temporary).exists())

    def test_unused_gpu_is_hidden_without_changing_physical_layer_ownership(self):
        requested = dict(root=dict(kind='cuda',index=2),layers=[dict(kind='cuda',index=0),dict(kind='cuda',index=2)])
        plan, uuids, mapping = visible_plan(requested,['GPU-first','GPU-unused','GPU-third'])
        self.assertEqual(uuids,['GPU-first','GPU-third'])
        self.assertEqual(mapping,{0:0,2:1})
        self.assertEqual(plan['root']['index'],1)
        self.assertEqual(requested['root']['index'],2)

    def test_precision_gates_follow_profile_instead_of_assuming_hardware_features(self):
        self.assertEqual(cuda_model_gates('f32',1),['qwen_cuda_f32'])
        self.assertEqual(cuda_model_gates('bf16',2),['qwen_cuda_bf16','qwen_dual_cuda_bf16'])
        config = dict(num_hidden_layers=2, bos_token_id=1, vocab_size=10)
        args = SimpleNamespace(runtime_plan=None,cpu=False,gpu_uuid=['GPU-any'],max_bytes=1024)
        plan, _ = make_plan(config,args)
        self.assertEqual(plan['precision'],'f32')

    def test_multiple_profiles_collect_once_each_and_reclaim_payloads_between_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            config = SimpleNamespace(tmp_dir=directory, failure_dir=directory/'failures', runtime_plan=[Path('one.json'),Path('two.json')])
            seen = []
            def collect(config, output):
                if seen:
                    self.assertFalse(any(p.suffix=='.safetensors' for p in directory.rglob('*')))
                seen.append(config.runtime_plan)
                (output/'model.safetensors').write_bytes(b'payload')
                return dict(status='passed',profile_status='supported',capacity_passed=['small'],capacity=dict(small=dict(status='passed')),
                            diagnostics=dict(backward=dict(status='passed')),pilot=dict(run=dict(status='passed')),
                            checks=dict(source_files_unchanged=True,fixtures_unchanged=True,pilot_restart_identical=True),adamw=dict(status='passed'))
            with mock.patch.object(sweep,'collect',side_effect=collect):
                report, archive, successful, temporary = suite_runner.run(config,stream=io.StringIO())
            self.assertTrue(successful)
            self.assertEqual(seen,[Path('one.json'),Path('two.json')])
            self.assertEqual(len(report['validated_profiles']),2)
            self.assertIsNone(archive)
            self.assertFalse(Path(temporary).exists())

    def test_portable_layer_placement_and_explicit_offload_capability(self):
        config = dict(num_hidden_layers=4, bos_token_id=1, vocab_size=10)
        args = SimpleNamespace(runtime_plan=None, cpu=False, gpu_uuid=['GPU-a','GPU-b'], max_bytes=1024)
        plan, reason = make_plan(config,args)
        self.assertIsNone(reason)
        self.assertEqual([t['index'] for t in plan['layers']], [0,0,1,1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'runtime.json'
            plan['layers'][0] = dict(kind='cpu')
            path.write_text(json.dumps(plan))
            args.runtime_plan = path
            _, reason = make_plan(config,args)
            self.assertIn('offload', reason)
            plan['layers'][0] = dict(kind='cuda',index=3)
            path.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                make_plan(config,args)

    def test_admission_never_pools_asymmetric_gpu_or_cpu_memory(self):
        mib = 1024*1024
        observation = dict(devices=[dict(uuid='GPU-small',free_mib=1500),dict(uuid='GPU-large',free_mib=60000)],host_available_mib=32000)
        admission = layout_admission({'cuda:0':400*mib,'cuda:1':mib},observation,['GPU-small','GPU-large'],1024,1024)
        self.assertFalse(admission['admitted'])
        self.assertEqual(admission['devices']['cuda:0']['available_after_reserve_bytes'],476*mib)
        self.assertGreater(admission['devices']['cuda:1']['available_after_reserve_bytes'], admission['devices']['cuda:1']['transactional_step_bytes'])
        self.assertTrue(layout_admission({'cpu':mib},observation,[],1024,1024)['admitted'])

    def test_parameter_layout_keeps_tied_root_and_local_layers_with_their_owners(self):
        import struct
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.safetensors'
            header = {'model.embed_tokens.weight':dict(shape=[16,4]), 'model.layers.0.q_proj.weight':dict(shape=[8,8]),'model.layers.1.q_proj.weight':dict(shape=[8,8])}
            data = json.dumps(header).encode()
            path.write_bytes(struct.pack('<Q',len(data))+data)
            plan = dict(root=dict(kind='cuda',index=0),layers=[dict(kind='cuda',index=0),dict(kind='cuda',index=1)])
            self.assertEqual(parameter_layout(path,plan),{'cuda:0':512,'cuda:1':256})

    def test_multi_device_reserve_checks_each_physical_gpu(self):
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            runner = ProbeRunner(Path(directory),Path(directory),['GPU-a','GPU-b'],1024,5)
            readings = [subprocess.CompletedProcess([],0,'GPU-a, Device A, 8192, 7000, 900\n'),subprocess.CompletedProcess([],0,'GPU-b, Device B, 65536, 100, 60000\n')]
            with mock.patch('tests.local_training.process.subprocess.run',side_effect=readings):
                value = runner.observe()
            self.assertEqual(value['free_mib'],900)
            self.assertEqual(len(value['devices']),2)
            self.assertEqual(runner.env['CUDA_VISIBLE_DEVICES'],'GPU-a,GPU-b')

    def test_runtime_plan_uses_actual_layer_count_and_selected_visible_device(self):
        config = dict(num_hidden_layers=28, bos_token_id=151643, vocab_size=151936)
        plan = runtime_plan(config, 4294967296)
        self.assertEqual(len(plan['layers']), 28)
        self.assertEqual(plan['root'], dict(kind='cuda', index=0))
        self.assertTrue(all(p == plan['root'] for p in plan['layers']))
        with self.assertRaises(ValueError):
            runtime_plan(dict(config, num_hidden_layers=0), 4294967296)

    def test_success_and_failure_cleanup_with_failure_only_export(self):
        baseline = dict(status='completed_with_limits', capacity_passed=['small'],
                        capacity=dict(small=dict(status='passed'), full=dict(status='unsupported_by_runtime', reason='row limit')),
                        checks=dict(source_files_unchanged=True, fixtures_unchanged=True, pilot_restart_identical=True),
                        diagnostics=dict(backward=dict(status='passed')), pilot=dict(run=dict(status='passed')),
                        adamw=dict(status='insufficient_budget', reason='known budget'))
        for fail, required in [(False,True),(True,True),(True,False)]:
            with self.subTest(failure=fail,required=required), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                scratch = directory/'scratch'
                scratch.mkdir()
                config = SimpleNamespace(tmp_dir=scratch, failure_dir=directory/'failures', capacity_case=['small'] if required else None)
                def collect(config, output):
                    (output/'model.safetensors').write_bytes(b'weights')
                    (output/'check.stderr.log').write_text('diagnostic')
                    report = copy.deepcopy(baseline)
                    if fail:
                        report['capacity']['small']['status'] = 'out_of_memory'
                        report['capacity']['small']['error'] = 'allocation failed'
                    return report
                with mock.patch.object(sweep, 'collect', side_effect=collect) as collector:
                    report, archive, successful, temporary = suite_runner.run(config, stream=io.StringIO())
                    collector.assert_called_once()
                self.assertEqual(successful, not fail or not required)
                self.assertFalse(Path(temporary).exists())
                self.assertEqual(list(scratch.iterdir()), [])
                if fail and required:
                    self.assertTrue(archive.is_file())
                    with tarfile.open(archive) as data:
                        self.assertFalse(any(n.endswith('.safetensors') for n in data.getnames()))
                        self.assertTrue(any(n.endswith('check.stderr.log') for n in data.getnames()))
                else:
                    self.assertIsNone(archive)
                    self.assertFalse(config.failure_dir.exists())

    def test_unexpected_collection_error_is_not_retried_and_exports_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            config = SimpleNamespace(tmp_dir=directory, failure_dir=directory/'failures')
            with mock.patch.object(sweep, 'collect', side_effect=RuntimeError('test orchestration failure')) as collector:
                report, archive, successful, temporary = suite_runner.run(config, stream=io.StringIO())
            collector.assert_called_once()
            self.assertFalse(successful)
            self.assertEqual(report['test_status'], 'failed')
            self.assertTrue(archive.is_file())
            self.assertFalse(Path(temporary).exists())

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
