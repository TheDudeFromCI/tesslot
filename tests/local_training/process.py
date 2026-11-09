"""Shared runtime-plan construction for local pretrained CUDA probes."""
import csv
import json
import os
import subprocess
import time


def runtime_plan(config, parameter_budget_bytes):
    layers = config['num_hidden_layers']
    token = config['bos_token_id']
    if not isinstance(layers, int) or not 1 <= layers <= 1024 or not isinstance(token, int) or not 0 <= token < config['vocab_size']:
        raise ValueError('invalid layer count or BOS ID')
    target = dict(kind='cuda', index=0)
    return dict(root=target, layers=[target.copy() for _ in range(layers)], precision='tf32', parameter_budget_bytes=parameter_budget_bytes)


class ReserveExceeded(RuntimeError):
    """A sampled GPU reserve breach, distinct from a Python host allocation failure."""


class ProbeRunner:
    """Sequential child-process isolation, per-stage memory logs and reserve enforcement."""
    def __init__(self, root, output, gpu_uuid, reserve_mib, timeout_seconds, host_reserve_mib=0):
        self.root, self.output = root, output
        self.gpu_uuids = [gpu_uuid] if isinstance(gpu_uuid, str) else list(gpu_uuid or [])
        self.reserve_mib, self.timeout_seconds = reserve_mib, timeout_seconds
        self.host_reserve_mib = host_reserve_mib
        self.env = dict(os.environ)
        if self.gpu_uuids:
            self.env.update(CUDA_VISIBLE_DEVICES=','.join(self.gpu_uuids), CUDA_DEVICE_ORDER='PCI_BUS_ID')

    def observe(self):
        row = dict(time_unix=time.time())
        devices = []
        for gpu_uuid in self.gpu_uuids:
            result = subprocess.run(['nvidia-smi', '--id='+gpu_uuid,
                                     '--query-gpu=uuid,name,memory.total,memory.used,memory.free',
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15, check=True)
            rows = list(csv.reader(result.stdout.splitlines()))
            if len(rows) != 1 or len(rows[0]) != 5 or rows[0][0].strip() != gpu_uuid:
                raise ValueError('GPU UUID must identify exactly one physical device')
            uuid, name, total, used, free = [s.strip() for s in rows[0]]
            devices.append(dict(uuid=uuid, name=name, total_mib=int(total), used_mib=int(used), free_mib=int(free)))
        if devices:
            row['devices'] = devices
            row['free_mib'] = min(d['free_mib'] for d in devices)
            row['used_mib'] = max(d['used_mib'] for d in devices)
        try:
            with open('/proc/meminfo') as stream:
                for line in stream:
                    if line.startswith('MemAvailable:'):
                        row['host_available_mib'] = int(line.split()[1])//1024
                        break
        except OSError:
            pass
        return row

    def run(self, name, command):
        record = dict(command=command, status='running', process_exit_code=None, phases=[])
        child = None
        start = time.monotonic()
        samples = []
        stdout_path, stderr_path = self.output/(name+'.stdout.log'), self.output/(name+'.stderr.log')
        try:
            with (self.output/(name+'.memory.jsonl')).open('w') as memory, stdout_path.open('w') as stdout, stderr_path.open('w') as stderr:
                def sample():
                    value = self.observe()
                    samples.append(value)
                    memory.write(json.dumps(value)+'\n')
                    memory.flush()
                    if self.gpu_uuids and value['free_mib'] < self.reserve_mib:
                        raise ReserveExceeded('sampled free VRAM fell below reserve')
                    if value.get('host_available_mib', self.host_reserve_mib) < self.host_reserve_mib:
                        raise ReserveExceeded('sampled available host memory fell below reserve')
                sample()
                child = subprocess.Popen(command, cwd=self.root, env=self.env, stdout=stdout, stderr=stderr)
                while True:
                    sample()
                    if child.poll() is not None:
                        break
                    if time.monotonic()-start > self.timeout_seconds:
                        raise TimeoutError('stage exceeded configured timeout')
                    time.sleep(0.25)
            record['status'] = 'passed' if child.returncode == 0 else 'failed'
        except ReserveExceeded as error:
            record.update(status='reserve_exceeded', error=str(error))
        except TimeoutError as error:
            record.update(status='timed_out', error=str(error))
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            record.update(status='failed', error=str(error))
        finally:
            if child is not None:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                record['process_exit_code'] = child.returncode
            content = stderr_path.read_text(errors='replace') if stderr_path.exists() else ''
            for line in content.splitlines():
                try:
                    value = json.loads(line)
                    if isinstance(value, dict) and 'phase' in value:
                        record['phases'].append(value)
                except ValueError:
                    pass
            if record['status'] == 'failed' and any(s in content.lower() for s in ['out of memory', 'cuda_error_out_of_memory', 'cudaerroroutofmemory']):
                record.update(status='out_of_memory', error='child reported out of memory; inspect stderr')
            record['last_phase'] = record['phases'][-1] if record['phases'] else None
            record['elapsed_seconds'] = time.monotonic()-start
            record['sampled_peak_device_used_mib'] = max((s['used_mib'] for s in samples if 'used_mib' in s), default=None)
            record['sampled_min_free_mib'] = min((s['free_mib'] for s in samples if 'free_mib' in s), default=None)
            record['sampled_min_host_available_mib'] = min((s['host_available_mib'] for s in samples if 'host_available_mib' in s), default=None)
            per_device = {}
            for sample in samples:
                for device in sample.get('devices', []):
                    summary = per_device.setdefault(device['uuid'], dict(peak_used_mib=0, min_free_mib=device['free_mib']))
                    summary['peak_used_mib'] = max(summary['peak_used_mib'], device['used_mib'])
                    summary['min_free_mib'] = min(summary['min_free_mib'], device['free_mib'])
            record['per_device_memory'] = per_device
        return record
