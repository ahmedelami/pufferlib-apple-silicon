import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    'apple_silicon_benchmark', _ROOT / 'benchmarks' / 'apple_silicon.py')
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def _fake_args():
    return {
        'torch': {
            'device': 'cpu',
            'rollout_device': 'cpu',
            'amp_dtype': 'float32',
            'mps_host_alias': 'off',
            'encoder': 'DefaultEncoder',
            'network': 'MinGRU',
            'decoder': 'DefaultDecoder',
            'future_torch_option': 17,
        },
        'train': {
            'replay_ratio': 1.0,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'prio_alpha': 0.8,
            'prio_beta0': 0.2,
            'vtrace_rho_clip': 1.0,
            'vtrace_c_clip': 1.0,
            'future_train_option': [1, 2],
        },
        'vec': {
            'total_agents': 4,
            'num_buffers': 1,
            'num_threads': 2,
            'future_vec_option': True,
        },
        'env': {
            'frameskip': 4,
            'future_env_option': 'kept',
        },
        'policy': {
            'hidden_size': 64,
            'num_layers': 2,
            'expansion_factor': 1,
        },
    }


def test_mps_host_alias_off_cli_override_parses_without_mps(monkeypatch):
    monkeypatch.setattr(
        sys, 'argv', ['puffer', '--torch.mps-host-alias', 'off'])
    args = benchmark.load_config('breakout')
    assert args['torch']['mps_host_alias'] == 'off'


def test_run_summary_keeps_raw_samples_and_complete_effective_config(
        monkeypatch):
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK', '0')
    args = _fake_args()
    trainer = SimpleNamespace(
        host_horizon_io=False,
        mps_host_alias_io=False,
        policy_compile_wrapper_verified=True,
        rollout_sampler_requested='fused_mps_philox',
        rollout_sampler_effective='torch_multinomial',
        rollout_sampler_reason='compiled path inactive',
        rollout_sampler_startup_seconds=0.125,
        optimization_startup_seconds=0.5,
        model_size=1234,
    )

    result = benchmark._summarize_run(
        mode='cpu',
        device=torch.device('cpu'),
        rollout_device=torch.device('cpu'),
        trainer=trainer,
        args=args,
        agents=4,
        horizon=8,
        minibatch_size=32,
        epochs=2,
        seed=42,
        threads=2,
        warmup_epochs=1,
        requested_amp_dtype='bfloat16',
        effective_amp_dtype='float32',
        rollout_seconds=[0.25, 0.5],
        train_seconds=[0.75, 0.5],
    )

    assert result['requested_mps_host_alias'] == 'off'
    assert result['pytorch_enable_mps_fallback'] == '0'
    assert result['effective_config'] == {
        section: args[section]
        for section in ('torch', 'train', 'vec', 'env')
    }
    assert result['samples'] == [
        {
            'epoch': 1,
            'rollout_seconds': 0.25,
            'train_seconds': 0.75,
            'total_seconds': 1.0,
            'sps': 32.0,
        },
        {
            'epoch': 2,
            'rollout_seconds': 0.5,
            'train_seconds': 0.5,
            'total_seconds': 1.0,
            'sps': 32.0,
        },
    ]
    assert result['sps_median'] == 32.0
    assert result['requested_amp_dtype'] == 'bfloat16'
    assert result['effective_amp_dtype'] == 'float32'
    assert result['requested_rollout_sampler'] == 'fused_mps_philox'
    assert result['effective_rollout_sampler'] == 'torch_multinomial'
    assert result['rollout_sampler_reason'] == 'compiled path inactive'
    assert result['rollout_sampler_startup_seconds'] == 0.125
    assert result['optimization_startup_seconds'] == 0.5
    assert result['policy_compile_wrapper_verified'] is True
    json.dumps(result)


@pytest.mark.parametrize(
    ('rollouts', 'training', 'epochs'),
    [([0.1], [], 1), ([0.1], [0.2], 2)],
)
def test_run_summary_rejects_incomplete_epoch_samples(
        rollouts, training, epochs):
    with pytest.raises(ValueError):
        benchmark._summarize_run(
            mode='cpu',
            device=torch.device('cpu'),
            rollout_device=torch.device('cpu'),
            trainer=SimpleNamespace(
                host_horizon_io=False,
                mps_host_alias_io=False,
                model_size=1,
            ),
            args=_fake_args(),
            agents=1,
            horizon=1,
            minibatch_size=1,
            epochs=epochs,
            seed=1,
            threads=1,
            warmup_epochs=0,
            requested_amp_dtype='float32',
            effective_amp_dtype='float32',
            rollout_seconds=rollouts,
            train_seconds=training,
        )


def test_working_tree_fingerprint_covers_tracked_and_untracked_content(tmp_path):
    def git(*args):
        return subprocess.run(
            ['git', *args], cwd=tmp_path, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    git('init', '-q')
    git('config', 'user.name', 'Benchmark Test')
    git('config', 'user.email', 'benchmark@example.invalid')
    tracked = tmp_path / 'tracked.txt'
    tracked.write_text('baseline\n', encoding='utf-8')
    git('add', 'tracked.txt')
    git('commit', '-qm', 'baseline')

    clean = benchmark.working_tree_patch_fingerprint(str(tmp_path))
    tracked.write_text('tracked-secret-change\n', encoding='utf-8')
    (tmp_path / 'untracked.txt').write_text(
        'untracked-secret-change\n', encoding='utf-8')
    dirty = benchmark.working_tree_patch_fingerprint(str(tmp_path))
    (tmp_path / 'work').mkdir()
    (tmp_path / 'work' / 'live-output.json').write_text(
        'generated benchmark evidence\n', encoding='utf-8')
    dirty_with_work_output = benchmark.working_tree_patch_fingerprint(
        str(tmp_path))

    assert clean['digest'] != dirty['digest']
    assert dirty_with_work_output == dirty
    assert dirty['algorithm'] == 'sha256'
    assert dirty['schema'] == 'pufferlib-working-tree-patch-v2'
    assert len(dirty['digest']) == 64
    assert dirty['tracked_patch_bytes'] > 0
    assert dirty['untracked_files'] == 1
    assert dirty['untracked_bytes'] > 0
    assert dirty['unreadable_files'] == 0
    serialized = json.dumps(dirty)
    assert 'tracked-secret-change' not in serialized
    assert 'untracked-secret-change' not in serialized


def test_system_profiler_extracts_only_reproducibility_identity(monkeypatch):
    report = {
        'SPHardwareDataType': [{
            'machine_name': 'MacBook Pro',
            'machine_model': 'Mac17,8',
            'chip_type': 'Apple M5 Pro',
            'serial_number': 'must-not-be-recorded',
            'platform_UUID': 'must-not-be-recorded-either',
        }],
        'SPDisplaysDataType': [{
            '_name': 'Apple M5 Pro',
            'sppci_device_type': 'spdisplays_gpu',
            'sppci_model': 'Apple M5 Pro',
            'sppci_cores': '20',
        }],
    }
    monkeypatch.setattr(
        benchmark, '_command_output', lambda *args: json.dumps(report))

    hardware = benchmark._system_profiler_hardware()

    assert hardware == {
        'machine_name': 'MacBook Pro',
        'hardware_model': 'Mac17,8',
        'chip': 'Apple M5 Pro',
        'gpu_model': 'Apple M5 Pro',
        'gpu_cores': 20,
    }
    assert 'serial' not in json.dumps(hardware).lower()
    assert 'uuid' not in json.dumps(hardware).lower()


def test_system_metadata_json_shape_without_accelerator_work(monkeypatch):
    commands = {
        ('sysctl', '-n', 'hw.model'): 'Mac17,8',
        ('sw_vers', '-productVersion'): '27.0',
        ('sw_vers', '-buildVersion'): '26A5378j',
        ('git', '-C', str(benchmark._REPO_DIR), 'rev-parse', 'HEAD'): 'abc123',
        ('git', '-C', str(benchmark._REPO_DIR), 'status', '--porcelain'): ' M file',
    }
    monkeypatch.setattr(
        benchmark, '_command_output', lambda *args: commands.get(args))
    monkeypatch.setattr(benchmark, '_system_profiler_hardware', lambda: {
        'machine_name': 'MacBook Pro',
        'hardware_model': 'Mac17,8',
        'chip': 'Apple M5 Pro',
        'gpu_model': 'Apple M5 Pro',
        'gpu_cores': 20,
    })
    monkeypatch.setattr(benchmark, '_total_memory_bytes', lambda: 24 * 2**30)
    monkeypatch.setattr(benchmark, '_cpu_brand', lambda: 'Apple M5 Pro')
    monkeypatch.setattr(
        benchmark, 'working_tree_patch_fingerprint',
        lambda: {'algorithm': 'sha256', 'digest': 'a' * 64})
    monkeypatch.setattr(benchmark.torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(benchmark.torch.backends.mps, 'is_built', lambda: True)
    monkeypatch.setattr(benchmark.torch.backends.mps, 'is_available', lambda: True)
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK', '0')

    metadata = benchmark.system_metadata()

    assert metadata['macos_build'] == '26A5378j'
    assert metadata['hardware_model'] == 'Mac17,8'
    assert metadata['memory_bytes'] == 24 * 2**30
    assert metadata['gpu_model'] == 'Apple M5 Pro'
    assert metadata['gpu_cores'] == 20
    assert metadata['git_revision'] == 'abc123'
    assert metadata['git_dirty'] is True
    assert metadata['working_tree_patch']['digest'] == 'a' * 64
    assert metadata['environment_variables'] == {
        'PYTORCH_ENABLE_MPS_FALLBACK': '0',
        'TORCHINDUCTOR_FORCE_LAYOUT_OPT': None,
        'TORCHINDUCTOR_LAYOUT_OPTIMIZATION': None,
        'TORCHDYNAMO_DISABLE': None,
        'TORCH_BISECT_BACKEND': None,
    }
    assert metadata['torch_git_revision'] == torch.version.git_version
    assert metadata['cuda_available'] is False
    json.dumps(metadata)
