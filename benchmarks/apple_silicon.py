"""Repeatable CPU/MPS/CUDA execution-shape benchmark for PufferLib training.

Build the selected environment's CPU extension first, for example::

    ./build.sh breakout --cpu
    python benchmarks/apple_silicon.py --env breakout

On an NVIDIA host, ``./build.sh breakout --float`` plus ``--modes cuda``
exercises the same portable Torch trainer with direct accelerator rollouts.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import platform
import stat
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

from pufferlib import _C
from pufferlib.device import resolve_device, resolve_rollout_device, synchronize
from pufferlib.pufferl import load_config
from pufferlib.torch_pufferl import PuffeRL, load_policy


@contextmanager
def clean_argv():
    previous = sys.argv
    sys.argv = [previous[0]]
    try:
        yield
    finally:
        sys.argv = previous


_CONFIG_SECTIONS = ('torch', 'train', 'vec', 'env')
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def make_args(env_name, agents, horizon, minibatch_size, mode, threads,
        amp_dtype, mps_host_alias='auto', compile_policy='off',
        compile_ppo='off', mingru_train_scan='off'):
    with clean_argv():
        args = load_config(env_name)
    training_device, rollout_device = {
        'cpu': ('cpu', 'cpu'),
        'hybrid': ('mps', 'cpu'),
        'mps': ('mps', 'mps'),
        # Same portable Torch trainer and CPU-native vector environment,
        # allowing a controlled accelerator comparison on an NVIDIA host.
        'cuda': ('cuda', 'cuda'),
    }[mode]
    args['slowly'] = True
    args['profile'] = False
    args['torch']['device'] = training_device
    args['torch']['rollout_device'] = rollout_device
    args['torch']['amp_dtype'] = amp_dtype
    args['torch']['mps_host_alias'] = mps_host_alias
    args['torch']['compile_policy'] = compile_policy
    args['torch']['compile_ppo'] = compile_ppo
    args['torch']['mingru_train_scan'] = mingru_train_scan
    args['vec']['total_agents'] = agents
    args['vec']['num_buffers'] = 1
    args['vec']['num_threads'] = threads
    args['train']['horizon'] = horizon
    args['train']['minibatch_size'] = minibatch_size
    args['train']['replay_ratio'] = 1.0
    args['train']['total_timesteps'] = agents * horizon * 100
    args['world_size'] = 1
    return args


def run_mode(env_name, mode, agents, horizon, minibatch_size,
        threads, warmup_epochs, epochs, seed, amp_dtype='float32',
        mps_host_alias='auto', compile_policy='off', compile_ppo='off',
        mingru_train_scan='off'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    effective_amp_dtype = 'float32' if mode == 'cpu' else amp_dtype
    args = make_args(env_name, agents, horizon, minibatch_size,
        mode, threads, effective_amp_dtype, mps_host_alias, compile_policy,
        compile_ppo, mingru_train_scan)
    device = resolve_device(args['torch']['device'], native_cuda=False)
    vec = _C.create_vec(args, 0)
    trainer = None
    try:
        rollout_device = resolve_rollout_device(
            args['torch']['rollout_device'], device, vec_gpu=False)
        policy = load_policy(args, vec, device=rollout_device)
        trainer = PuffeRL(args, vec, policy, verbose=False,
            device=device, rollout_device=rollout_device)
        for _ in range(warmup_epochs):
            trainer.rollouts()
            trainer.train()
        synchronize(device)
        synchronize(rollout_device)

        from torch._dynamo.utils import counters
        counters.clear()

        rollout_seconds = []
        train_seconds = []
        for _ in range(epochs):
            synchronize(device)
            synchronize(rollout_device)
            start = time.perf_counter()
            trainer.rollouts()
            synchronize(rollout_device)
            rollout_seconds.append(time.perf_counter() - start)

            synchronize(device)
            start = time.perf_counter()
            trainer.train()
            synchronize(device)
            train_seconds.append(time.perf_counter() - start)

        if not all(np.isfinite(value) for value in trainer.losses.values()):
            raise RuntimeError(f'non-finite training loss in {mode}: {trainer.losses}')
        if not all(torch.isfinite(param).all().item()
                for param in trainer.policy.parameters()):
            raise RuntimeError(f'non-finite policy parameter in {mode}')
        post_preflight_dynamo_frames_total = int(
            counters['frames']['total'])
        post_preflight_dynamo_unique_graphs = int(
            counters['stats']['unique_graphs'])
    finally:
        if trainer is None:
            vec.close()
        else:
            trainer.close()

    result = _summarize_run(
        mode=mode,
        device=device,
        rollout_device=rollout_device,
        trainer=trainer,
        args=args,
        agents=agents,
        horizon=horizon,
        minibatch_size=minibatch_size,
        epochs=epochs,
        seed=seed,
        threads=threads,
        warmup_epochs=warmup_epochs,
        requested_amp_dtype=amp_dtype,
        effective_amp_dtype=effective_amp_dtype,
        rollout_seconds=rollout_seconds,
        train_seconds=train_seconds,
    )
    result.update({
        'post_preflight_dynamo_frames_total':
            post_preflight_dynamo_frames_total,
        'post_preflight_dynamo_unique_graphs':
            post_preflight_dynamo_unique_graphs,
    })
    return result


def _json_safe(value):
    """Return a detached, JSON-serializable copy of benchmark configuration."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _summarize_run(*, mode, device, rollout_device, trainer, args, agents,
        horizon, minibatch_size, epochs, seed, threads, warmup_epochs,
        requested_amp_dtype, effective_amp_dtype, rollout_seconds,
        train_seconds):
    """Build the complete, machine-readable result for one measured shape."""
    if len(rollout_seconds) != len(train_seconds):
        raise ValueError('rollout and train sample counts must match')
    if len(rollout_seconds) != epochs:
        raise ValueError(
            f'expected {epochs} measured epochs, got {len(rollout_seconds)}')

    steps = agents * horizon
    samples = []
    for epoch, (rollout_s, train_s) in enumerate(
            zip(rollout_seconds, train_seconds), start=1):
        total_s = rollout_s + train_s
        samples.append({
            'epoch': epoch,
            'rollout_seconds': rollout_s,
            'train_seconds': train_s,
            'total_seconds': total_s,
            'sps': steps / total_s,
        })

    totals = [sample['total_seconds'] for sample in samples]
    sps_samples = [sample['sps'] for sample in samples]
    return {
        'mode': mode,
        'training_device': str(device),
        'rollout_device': str(rollout_device),
        'host_horizon_io': trainer.host_horizon_io,
        'mps_host_alias_io': trainer.mps_host_alias_io,
        'requested_mps_host_alias': _json_safe(
            args['torch'].get('mps_host_alias', 'auto')),
        'requested_policy_compile': getattr(
            trainer, 'policy_compile_requested', 'off'),
        'effective_policy_compile': getattr(
            trainer, 'policy_compile_effective', 'off'),
        'policy_compile_reason': getattr(
            trainer, 'policy_compile_reason', None),
        'policy_compile_preflight': bool(getattr(
            trainer, 'policy_compile_preflight', False)),
        'policy_compile_wrapper_verified': bool(getattr(
            trainer, 'policy_compile_wrapper_verified', False)),
        'policy_compile_startup_seconds': float(getattr(
            trainer, 'policy_compile_startup_seconds', 0.0)),
        'requested_mingru_train_scan': getattr(
            trainer, 'mingru_train_scan_requested', 'off'),
        'effective_mingru_train_scan': getattr(
            trainer, 'mingru_train_scan_effective', 'off'),
        'mingru_train_scan_reason': getattr(
            trainer, 'mingru_train_scan_reason', None),
        'mingru_train_scan_preflight': bool(getattr(
            trainer, 'mingru_train_scan_preflight', False)),
        'mingru_train_scan_startup_seconds': float(getattr(
            trainer, 'mingru_train_scan_startup_seconds', 0.0)),
        'requested_ppo_compile': getattr(
            trainer, 'ppo_compile_requested', 'off'),
        'effective_ppo_compile': getattr(
            trainer, 'ppo_compile_effective', 'off'),
        'ppo_compile_reason': getattr(
            trainer, 'ppo_compile_reason', None),
        'ppo_compile_preflight': bool(getattr(
            trainer, 'ppo_compile_preflight', False)),
        'ppo_compile_wrapper_verified': bool(getattr(
            trainer, 'ppo_compile_wrapper_verified', False)),
        'ppo_compile_startup_seconds': float(getattr(
            trainer, 'ppo_compile_startup_seconds', 0.0)),
        'requested_rollout_sampler': getattr(
            trainer, 'rollout_sampler_requested', 'torch_multinomial'),
        'effective_rollout_sampler': getattr(
            trainer, 'rollout_sampler_effective', 'torch_multinomial'),
        'rollout_sampler_reason': getattr(
            trainer, 'rollout_sampler_reason', None),
        'rollout_sampler_startup_seconds': float(getattr(
            trainer, 'rollout_sampler_startup_seconds', 0.0)),
        'optimization_startup_seconds': float(getattr(
            trainer, 'optimization_startup_seconds', 0.0)),
        'pytorch_enable_mps_fallback': os.environ.get(
            'PYTORCH_ENABLE_MPS_FALLBACK'),
        'agents': agents,
        'horizon': horizon,
        'minibatch_size': minibatch_size,
        'epochs': epochs,
        'seed': seed,
        'threads': threads,
        'torch_threads': torch.get_num_threads(),
        'torch_interop_threads': torch.get_num_interop_threads(),
        'warmup_epochs': warmup_epochs,
        'requested_amp_dtype': requested_amp_dtype,
        'effective_amp_dtype': effective_amp_dtype,
        'model_parameters': trainer.model_size,
        'effective_config': {
            section: _json_safe(args.get(section, {}))
            for section in _CONFIG_SECTIONS
        },
        'policy': {
            'encoder': args['torch']['encoder'],
            'network': args['torch']['network'],
            'decoder': args['torch']['decoder'],
            'hidden_size': args['policy']['hidden_size'],
            'num_layers': args['policy']['num_layers'],
            'expansion_factor': args['policy']['expansion_factor'],
        },
        'train': {
            key: args['train'][key] for key in (
                'replay_ratio', 'gamma', 'gae_lambda', 'prio_alpha',
                'prio_beta0', 'vtrace_rho_clip', 'vtrace_c_clip')
        },
        'rollout_ms_median': 1000 * statistics.median(rollout_seconds),
        'train_ms_median': 1000 * statistics.median(train_seconds),
        'total_ms_median': 1000 * statistics.median(totals),
        'sps_median': statistics.median(sps_samples),
        'sps_min': min(sps_samples),
        'sps_max': max(sps_samples),
        'samples': samples,
    }


def _command_output(*command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _command_bytes(*command):
    try:
        return subprocess.check_output(command, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None


def working_tree_patch_fingerprint(repo_dir=_REPO_DIR):
    """Hash source patches without returning their contents.

    Generated ``work/`` evidence is excluded so redirecting this benchmark's
    own output cannot change the implementation fingerprint while it runs.
    """
    root = _command_output(
        'git', '-C', repo_dir, 'rev-parse', '--show-toplevel')
    if not root:
        return None

    patch = _command_bytes(
        'git', '-C', root, 'diff', '--binary', '--no-ext-diff', 'HEAD', '--')
    untracked = _command_bytes(
        'git', '-C', root, 'ls-files', '--others', '--exclude-standard',
        '--exclude=work/**', '-z')
    if patch is None or untracked is None:
        return None

    digest = hashlib.sha256()
    digest.update(b'pufferlib-working-tree-patch-v2\0')
    digest.update(len(patch).to_bytes(8, byteorder='big'))
    digest.update(patch)

    paths = [path for path in untracked.split(b'\0') if path]
    untracked_bytes = 0
    unreadable_files = 0
    for raw_path in paths:
        digest.update(b'untracked\0')
        digest.update(len(raw_path).to_bytes(8, byteorder='big'))
        digest.update(raw_path)
        path = os.path.join(root, os.fsdecode(raw_path))
        try:
            file_stat = os.lstat(path)
            mode = file_stat.st_mode
            digest.update(mode.to_bytes(8, byteorder='big'))
            if stat.S_ISLNK(mode):
                contents = os.fsencode(os.readlink(path))
                untracked_bytes += len(contents)
                digest.update(contents)
            elif stat.S_ISREG(mode):
                with open(path, 'rb') as untracked_file:
                    while True:
                        chunk = untracked_file.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                        untracked_bytes += len(chunk)
            else:
                digest.update(b'non-regular-file')
        except OSError:
            unreadable_files += 1
            digest.update(b'unreadable-file')

    return {
        'algorithm': 'sha256',
        'schema': 'pufferlib-working-tree-patch-v2',
        'digest': digest.hexdigest(),
        'tracked_patch_bytes': len(patch),
        'untracked_files': len(paths),
        'untracked_bytes': untracked_bytes,
        'unreadable_files': unreadable_files,
    }


def _cpu_brand():
    brand = _command_output('sysctl', '-n', 'machdep.cpu.brand_string')
    if brand:
        return brand
    try:
        with open('/proc/cpuinfo', encoding='utf-8') as cpuinfo:
            for line in cpuinfo:
                if line.lower().startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _integer_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _system_profiler_hardware():
    raw = _command_output(
        'system_profiler', 'SPHardwareDataType', 'SPDisplaysDataType', '-json')
    if not raw:
        return {}
    try:
        report = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    hardware_entries = report.get('SPHardwareDataType') or []
    display_entries = report.get('SPDisplaysDataType') or []
    hardware = hardware_entries[0] if hardware_entries else {}
    gpu = next((entry for entry in display_entries
        if entry.get('sppci_device_type') == 'spdisplays_gpu'), {})
    return {
        'machine_name': hardware.get('machine_name'),
        'hardware_model': hardware.get('machine_model'),
        'chip': hardware.get('chip_type'),
        'gpu_model': gpu.get('sppci_model') or gpu.get('_name'),
        'gpu_cores': _integer_or_none(gpu.get('sppci_cores')),
    }


def _total_memory_bytes():
    memory = _integer_or_none(_command_output('sysctl', '-n', 'hw.memsize'))
    if memory is not None:
        return memory
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (OSError, ValueError):
        return None


def system_metadata():
    """Capture enough context to make throughput results reproducible."""
    cuda_available = bool(torch.cuda.is_available())
    profiler_hardware = _system_profiler_hardware()
    hardware_model = (
        _command_output('sysctl', '-n', 'hw.model')
        or profiler_hardware.get('hardware_model'))
    metadata = {
        'platform': platform.platform(),
        'machine': platform.machine(),
        'macos_version': (
            platform.mac_ver()[0]
            or _command_output('sw_vers', '-productVersion')),
        'macos_build': _command_output('sw_vers', '-buildVersion'),
        'machine_name': profiler_hardware.get('machine_name'),
        'hardware_model': hardware_model,
        'chip': profiler_hardware.get('chip'),
        'memory_bytes': _total_memory_bytes(),
        'gpu_model': profiler_hardware.get('gpu_model'),
        'gpu_cores': profiler_hardware.get('gpu_cores'),
        'cpu_brand': _cpu_brand(),
        'cpu_count': os.cpu_count(),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'torch_git_revision': getattr(torch.version, 'git_version', None),
        'git_revision': _command_output(
            'git', '-C', _REPO_DIR, 'rev-parse', 'HEAD'),
        'git_dirty': bool(_command_output(
            'git', '-C', _REPO_DIR, 'status', '--porcelain')),
        'working_tree_patch': working_tree_patch_fingerprint(),
        'environment_variables': {
            'PYTORCH_ENABLE_MPS_FALLBACK': os.environ.get(
                'PYTORCH_ENABLE_MPS_FALLBACK'),
            'TORCHINDUCTOR_FORCE_LAYOUT_OPT': os.environ.get(
                'TORCHINDUCTOR_FORCE_LAYOUT_OPT'),
            'TORCHINDUCTOR_LAYOUT_OPTIMIZATION': os.environ.get(
                'TORCHINDUCTOR_LAYOUT_OPTIMIZATION'),
            'TORCHDYNAMO_DISABLE': os.environ.get('TORCHDYNAMO_DISABLE'),
            'TORCH_BISECT_BACKEND': os.environ.get('TORCH_BISECT_BACKEND'),
        },
        'compiled_environment': getattr(_C, 'env_name', None),
        'extension_gpu': bool(getattr(_C, 'gpu', False)),
        'extension_precision_bytes': getattr(_C, 'precision_bytes', None),
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'mps_built': bool(torch.backends.mps.is_built()),
        'mps_available': bool(torch.backends.mps.is_available()),
        'mps_host_alias_api': hasattr(torch.mps, '_host_alias_storage'),
        'cuda_available': cuda_available,
        'cuda_device_count': torch.cuda.device_count() if cuda_available else 0,
        'cuda_runtime': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
    }
    if cuda_available:
        current_device = torch.cuda.current_device()
        metadata.update({
            'cuda_current_device': current_device,
            'cuda_device_name': torch.cuda.get_device_name(current_device),
            'cuda_capability': list(torch.cuda.get_device_capability(current_device)),
        })
    else:
        metadata.update({
            'cuda_current_device': None,
            'cuda_device_name': None,
            'cuda_capability': None,
        })
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='breakout')
    parser.add_argument('--agents', type=int, nargs='+', default=[256, 1024, 4096])
    parser.add_argument('--horizon', type=int, default=64)
    parser.add_argument('--minibatch-size', type=int, default=65536)
    parser.add_argument('--threads', type=int, default=18,
        help='OpenMP threads used by the native environment step')
    parser.add_argument('--torch-threads', type=int, default=torch.get_num_threads(),
        help='PyTorch CPU intra-op threads (independent of environment threads)')
    parser.add_argument('--torch-interop-threads', type=int,
        default=torch.get_num_interop_threads(),
        help='PyTorch CPU inter-op threads (must be set before benchmark work)')
    parser.add_argument('--warmup-epochs', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp-dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--mps-host-alias', choices=['auto', 'on', 'off'],
        default='auto', help='request Apple unified-memory rollout aliasing')
    parser.add_argument('--compile-policy', choices=['off', 'auto', 'inductor'],
        default='off', help='guarded policy compiler mode')
    parser.add_argument('--compile-ppo', choices=['off', 'auto', 'inductor'],
        default='off', help='guarded FP32 policy + PPO compiler mode')
    parser.add_argument('--mingru-train-scan', choices=['off', 'auto', 'metal'],
        default='off', help='guarded FP32 training-only Metal MinGRU scan')
    parser.add_argument('--modes', nargs='+', choices=['cpu', 'hybrid', 'mps', 'cuda'],
        default=['cpu', 'hybrid', 'mps'])
    options = parser.parse_args()
    if (options.threads < 1 or options.torch_threads < 1
            or options.torch_interop_threads < 1):
        parser.error('thread counts must be positive')
    torch.set_num_threads(options.torch_threads)
    torch.set_num_interop_threads(options.torch_interop_threads)
    if options.amp_dtype != 'float32' and 'hybrid' in options.modes:
        parser.error('BF16 requires direct MPS rollout; remove hybrid from --modes')

    compiled_env = getattr(_C, 'env_name', None)
    if compiled_env != options.env:
        raise RuntimeError(
            f'_C was built for {compiled_env!r}; run ./build.sh {options.env} --cpu')
    if any(mode in ('hybrid', 'mps') for mode in options.modes):
        resolve_device('mps')
    if 'cuda' in options.modes:
        resolve_device('cuda')

    results = []
    for agents in options.agents:
        batch_size = agents * options.horizon
        minibatch_size = min(options.minibatch_size, batch_size)
        minibatch_size -= minibatch_size % options.horizon
        for mode in options.modes:
            result = run_mode(
                options.env, mode, agents, options.horizon, minibatch_size,
                options.threads, options.warmup_epochs, options.epochs,
                options.seed, options.amp_dtype, options.mps_host_alias,
                options.compile_policy, options.compile_ppo,
                options.mingru_train_scan)
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    print(json.dumps({
        'environment': options.env,
        'system': system_metadata(),
        'results': results,
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
