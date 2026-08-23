## puffer [train | eval | sweep] [env_name] [optional args] -- See https://puffer.ai for full detail0
# This is the same as python -m pufferlib.pufferl [train | eval | sweep] [env_name] [optional args]
# Distributed example: torchrun --standalone --nnodes=1 --nproc-per-node=6 \
#   -m pufferlib.pufferl train nmmo3 --slowly

import os
import glob
import json
import time
import ctypes
from functools import lru_cache
import platform
import subprocess
from copy import deepcopy
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType

import numpy as np

import torch
import torch.distributed

import pufferlib
import pufferlib.models
import pufferlib.pufferl
from pufferlib.muon import Muon
from pufferlib import _C
from pufferlib.device import (
    agent_major_rollout,
    mps_host_alias,
    resolve_device,
    resolve_rollout_device,
    synchronize,
)
if _C.precision_bytes != 4:
    raise RuntimeError(
        f'_C was compiled with bf16 precision (precision_bytes={_C.precision_bytes}). '
        'The PyTorch backend requires float32. Rerun build.sh with --float'
    )

_OBS_DTYPE_MAP = {
    'ByteTensor':   torch.uint8,
    'FloatTensor':  torch.float32,
}

_TORCH_TO_TYPESTR = {
    torch.uint8:   '|u1',
    torch.float32: '<f4',
}

def _log_prob(logits, value):
    value = value.long().unsqueeze(-1)
    value, log_pmf = torch.broadcast_tensors(value, logits)
    value = value[..., :1]
    return log_pmf.gather(-1, value).squeeze(-1)

def sample_logits(logits, action=None, compute_entropy=True,
        rollout_sampler=None):
    """Sample policy actions and evaluate their log probability.

    Rollout does not consume entropy, so ``compute_entropy=False`` avoids an
    otherwise redundant probability pass on every environment step. Discrete
    policies avoid computing rollout entropy and supplied-action sampling
    probabilities. The remaining operations deliberately retain their original
    order so fixed-seed action trajectories stay bitwise reproducible.
    """
    is_discrete = isinstance(logits, torch.Tensor)
    if isinstance(logits, torch.distributions.Normal):
        batch = logits.loc.shape[0]
        if action is None:
            action = logits.sample().view(batch, -1)
        log_probs = logits.log_prob(action.view(batch, -1)).sum(1)
        logits_entropy = None
        if compute_entropy:
            logits_entropy = logits.entropy().view(batch, -1).sum(1)
        return action, log_probs, logits_entropy
    elif is_discrete:
        logits = logits.unsqueeze(0)
    else: # multi-discrete
        logits = torch.nn.utils.rnn.pad_sequence(
            [l.transpose(0,1) for l in logits],
            batch_first=False,
            padding_value=-torch.inf
        ).permute(1,2,0)

    log_probs = logits - logits.logsumexp(dim=-1, keepdim=True)
    if action is None:
        # Keep the pre-optimization softmax path exactly: exp(log_softmax)
        # is distributionally equivalent but can cross a multinomial boundary
        # and change a fixed-seed trajectory by one action.
        probs = logits.softmax(dim=-1)
    elif compute_entropy:
        probs = log_probs.softmax(dim=-1)
    else:
        probs = None

    sampled_logprob = None
    if action is None:
        if rollout_sampler is not None and is_discrete \
                and not compute_entropy:
            action, sampled_logprob = rollout_sampler.sample(
                probs.reshape(-1, probs.shape[-1]),
                log_probs.reshape(-1, log_probs.shape[-1]))
            action = action.reshape(probs.shape[:-1])
            sampled_logprob = sampled_logprob.reshape(probs.shape[:-1])
        else:
            probs = torch.nan_to_num(probs, 1e-8, 1e-8, 1e-8)
            action = torch.multinomial(
                probs.reshape(-1, probs.shape[-1]),
                1, replacement=True).int()
            action = action.reshape(probs.shape[:-1])
    else:
        batch = logits[0].shape[0]
        action = action.view(batch, -1).T

    logprob = (
        sampled_logprob
        if sampled_logprob is not None
        else _log_prob(log_probs, action))
    logits_entropy = None
    if compute_entropy:
        min_real = torch.finfo(log_probs.dtype).min
        safe_log_probs = torch.clamp(log_probs, min=min_real)
        logits_entropy = -(safe_log_probs * probs).sum(-1).sum(0)

    if is_discrete:
        if logits_entropy is not None:
            logits_entropy = logits_entropy.squeeze(0)
        return action.T, logprob.squeeze(0), logits_entropy

    return action.T, logprob.sum(0), logits_entropy

def _float_policy_output(logits):
    if isinstance(logits, torch.Tensor):
        return logits.float()
    if isinstance(logits, torch.distributions.Normal):
        return torch.distributions.Normal(logits.loc.float(), logits.scale.float())
    return tuple(value.float() for value in logits)


_VALIDATED_COMPILE_TORCH_VERSION = '2.13.0'
_VALIDATED_COMPILE_TORCH_GIT = 'cf30153c4c131c8164ee7798e5022d810682e2cb'
_VALIDATED_COMPILE_HARDWARE = 'Mac17,8'
_VALIDATED_COMPILE_CHIP = 'Apple M5 Pro'
_VALIDATED_COMPILE_GPU_CORES = 20
_VALIDATED_COMPILE_MEMORY_BYTES = 24 * 2**30
_VALIDATED_COMPILE_MACOS_VERSION = '27.0'
_VALIDATED_COMPILE_MACOS_BUILD = '26A5378j'
_VALIDATED_POLICY_FORWARD = pufferlib.models.Policy.forward
_VALIDATED_POLICY_FORWARD_EVAL = pufferlib.models.Policy.forward_eval
_VALIDATED_MINGRU_FORWARD_TRAIN = pufferlib.models.MinGRU.forward_train
_VALIDATED_PPO_COMPILE_BREAK_EVEN_TIMESTEPS = 54_800_000


def _quiet_command(*command):
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return None


def _integer_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _truthy_environment(name):
    return os.environ.get(name, '').strip().lower() in (
        '1', 'true', 'yes', 'on')


def _compiler_bisect_backend():
    try:
        from torch._inductor.compiler_bisector import CompilerBisector
        return CompilerBisector.get_backend()
    except Exception as exc:
        # Failure to prove that the requested backend is unmodified must keep
        # the exact validated path off, not silently count as no override.
        return f'probe failed: {type(exc).__name__}'


def _compile_hardware_profile():
    raw = _quiet_command(
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
        'chip': hardware.get('chip_type'),
        'gpu_model': gpu.get('sppci_model') or gpu.get('_name'),
        'gpu_cores': _integer_or_none(gpu.get('sppci_cores')),
    }


@lru_cache(maxsize=1)
def _compile_system_identity():
    hardware_profile = _compile_hardware_profile()
    return {
        'system': platform.system(),
        'machine': platform.machine(),
        'hardware': _quiet_command('sysctl', '-n', 'hw.model'),
        'chip': hardware_profile.get('chip'),
        'gpu_model': hardware_profile.get('gpu_model'),
        'gpu_cores': hardware_profile.get('gpu_cores'),
        'memory_bytes': _integer_or_none(
            _quiet_command('sysctl', '-n', 'hw.memsize')),
        'macos_version': _quiet_command('sw_vers', '-productVersion'),
        'macos_build': _quiet_command('sw_vers', '-buildVersion'),
        'torch_git': getattr(torch.version, 'git_version', None),
    }


def _policy_geometry_mismatches(policy, device):
    models = pufferlib.models
    if not (
            type(policy) is models.Policy
            and type(policy.encoder) is models.DefaultEncoder
            and type(policy.network) is models.MinGRU
            and type(policy.decoder) is models.DefaultDecoder):
        return []

    mismatches = []
    encoder = policy.encoder.encoder
    network = policy.network
    decoder = policy.decoder
    if ('forward' in policy.__dict__
            or getattr(policy.forward, '__func__', None)
                is not _VALIDATED_POLICY_FORWARD):
        mismatches.append('actual policy forward implementation is not validated')
    if ('forward_eval' in policy.__dict__
            or getattr(policy.forward_eval, '__func__', None)
                is not _VALIDATED_POLICY_FORWARD_EVAL):
        mismatches.append(
            'actual policy forward_eval implementation is not validated')
    network_forward = getattr(network.forward_train, '__func__', None)
    metal_module = getattr(pufferlib, 'mps_mingru', None)
    metal_forward = getattr(metal_module, 'forward_train', None)
    portable_network_forward = (
        'forward_train' not in network.__dict__
        and network_forward is _VALIDATED_MINGRU_FORWARD_TRAIN)
    installed_metal_forward = (
        'forward_train' in network.__dict__
        and metal_forward is not None
        and network_forward is metal_forward)
    if not (portable_network_forward or installed_metal_forward):
        mismatches.append(
            'actual MinGRU forward_train implementation is not validated')
    if (encoder.in_features, encoder.out_features) != (118, 64):
        mismatches.append('actual encoder geometry is not 118x64')
    if (network.hidden_size, network.num_layers) != (64, 2):
        mismatches.append('actual MinGRU geometry is not 2x64')
    if (len(network.layers) != 2
            or any(tuple(layer.weight.shape) != (192, 64)
                for layer in network.layers)):
        mismatches.append('actual MinGRU layer weights are not 2x[192,64]')
    if (decoder.nvec != (3,) or decoder.is_continuous
            or tuple(decoder.decoder.weight.shape) != (3, 64)
            or tuple(decoder.value_function.weight.shape) != (1, 64)):
        mismatches.append('actual decoder geometry is not discrete [3] at width 64')
    parameters = tuple(policy.parameters())
    if sum(parameter.numel() for parameter in parameters) != 32_452:
        mismatches.append('actual policy parameter count is not 32452')
    expected_device = torch.device(device)
    if any(
            parameter.device.type != expected_device.type
            or (expected_device.index is not None
                and parameter.device.index != expected_device.index)
            for parameter in parameters):
        mismatches.append('policy Parameters are not on the training device')
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        mismatches.append('policy Parameters are not float32')
    return mismatches


def _policy_compile_mismatches(args, vec, policy, device, rollout_device,
        amp_dtype, mps_host_alias_io):
    """Return why the measured M5 Pro policy graph is not applicable.

    The current MPS Inductor backend is still shape- and model-sensitive. Keep
    this optimization narrower than the general eager MPS port until each new
    environment/model/runtime combination has its own correctness and learning
    gate.
    """
    models = pufferlib.models
    torch_version = str(torch.__version__).split('+', 1)[0]
    act_sizes = [int(size) for size in vec.act_sizes]
    identity = _compile_system_identity()
    policy_encoder = getattr(policy, 'encoder', None)
    policy_network = getattr(policy, 'network', None)
    policy_decoder = getattr(policy, 'decoder', None)
    mismatches = []
    checks = (
        (args.get('env_name') == 'breakout', 'environment is not breakout'),
        (torch_version == _VALIDATED_COMPILE_TORCH_VERSION,
            f'torch {torch_version} is not validated torch '
            f'{_VALIDATED_COMPILE_TORCH_VERSION}'),
        (identity['torch_git'] == _VALIDATED_COMPILE_TORCH_GIT,
            'Torch source revision is not the validated build'),
        (identity['system'] == 'Darwin' and identity['machine'] == 'arm64',
            'host is not Darwin arm64'),
        (identity['hardware'] == _VALIDATED_COMPILE_HARDWARE,
            f'hardware is not {_VALIDATED_COMPILE_HARDWARE}'),
        (identity['chip'] == _VALIDATED_COMPILE_CHIP,
            f'chip is not {_VALIDATED_COMPILE_CHIP}'),
        (identity['gpu_model'] == _VALIDATED_COMPILE_CHIP,
            f'GPU is not {_VALIDATED_COMPILE_CHIP}'),
        (identity['gpu_cores'] == _VALIDATED_COMPILE_GPU_CORES,
            f'GPU core count is not {_VALIDATED_COMPILE_GPU_CORES}'),
        (identity['memory_bytes'] == _VALIDATED_COMPILE_MEMORY_BYTES,
            'memory size is not the validated 24 GiB'),
        (identity['macos_version'] == _VALIDATED_COMPILE_MACOS_VERSION,
            f'macOS is not {_VALIDATED_COMPILE_MACOS_VERSION}'),
        (identity['macos_build'] == _VALIDATED_COMPILE_MACOS_BUILD,
            f'macOS build is not {_VALIDATED_COMPILE_MACOS_BUILD}'),
        (torch.device(device).type == 'mps', 'training device is not MPS'),
        (torch.device(rollout_device).type == 'mps',
            'rollout device is not MPS'),
        (torch.device(device) == torch.device(rollout_device),
            'training and rollout devices differ'),
        (amp_dtype in (None, torch.bfloat16),
            'AMP dtype is not float32 or bfloat16'),
        (bool(mps_host_alias_io), 'MPS host aliasing is inactive'),
        (os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0',
            'PYTORCH_ENABLE_MPS_FALLBACK is not 0'),
        (os.environ.get('TORCHINDUCTOR_FORCE_LAYOUT_OPT', '0') != '1',
            'TORCHINDUCTOR_FORCE_LAYOUT_OPT is enabled'),
        (not bool(getattr(torch._dynamo.config, 'suppress_errors', False)),
            'torch._dynamo.config.suppress_errors is enabled'),
        (not bool(getattr(torch._dynamo.config, 'disable', False)),
            'torch._dynamo.config.disable is enabled'),
        (not _truthy_environment('TORCHDYNAMO_DISABLE'),
            'TORCHDYNAMO_DISABLE is enabled'),
        (_compiler_bisect_backend() is None,
            'Torch compiler bisector backend override is active'),
        (int(args.get('world_size', 1)) == 1,
            'distributed training is enabled'),
        (bool(args.get('reset_state', True)),
            'persistent cross-horizon recurrent state is unvalidated'),
        (not bool(getattr(vec, 'gpu', False)),
            'native vector backend is not CPU'),
        (type(policy) is models.Policy, 'policy class is not Policy'),
        (type(policy_encoder) is models.DefaultEncoder,
            'encoder is not DefaultEncoder'),
        (type(policy_network) is models.MinGRU, 'network is not MinGRU'),
        (type(policy_decoder) is models.DefaultDecoder,
            'decoder is not DefaultDecoder'),
        (int(vec.total_agents) == 4096, 'total_agents is not 4096'),
        (int(vec.obs_size) == 118, 'observation size is not 118'),
        (act_sizes == [3], 'action sizes are not [3]'),
        (int(args['train']['horizon']) == 64, 'horizon is not 64'),
        (int(args['train']['minibatch_size']) == 65_536,
            'minibatch_size is not 65536'),
        (int(args['policy']['hidden_size']) == 64,
            'hidden_size is not 64'),
        (int(args['policy']['num_layers']) == 2,
            'num_layers is not 2'),
    )
    for matched, reason in checks:
        if not matched:
            mismatches.append(reason)
    mismatches.extend(_policy_geometry_mismatches(policy, device))
    return mismatches


def _mingru_train_scan_mismatches(args, vec, policy, device, rollout_device,
        amp_dtype, mps_host_alias_io):
    """Return why the fixed-shape FP32 Metal training scan is inapplicable."""
    mismatches = _policy_compile_mismatches(
        args, vec, policy, device, rollout_device, amp_dtype,
        mps_host_alias_io)
    network = getattr(policy, 'network', None)
    compile_policy = str(
        args.get('torch', {}).get('compile_policy', 'off')).strip().lower()
    horizon = int(args.get('train', {}).get('horizon', 0))
    minibatch_size = int(
        args.get('train', {}).get('minibatch_size', 0))
    checks = (
        (amp_dtype is None, 'Metal MinGRU is validated only for FP32'),
        (compile_policy in ('auto', 'inductor'),
            'validated policy compilation is not requested'),
        (horizon == 64 and minibatch_size // max(horizon, 1) == 1024,
            'training sequence geometry is not [1024,64]'),
        (pufferlib.models.MinGRU.forward_train
            is _VALIDATED_MINGRU_FORWARD_TRAIN,
            'MinGRU.forward_train class method is modified'),
        (network is not None
            and 'forward_train' not in getattr(network, '__dict__', {})
            and getattr(getattr(network, 'forward_train', None), '__func__', None)
                is _VALIDATED_MINGRU_FORWARD_TRAIN,
            'actual MinGRU forward_train implementation is not validated'),
    )
    for matched, reason in checks:
        if not matched and reason not in mismatches:
            mismatches.append(reason)
    return mismatches


def _prepare_mingru_train_scan(args, vec, policy, device, rollout_device,
        amp_dtype, mps_host_alias_io):
    """Guard and install the scan on one network instance before compilation.

    Returns metadata plus the installed network (or ``None``). The caller must
    restore the instance method if policy compilation does not promote.
    """
    requested = str(
        args.get('torch', {}).get('mingru_train_scan', 'off')).strip().lower()
    if requested not in ('off', 'auto', 'metal'):
        raise ValueError(
            'torch.mingru_train_scan must be off, auto, or metal')
    if requested == 'off':
        return (requested, 'off', 'disabled by configuration',
            0.0, False, None)

    attempt_started = time.perf_counter()
    mismatches = _mingru_train_scan_mismatches(
        args, vec, policy, device, rollout_device, amp_dtype,
        mps_host_alias_io)
    if mismatches:
        reason = '; '.join(mismatches)
        if requested == 'auto':
            return (requested, 'off', reason,
                time.perf_counter() - attempt_started, False, None)
        raise ValueError(
            'torch.mingru_train_scan=metal requires the validated FP32 '
            f'Breakout/M5 policy configuration: {reason}')

    try:
        from pufferlib import mps_mingru
        network = policy.network
        network.forward_train = MethodType(mps_mingru.forward_train, network)
    except Exception as exc:
        startup_seconds = time.perf_counter() - attempt_started
        if requested == 'auto':
            return (requested, 'off',
                f'setup failed: {type(exc).__name__}: {exc}',
                startup_seconds, False, None)
        raise RuntimeError(
            'failed to install the validated Metal MinGRU scan') from exc

    return (requested, 'pending',
        'guarded instance override awaiting policy compiler preflight',
        time.perf_counter() - attempt_started, False, network)


def _restore_mingru_train_scan(network):
    if network is not None and 'forward_train' in network.__dict__:
        del network.__dict__['forward_train']


def _finalize_mingru_train_scan(prepared, policy_compile_effective,
        policy_compile_preflight, policy_compile_reason):
    requested, effective, reason, startup_seconds, preflight, network = prepared
    if network is None:
        return requested, effective, reason, startup_seconds, preflight
    if (policy_compile_effective == 'inductor'
            and policy_compile_preflight):
        return (requested, 'metal',
            'validated FP32 M5 Pro training-only Metal forward/backward; '
            'policy fullgraph preflight passed; forward_eval unchanged',
            startup_seconds, True)

    _restore_mingru_train_scan(network)
    reason = 'validated policy compiler is inactive: ' + policy_compile_reason
    if requested == 'auto':
        return requested, 'off', reason, startup_seconds, False
    raise RuntimeError(
        'torch.mingru_train_scan=metal requires an effective validated policy '
        f'compiler: {reason}')


def configure_policy_and_mingru_train_scan(args, vec, policy, device,
        rollout_device, amp_dtype, mps_host_alias_io, state):
    """Atomically configure the training scan and enclosing policy graph.

    The scan must be visible while Dynamo traces the policy. If an ``auto``
    scan fails that trace or preflight, restore the untouched class method and
    retry the portable policy compiler once. Explicit ``metal`` remains
    fail-closed. No compiled wrapper is attached before a complete preflight.
    """
    prepared = _prepare_mingru_train_scan(
        args, vec, policy, device, rollout_device, amp_dtype,
        mps_host_alias_io)
    installed_network = prepared[-1]
    fallback_reason = None

    def configure_core_policy():
        return configure_policy_compile(
            args, vec, policy, device, rollout_device, amp_dtype,
            mps_host_alias_io, state)

    first_policy_started = time.perf_counter()
    try:
        policy_result = configure_core_policy()
    except Exception as exc:
        if installed_network is None or prepared[0] != 'auto':
            _restore_mingru_train_scan(installed_network)
            raise
        failed_startup = time.perf_counter() - first_policy_started
        _restore_mingru_train_scan(installed_network)
        fallback_reason = (
            'Metal policy compile/preflight failed: '
            f'{type(exc).__name__}: {exc}; portable policy retried')
        policy_result = configure_core_policy()
        policy_result = (
            *policy_result[:3],
            failed_startup + policy_result[3],
            policy_result[4],
        )
    else:
        if (installed_network is not None
                and policy_result[1] != 'inductor'
                and prepared[0] == 'auto'):
            _restore_mingru_train_scan(installed_network)
            fallback_reason = (
                'Metal policy compile/preflight did not promote: '
                f'{policy_result[2]}; portable policy retried')
            failed_startup = policy_result[3]
            policy_result = configure_core_policy()
            policy_result = (
                *policy_result[:3],
                failed_startup + policy_result[3],
                policy_result[4],
            )

    if fallback_reason is None:
        scan_result = _finalize_mingru_train_scan(
            prepared, policy_result[1], policy_result[4], policy_result[2])
    else:
        scan_result = (
            prepared[0], 'off', fallback_reason, prepared[3], False)
    return policy_result, scan_result


def _preflight_policy_compile(compiled_eval, compiled_train, policy, state,
        device, total_agents, horizon, minibatch_size, obs_size,
        amp_dtype=None):
    """Materialize eval, train, and backward graphs before promotion."""
    device = torch.device(device)
    segments = int(minibatch_size) // int(horizon)
    eval_observations = torch.zeros(
        int(total_agents), int(obs_size), dtype=torch.float32, device=device)
    train_observations = torch.zeros(
        segments, int(horizon), int(obs_size),
        dtype=torch.float32, device=device)
    preflight_state = state
    input_state_tensors = _state_tensors(preflight_state)
    expected_policy_dtype = amp_dtype or torch.float32
    if any(
            value.is_floating_point()
            and value.dtype != expected_policy_dtype
            for value in input_state_tensors):
        raise RuntimeError(
            'compiled policy preflight recurrent-state dtype does not match '
            f'the requested {expected_policy_dtype} execution contract')

    def autocast_context():
        if amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=device.type, dtype=amp_dtype)

    original_training = policy.training
    policy.zero_grad(set_to_none=True)
    synchronize(device)
    try:
        policy.eval()
        with torch.no_grad():
            eval_state = preflight_state
            # Materialize and exercise the persistent recurrent-state path,
            # not only its first call. BF16 state is fed back for a complete
            # rollout horizon so a step-two dtype/shape specialization cannot
            # escape the guarded setup phase.
            for _ in range(int(horizon)):
                with autocast_context():
                    eval_logits, eval_values, eval_state = compiled_eval(
                        eval_observations, eval_state)
        policy.train()
        with autocast_context():
            train_logits, train_values = compiled_train(train_observations)
        # Force AOTAutograd/Inductor to materialize the backward graph too.
        (train_logits.float().sum() + train_values.float().sum()).backward()
        synchronize(device)
        output_state_tensors = _state_tensors(eval_state)
        state_contract = (
            len(input_state_tensors) == len(output_state_tensors)
            and all(
                not original.is_floating_point()
                or original.dtype == expected_policy_dtype
                for original in input_state_tensors
            )
            and all(
                output.shape == original.shape
                and output.device == original.device
                and output.dtype == original.dtype
                for original, output in zip(
                    input_state_tensors, output_state_tensors)
            )
        )
        parameter_contract = all(
            parameter.dtype == torch.float32
            and parameter.grad is not None
            and parameter.grad.dtype == torch.float32
            for parameter in policy.parameters()
        )
        output_contract = all(
            value.dtype == expected_policy_dtype
            for value in (
                eval_logits, eval_values, train_logits, train_values)
        )
        if not (
                state_contract
                and parameter_contract
                and output_contract
                and torch.isfinite(eval_logits).all()
                and torch.isfinite(eval_values).all()
                and all(torch.isfinite(value).all()
                    for value in _state_tensors(eval_state))
                and torch.isfinite(train_logits).all()
                and torch.isfinite(train_values).all()
                and all(parameter.grad is not None
                    and torch.isfinite(parameter.grad).all()
                    for parameter in policy.parameters())):
            raise RuntimeError(
                'compiled policy preflight violated its recurrent-state or '
                'FP32-gradient contract, or produced non-finite values')
    finally:
        policy.zero_grad(set_to_none=True)
        policy.train(original_training)


def _state_tensors(value):
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(
            tensor for item in value for tensor in _state_tensors(item))
    if isinstance(value, dict):
        return tuple(
            tensor for item in value.values() for tensor in _state_tensors(item))
    return ()


def configure_policy_compile(args, vec, policy, device, rollout_device,
        amp_dtype, mps_host_alias_io, state):
    """Apply the validated policy compiler without replacing Parameters.

    Returns ``(requested, effective, reason, startup_seconds, preflight)``.
    ``auto`` safely remains eager outside the measured shape; explicit
    ``inductor`` fails closed instead of pretending an unvalidated graph is
    optimized or equivalent.
    """
    requested = str(
        args.get('torch', {}).get('compile_policy', 'off')).strip().lower()
    if requested not in ('off', 'auto', 'inductor'):
        raise ValueError(
            'torch.compile_policy must be off, auto, or inductor')
    if requested == 'off':
        return requested, 'off', 'disabled by configuration', 0.0, False

    attempt_started = time.perf_counter()
    mismatches = _policy_compile_mismatches(
        args, vec, policy, device, rollout_device, amp_dtype,
        mps_host_alias_io)
    if mismatches:
        reason = '; '.join(mismatches)
        if requested == 'auto':
            return requested, 'off', reason, (
                time.perf_counter() - attempt_started), False
        raise ValueError(
            'torch.compile_policy=inductor requires the validated '
            f'Breakout/MPS configuration: {reason}')

    if not hasattr(torch, 'compile'):
        if requested == 'auto':
            return requested, 'off', 'torch.compile is unavailable', (
                time.perf_counter() - attempt_started), False
        raise RuntimeError('torch.compile is unavailable in this Torch build')

    compile_options = {'layout_optimization': False}
    try:
        # Build both wrappers before mutating the module so a setup failure
        # cannot leave a half-compiled policy. Bound methods keep the same
        # Parameter objects, state_dict keys, and optimizer ownership.
        original_eval = policy.forward_eval
        original_train = policy.forward
        compiled_eval = torch.compile(
            original_eval,
            backend='inductor',
            fullgraph=True,
            dynamic=False,
            options=compile_options,
        )
        compiled_train = torch.compile(
            original_train,
            backend='inductor',
            fullgraph=True,
            dynamic=False,
            options=compile_options,
        )
        wrappers = (
            ('forward_eval', compiled_eval, original_eval),
            ('forward', compiled_train, original_train),
        )
        for name, compiled, original in wrappers:
            if (compiled is original
                    or getattr(compiled, '_torchdynamo_orig_callable', None)
                        is not original):
                raise RuntimeError(
                    f'torch.compile did not produce a Dynamo {name} wrapper')
        _preflight_policy_compile(
            compiled_eval,
            compiled_train,
            policy,
            state,
            device,
            vec.total_agents,
            args['train']['horizon'],
            args['train']['minibatch_size'],
            vec.obs_size,
            amp_dtype,
        )
    except Exception as exc:
        startup_seconds = time.perf_counter() - attempt_started
        if requested == 'auto':
            return requested, 'off', (
                'compile/preflight failed: '
                f'{type(exc).__name__}: {exc}'), startup_seconds, False
        raise RuntimeError(
            'failed to configure the validated MPS Inductor policy graph') \
            from exc

    policy.forward_eval = compiled_eval
    policy.forward = compiled_train
    startup_seconds = time.perf_counter() - attempt_started
    precision_contract = (
        'Breakout execution-contract verified experimental BF16 autocast'
        if amp_dtype is torch.bfloat16
        else 'validated Breakout FP32')
    return requested, 'inductor', (
        f'{precision_contract} direct-MPS graph; '
        'Dynamo wrappers verified; fullgraph eval/train/backward preflight; '
        'layout_optimization=False'), startup_seconds, True


def _ppo_train_outputs(policy_forward, observations, actions, old_logprobs,
        old_values, returns, advantages, priority, clip_coef, vf_clip_coef,
        vf_coef, ent_coef):
    """Run the supplied-action PPO math in its production operation order.

    The coefficients are callable inputs rather than benchmark constants. A
    trainer therefore materializes the graph with its own sweep/config values,
    while the fixed input shapes retain the validated fullgraph specialization.
    """
    logits, newvalue = policy_forward(observations)
    logits = logits.float()
    newvalue = newvalue.float()
    _, newlogprob, entropy = sample_logits(logits, action=actions)
    newlogprob = newlogprob.reshape(old_logprobs.shape)
    logratio = newlogprob - old_logprobs
    ratio = logratio.exp()

    normalized_advantages = priority * (
        advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss1 = -normalized_advantages * ratio
    pg_loss2 = -normalized_advantages * torch.clamp(
        ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    newvalue = newvalue.view(returns.shape)
    v_clipped = old_values + torch.clamp(
        newvalue - old_values, -vf_clip_coef, vf_clip_coef)
    v_loss_unclipped = (newvalue - returns) ** 2
    v_loss_clipped = (v_clipped - returns) ** 2
    v_loss = 0.5*torch.max(v_loss_unclipped, v_loss_clipped).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss + vf_coef*v_loss - ent_coef*entropy_loss
    # Raw policy/statistic outputs remain explicit graph outputs. Besides
    # preserving the validated boundary, this lets setup verify the fused graph
    # against the already-promoted compiled-policy-only implementation.
    return (
        logits, newvalue, newlogprob, entropy, logratio, ratio,
        pg_loss, v_loss, entropy_loss, loss)


def _preflight_ppo_compile(compiled_ppo, policy_forward, policy, device,
        horizon, minibatch_size, obs_size, coefficients, max_grad_norm):
    """Materialize and compare fused PPO forward/backward without mutation."""
    device = torch.device(device)
    segments = int(minibatch_size) // int(horizon)
    observations = torch.zeros(
        segments, int(horizon), int(obs_size),
        dtype=torch.float32, device=device)
    actions = torch.arange(
        int(minibatch_size), dtype=torch.int64, device=device
    ).remainder(3).reshape(segments, int(horizon), 1).float()
    old_logprobs = torch.zeros(
        segments, int(horizon), dtype=torch.float32, device=device)
    old_values = torch.zeros_like(old_logprobs)
    returns = torch.full_like(old_logprobs, 0.25)
    advantages = torch.linspace(
        -1.0, 1.0, int(minibatch_size),
        dtype=torch.float32, device=device).reshape(segments, int(horizon))
    priority = torch.linspace(
        0.75, 1.25, segments,
        dtype=torch.float32, device=device).unsqueeze(1)
    inputs = (
        observations, actions, old_logprobs, old_values,
        returns, advantages, priority, *coefficients)

    original_training = policy.training
    parameter_ids = tuple(id(parameter) for parameter in policy.parameters())
    parameter_values = tuple(
        parameter.detach().clone() for parameter in policy.parameters())
    cpu_rng = torch.random.get_rng_state().clone()
    accelerator_rng = (
        torch.mps.get_rng_state().clone()
        if device.type == 'mps' else None)
    policy.zero_grad(set_to_none=True)
    synchronize(device)
    try:
        policy.train()
        reference = _ppo_train_outputs(policy_forward, *inputs)
        reference[-1].backward()
        synchronize(device)
        reference = tuple(value.detach().clone() for value in reference)
        reference_grads = tuple(
            parameter.grad.detach().clone()
            for parameter in policy.parameters())
        reference_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), max_grad_norm).detach().clone()
        reference_clipped = tuple(
            parameter.grad.detach().clone()
            for parameter in policy.parameters())
        policy.zero_grad(set_to_none=True)

        candidate = compiled_ppo(*inputs)
        candidate[-1].backward()
        synchronize(device)
        candidate = tuple(value.detach().clone() for value in candidate)
        candidate_grads = tuple(
            parameter.grad.detach().clone()
            for parameter in policy.parameters())
        candidate_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), max_grad_norm).detach().clone()
        candidate_clipped = tuple(
            parameter.grad.detach().clone()
            for parameter in policy.parameters())
        synchronize(device)

        output_contract = (
            len(reference) == len(candidate)
            and all(
                expected.shape == actual.shape
                and expected.dtype == actual.dtype
                and torch.isfinite(expected).all()
                and torch.isfinite(actual).all()
                and torch.allclose(
                    expected, actual, rtol=2e-5, atol=2e-6)
                for expected, actual in zip(reference, candidate))
        )
        gradient_contract = (
            len(reference_grads) == len(candidate_grads)
            and all(
                torch.isfinite(expected).all()
                and torch.isfinite(actual).all()
                and torch.allclose(
                    expected, actual, rtol=5e-5, atol=5e-6)
                for expected, actual in zip(
                    reference_grads, candidate_grads))
            and torch.allclose(
                reference_norm, candidate_norm, rtol=5e-5, atol=5e-6)
            and all(torch.allclose(
                    expected, actual, rtol=5e-5, atol=5e-6)
                for expected, actual in zip(
                    reference_clipped, candidate_clipped))
        )
        invariant_contract = (
            tuple(id(parameter) for parameter in policy.parameters())
                == parameter_ids
            and all(torch.equal(before, after)
                for before, after in zip(
                    parameter_values, policy.parameters()))
            and torch.equal(torch.random.get_rng_state(), cpu_rng)
            and (accelerator_rng is None or torch.equal(
                torch.mps.get_rng_state(), accelerator_rng))
        )
        if not (output_contract and gradient_contract and invariant_contract):
            raise RuntimeError(
                'compiled PPO preflight violated output, FP32-gradient, '
                'Parameter-identity, or RNG invariants')
    finally:
        policy.zero_grad(set_to_none=True)
        policy.train(original_training)
        torch.random.set_rng_state(cpu_rng)
        if accelerator_rng is not None:
            torch.mps.set_rng_state(accelerator_rng)


def configure_ppo_compile(args, policy, device, amp_dtype,
        policy_compile_effective):
    """Optionally fuse the validated FP32 policy and supplied-action PPO loss.

    This is an additive optimization. Any setup failure retains the already
    validated compiled-policy-only path, including for an explicit core policy
    compiler request.
    """
    requested = str(
        args.get('torch', {}).get('compile_ppo', 'off')).strip().lower()
    if requested not in ('off', 'auto', 'inductor'):
        raise ValueError(
            'torch.compile_ppo must be off, auto, or inductor')
    if requested == 'off':
        return (None, requested, 'off', 'PPO compiler is disabled by configuration',
            0.0, False, False)
    if policy_compile_effective != 'inductor':
        reason = 'validated compiled-policy path is inactive'
        if requested == 'auto':
            return (None, requested, 'off', reason,
                0.0, False, False)
        raise ValueError(
            'torch.compile_ppo=inductor requires the validated FP32 '
            f'compiled-policy path: {reason}')
    if amp_dtype is not None:
        reason = 'fused PPO graph is validated only for FP32'
        if requested == 'auto':
            return (None, requested, 'off', reason,
                0.0, False, False)
        raise ValueError(
            'torch.compile_ppo=inductor requires the validated FP32 '
            f'compiled-policy path: {reason}')
    total_timesteps = int(args.get('train', {}).get('total_timesteps', 0))
    if (requested == 'auto'
            and total_timesteps
                < _VALIDATED_PPO_COMPILE_BREAK_EVEN_TIMESTEPS):
        return (None, requested, 'off',
            f'configured total_timesteps {total_timesteps} is below the '
            f'measured {_VALIDATED_PPO_COMPILE_BREAK_EVEN_TIMESTEPS}-step '
            'compiled-PPO break-even; use inductor to force cold setup',
            0.0, False, False)

    attempt_started = time.perf_counter()
    compiled_policy_forward = policy.forward
    original_forward = getattr(
        compiled_policy_forward, '_torchdynamo_orig_callable', None)
    if original_forward is None:
        reason = (
            'compiled policy forward does not expose its original callable')
        startup_seconds = time.perf_counter() - attempt_started
        if requested == 'auto':
            return (None, requested, 'off', reason,
                startup_seconds, False, False)
        raise RuntimeError(
            'failed to configure the validated FP32 compiled PPO graph: '
            + reason)

    config = args['train']
    coefficients = (
        config['clip_coef'], config['vf_clip_coef'],
        config['vf_coef'], config['ent_coef'])

    def full_ppo(*inputs):
        return _ppo_train_outputs(original_forward, *inputs)

    try:
        compiled_ppo = torch.compile(
            full_ppo,
            backend='inductor',
            fullgraph=True,
            dynamic=False,
            options={'layout_optimization': False},
        )
        if (compiled_ppo is full_ppo
                or getattr(compiled_ppo, '_torchdynamo_orig_callable', None)
                    is not full_ppo):
            raise RuntimeError(
                'torch.compile did not produce a Dynamo PPO wrapper')
        _preflight_ppo_compile(
            compiled_ppo,
            compiled_policy_forward,
            policy,
            device,
            config['horizon'],
            config['minibatch_size'],
            policy.encoder.encoder.in_features,
            coefficients,
            config['max_grad_norm'],
        )
    except Exception as exc:
        startup_seconds = time.perf_counter() - attempt_started
        if requested == 'auto':
            return (None, requested, 'off',
                f'compile/preflight failed: {type(exc).__name__}: {exc}',
                startup_seconds, False, False)
        raise RuntimeError(
            'failed to configure the validated FP32 compiled PPO graph') \
            from exc

    return (compiled_ppo, requested, 'inductor',
        'validated FP32 policy + supplied-action PPO fullgraph; '
        'trainer coefficients supplied as graph inputs; '
        'tolerance-based output/backward and exact RNG preflight passed; '
        'not bitwise trajectory parity; layout_optimization=False',
        time.perf_counter() - attempt_started, True, True)


def configure_rollout_sampler(args, vec, device, rollout_device, amp_dtype,
        mps_host_alias_io, policy_compile_requested,
        policy_compile_effective):
    """Configure the exact-RNG fused sampler only for the validated graph.

    Auto mode safely keeps ``torch.multinomial`` if the optional native bridge
    or Metal shader cannot initialize. Explicit ``inductor`` mode fails closed
    because silently dropping one part of its validated execution path would
    make the requested optimization metadata false.
    """
    fused_name = 'fused_mps_philox'
    baseline_name = 'torch_multinomial'
    requested = (
        fused_name
        if policy_compile_requested in ('auto', 'inductor')
        else baseline_name)
    if policy_compile_effective != 'inductor':
        return None, requested, baseline_name, (
            'validated compiled policy path is inactive'), 0.0

    checks = (
        (args.get('env_name') == 'breakout', 'environment is not breakout'),
        (torch.device(device).type == 'mps', 'training device is not MPS'),
        (torch.device(rollout_device).type == 'mps',
            'rollout device is not MPS'),
        (torch.device(device) == torch.device(rollout_device),
            'training and rollout devices differ'),
        (amp_dtype in (None, torch.bfloat16),
            'AMP dtype is not float32 or bfloat16'),
        (bool(mps_host_alias_io), 'MPS host aliasing is inactive'),
        (not bool(getattr(vec, 'gpu', False)),
            'native vector backend is not CPU'),
        (int(vec.total_agents) == 4096, 'total_agents is not 4096'),
        ([int(size) for size in vec.act_sizes] == [3],
            'action sizes are not [3]'),
        (int(args['train']['horizon']) == 64, 'horizon is not 64'),
    )
    mismatches = [reason for matched, reason in checks if not matched]
    if mismatches:
        reason = '; '.join(mismatches)
        if policy_compile_requested == 'inductor':
            raise RuntimeError(
                'fused MPS sampler requires the validated compiled '
                f'Breakout path: {reason}')
        return None, requested, baseline_name, reason, 0.0

    started = time.perf_counter()
    try:
        from pufferlib.mps_kernels import MPSCategoricalSampler
        sampler = MPSCategoricalSampler(4096, 3)
    except Exception as exc:
        startup_seconds = time.perf_counter() - started
        reason = f'sampler initialization failed: {type(exc).__name__}: {exc}'
        if policy_compile_requested == 'inductor':
            raise RuntimeError(
                'failed to configure the validated fused MPS sampler') \
                from exc
        return None, requested, baseline_name, reason, startup_seconds

    return sampler, requested, fused_name, (
        'validated Breakout [4096,3] float32 Philox exponential race; '
        'atomic default-generator reservation'), (
        time.perf_counter() - started)


def _map_state(value, fn):
    """Apply ``fn`` to tensors in recurrent state without changing metadata."""
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, tuple):
        items = [_map_state(item, fn) for item in value]
        if hasattr(value, '_fields'):
            return type(value)(*items)
        return type(value)(items)
    if isinstance(value, list):
        return type(value)(_map_state(item, fn) for item in value)
    if isinstance(value, dict):
        items = ((key, _map_state(item, fn)) for key, item in value.items())
        try:
            return type(value)(items)
        except TypeError:
            return {key: _map_state(item, fn) for key, item in value.items()}
    return value

class _CudaPtr:
    '''Wraps a raw CUDA pointer so torch.as_tensor can consume it via
    __cuda_array_interface__ without any copy or C++ torch dependency.'''
    def __init__(self, ptr, shape, dtype):
        self.__cuda_array_interface__ = {
            'data':    (ptr, False),
            'shape':   shape,
            'typestr': _TORCH_TO_TYPESTR[dtype],
            'version': 2,
        }

_TORCH_TO_CTYPE = {
    torch.uint8:   ctypes.c_uint8,
    torch.float32: ctypes.c_float,
}

def _cpu_tensor(ptr, shape, dtype):
    '''Zero-copy CPU tensor from a raw pointer via ctypes.'''
    ctype = _TORCH_TO_CTYPE[dtype]
    n = 1
    for s in shape:
        n *= s
    arr = (ctype * n).from_address(ptr)
    return torch.frombuffer(arr, dtype=dtype).reshape(shape)

class PuffeRL:
    def __init__(self, args, vec, policy, verbose=True,
            device=None, rollout_device=None):
        config = args['train']
        torch_config = args.get('torch', {})
        device = torch.device(device or resolve_device(
            torch_config.get('device', 'auto'), native_cuda=bool(_C.gpu)))
        rollout_device = torch.device(rollout_device or resolve_rollout_device(
            torch_config.get('rollout_device', 'auto'), device,
            vec_gpu=bool(vec.gpu), total_agents=vec.total_agents,
            mps_threshold=torch_config.get('mps_rollout_threshold', -1)))
        if vec.gpu and rollout_device.type != 'cuda':
            raise ValueError('A native CUDA vector backend requires CUDA rollouts')
        self.device = device
        self.rollout_device = rollout_device
        amp_name = str(torch_config.get('amp_dtype', 'float32')).lower()
        try:
            self.amp_dtype = {
                'float32': None,
                'float': None,
                'bfloat16': torch.bfloat16,
                'bf16': torch.bfloat16,
            }[amp_name]
        except KeyError as exc:
            if amp_name in ('float16', 'fp16'):
                raise ValueError(
                    'torch.amp_dtype=float16 is unsupported without gradient '
                    'scaling; use bfloat16 or float32') from exc
            raise ValueError(f'Unsupported torch.amp_dtype: {amp_name}') from exc
        if self.amp_dtype is not None and device.type not in ('cuda', 'mps'):
            raise ValueError(
                'torch.amp_dtype requires a CUDA or MPS training device; '
                'CPU training remains float32')
        if self.amp_dtype is not None and rollout_device != device:
            raise ValueError(
                'torch.amp_dtype requires rollout_device == device; hybrid '
                'actors use FP32 so their stored log-probabilities would not '
                'match an autocast learner')

        torch.set_float32_matmul_precision('high')
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

        self._vec = vec
        self.gpu = vec.gpu
        total_agents = vec.total_agents
        self.total_agents = total_agents
        obs_dtype = _OBS_DTYPE_MAP.get(vec.obs_dtype, torch.uint8)

        if self.gpu:
            self.vec_obs = torch.as_tensor(_CudaPtr(vec.gpu_obs_ptr,
                (total_agents, vec.obs_size), obs_dtype))
            self.vec_rewards = torch.as_tensor(_CudaPtr(vec.gpu_rewards_ptr,
                (total_agents,), torch.float32))
            self.vec_terminals = torch.as_tensor(_CudaPtr(vec.gpu_terminals_ptr,
                (total_agents,), torch.float32))
        else:
            self.vec_obs = _cpu_tensor(vec.obs_ptr,
                (total_agents, vec.obs_size), obs_dtype)
            self.vec_rewards = _cpu_tensor(vec.rewards_ptr,
                (total_agents,), torch.float32)
            self.vec_terminals = _cpu_tensor(vec.terminals_ptr,
                (total_agents,), torch.float32)

        vec.reset()
        horizon = config['horizon']
        num_atns = vec.num_atns

        self.observations = torch.zeros(horizon, total_agents, vec.obs_size,
            dtype=obs_dtype, device=rollout_device)
        self.values = torch.zeros(horizon, total_agents, device=rollout_device)
        self.logprobs = torch.zeros(horizon, total_agents, device=rollout_device)
        self.host_horizon_io = (
            not self.gpu and rollout_device.type == 'mps')
        horizon_io_device = (
            torch.device('cpu') if self.host_horizon_io else rollout_device)
        self.mps_host_alias_io = False
        self.host_observations = None
        self.host_actions = None
        alias_setting = torch_config.get('mps_host_alias', 'auto')
        if isinstance(alias_setting, bool):
            alias_requested = alias_setting
        else:
            alias_setting = str(alias_setting).lower()
            if alias_setting in ('auto', 'on', 'true', '1'):
                alias_requested = True
            elif alias_setting in ('off', 'false', '0'):
                alias_requested = False
            else:
                raise ValueError(
                    'torch.mps_host_alias must be auto, on, or off')
        if self.host_horizon_io and alias_requested:
            try:
                self.host_observations = mps_host_alias(self.observations)
                self.mps_host_alias_io = True
            except (AttributeError, RuntimeError, TypeError, ValueError):
                # The API is version- and allocator-dependent. Ordinary
                # staging remains the supported fallback on other runtimes.
                self.host_observations = None
        vec_action_device = rollout_device if self.gpu else torch.device('cpu')
        # CPU environments produce rewards/terminals on the host, so transfer
        # each completed horizon once during training. On supported Apple
        # runtimes, action storage remains on MPS and the CPU environment reads
        # its shared-buffer alias; the staged fallback stores actions on CPU.
        action_device = (
            rollout_device if self.mps_host_alias_io else horizon_io_device)
        self.actions = torch.zeros(
            horizon, total_agents, num_atns, device=action_device)
        if self.mps_host_alias_io:
            try:
                self.host_actions = mps_host_alias(self.actions)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                # Do not leave observation aliasing half-enabled if action
                # storage cannot use the same coherence protocol.
                self.mps_host_alias_io = False
                self.host_observations = None
                self.actions = torch.zeros(
                    horizon, total_agents, num_atns,
                    device=horizon_io_device)
        self.rewards = torch.zeros(
            horizon, total_agents, device=horizon_io_device)
        self.terminals = torch.zeros(
            horizon, total_agents, device=horizon_io_device)
        self.vec_actions = None
        if not self.host_horizon_io:
            self.vec_actions = torch.empty(
                total_agents, num_atns, dtype=torch.float32,
                device=vec_action_device)
        self.ratio = torch.ones(total_agents, horizon, device=device)
        self.advantages = torch.empty(total_agents, horizon, device=device)
        self.advantage_dispatch = (
            torch.empty(total_agents, dtype=torch.float32, device=device)
            if device.type == 'mps' else None
        )

        if rollout_device == device:
            self.policy = policy.to(device)
            self.rollout_policy = self.policy
        else:
            # Clone while on the rollout device (normally CPU), then move only
            # the learner. Avoid transiently holding two accelerator copies.
            policy = policy.to(rollout_device)
            self.rollout_policy = deepcopy(policy)
            self.policy = policy.to(device)
        self.state = self.rollout_policy.initial_state(
            total_agents, device=rollout_device)
        if self.amp_dtype is not None and rollout_device.type in ('cuda', 'mps'):
            self.state = _map_state(
                self.state,
                lambda value: value.to(self.amp_dtype)
                    if value.is_floating_point() else value)

        policy_compile, mingru_scan = (
            configure_policy_and_mingru_train_scan(
                args,
                vec,
                self.policy,
                device,
                rollout_device,
                self.amp_dtype,
                self.mps_host_alias_io,
                self.state,
            ))
        (self.policy_compile_requested,
         self.policy_compile_effective,
         self.policy_compile_reason,
         self.policy_compile_startup_seconds,
         self.policy_compile_preflight) = policy_compile
        (self.mingru_train_scan_requested,
         self.mingru_train_scan_effective,
         self.mingru_train_scan_reason,
         self.mingru_train_scan_startup_seconds,
         self.mingru_train_scan_preflight) = mingru_scan
        # Successful configuration cannot reach this state without both
        # verified Dynamo wrappers and the full eval/train/backward preflight.
        self.policy_compile_wrapper_verified = bool(
            self.policy_compile_effective == 'inductor'
            and self.policy_compile_preflight)
        (self.compiled_ppo,
         self.ppo_compile_requested,
         self.ppo_compile_effective,
         self.ppo_compile_reason,
         self.ppo_compile_startup_seconds,
         self.ppo_compile_preflight,
         self.ppo_compile_wrapper_verified) = configure_ppo_compile(
            args,
            self.policy,
            device,
            self.amp_dtype,
            self.policy_compile_effective,
        )
        (self.rollout_sampler,
         self.rollout_sampler_requested,
         self.rollout_sampler_effective,
         self.rollout_sampler_reason,
         self.rollout_sampler_startup_seconds) = configure_rollout_sampler(
            args,
            vec,
            device,
            rollout_device,
            self.amp_dtype,
            self.mps_host_alias_io,
            self.policy_compile_requested,
            self.policy_compile_effective,
        )
        self.optimization_startup_seconds = (
            self.mingru_train_scan_startup_seconds
            + self.policy_compile_startup_seconds
            + self.ppo_compile_startup_seconds
            + self.rollout_sampler_startup_seconds)

        self.batch_size = total_agents * horizon
        self.minibatch_segments = config['minibatch_size'] // horizon
        self.total_epochs = max(1, config['total_timesteps'] // self.batch_size)

        self.optimizer = Muon(
            self.policy.parameters(),
            lr=config['learning_rate'],
            momentum=config['beta1'],
            eps=config['eps'],
        )

        self.args = args
        self.config = config
        self.world_size = args['world_size']
        self.epoch = 0
        self.global_step = 0
        self.last_log_step = 0
        now = time.time()
        self.last_log_time = now - self.optimization_startup_seconds
        self.start_time = now - self.optimization_startup_seconds
        # Detailed accelerator events are useful for profiling, but creating
        # hundreds of them per horizon measurably perturbs normal training.
        self.profile = Profile(enabled=bool(args.get('profile', False)))
        self.verbose = verbose
        self._owns_process_group = False

        self.model_size = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        if verbose:
            pufferlib.pufferl.print_dashboard(args, self.model_size, {}, clear=True)

    @torch.no_grad()
    def _sync_rollout_policy(self):
        source_policy = pufferlib.models._unwrap_parallel_policy(self.policy)
        if self.rollout_policy is self.policy or self.rollout_policy is source_policy:
            return
        destination_policy = pufferlib.models._unwrap_parallel_policy(
            self.rollout_policy)
        source = source_policy.state_dict()
        destination = destination_policy.state_dict()
        if source.keys() != destination.keys():
            raise RuntimeError('learner and rollout policy state keys differ')
        for key, dst in destination.items():
            src = source[key]
            # Keep this synchronous. The source and destination remain alive,
            # and the CPU actor must see a complete, consistent policy.
            dst.copy_(src.detach())

    def _autocast(self, device):
        device = torch.device(device)
        if self.amp_dtype is None or device.type not in ('cuda', 'mps'):
            return nullcontext()
        return torch.autocast(device_type=device.type, dtype=self.amp_dtype)


    @property
    def uptime(self):
        return time.time() - self.start_time

    @property
    def sps(self):
        if self.global_step == self.last_log_step:
            return 0

        return (self.global_step - self.last_log_step) / (time.time() - self.last_log_time)

    def num_params(self):
        return self.model_size

    @torch.no_grad()
    def rollouts(self):
        prof = self.profile
        config = self.config
        device = self.rollout_device
        horizon = config['horizon']

        prof.set_device(device)
        self._sync_rollout_policy()
        self.rollout_policy.eval()
        if self.args.get('reset_state', True):
            self.state = _map_state(self.state, lambda value: value.zero_())
        o = self.vec_obs
        r = None
        d = None

        P = Profile
        prof.mark(0)
        if self.mps_host_alias_io:
            # Drain initialization or prior-epoch work before the first host
            # write through a shared Metal allocation.
            torch.mps.synchronize()
        for t in range(horizon):
            prof.mark(1)
            # Copy directly into persistent rollout storage. This avoids a
            # second MPS copy from a per-step observation temporary.
            if self.mps_host_alias_io:
                self.host_observations[t].copy_(o)
                # Make the host write visible before policy kernels consume it.
                torch.mps.synchronize()
            else:
                self.observations[t].copy_(o)
            if r is None:
                self.rewards[t].zero_()
                self.terminals[t].zero_()
            else:
                self.rewards[t].copy_(r)
                self.terminals[t].copy_(d)
            o_device = self.observations[t]
            with self._autocast(device):
                logits, value, state = self.rollout_policy.forward_eval(
                    o_device, self.state)
            logits = _float_policy_output(logits)
            value = value.float()
            action, logprob, _ = sample_logits(
                logits,
                compute_entropy=False,
                rollout_sampler=self.rollout_sampler)
            self.state = state
            self.actions[t].copy_(action)
            self.logprobs[t].copy_(logprob)
            self.values[t].copy_(value.flatten())
            if not self.host_horizon_io:
                self.vec_actions.copy_(
                    action.reshape(self.total_agents, -1))

            prof.mark(2)
            if self.gpu:
                self._vec.gpu_step(self.vec_actions.data_ptr())
                torch.cuda.synchronize()
            elif self.mps_host_alias_io:
                # Policy inference and the action-store kernel must complete
                # before the CPU environment reads the shared Metal buffer.
                torch.mps.synchronize()
                self._vec.cpu_step(self.host_actions[t].data_ptr())
                # The private host-alias API requires synchronization on both
                # sides of every CPU access. This closes the host-read window
                # before later training kernels consume the action horizon.
                torch.mps.synchronize()
            elif self.host_horizon_io:
                self._vec.cpu_step(self.actions[t].data_ptr())
            else:
                self._vec.cpu_step(self.vec_actions.data_ptr())

            o, r, d = self.vec_obs, self.vec_rewards, self.vec_terminals
            prof.mark(3)
            prof.elapsed(P.EVAL_GPU, 1, 2)
            prof.elapsed(P.EVAL_ENV, 2, 3)

        prof.mark(1)
        prof.elapsed(P.ROLLOUT, 0, 1)
        prof.resolve_pending()
        self.global_step += self.total_agents * horizon
        self.env_logs = self._vec.log()

    def train(self):
        prof = self.profile
        losses = defaultdict(float)
        config = self.config
        device = self.device
        prof.set_device(device)
        self.policy.train()

        b0 = config['prio_beta0']
        a = config['prio_alpha']
        clip_coef = config['clip_coef']
        vf_clip = config['vf_clip_coef']
        anneal_beta = b0 + (1 - b0)*a*self.epoch/self.total_epochs
        self.ratio[:] = 1

        learning_rate = config['learning_rate']
        if config['anneal_lr'] and self.epoch > 0:
            lr_ratio = self.epoch / self.total_epochs
            lr_min = config['learning_rate'] * config['min_lr_ratio']
            learning_rate = lr_min + 0.5*(learning_rate - lr_min) * (1 + np.cos(np.pi * lr_ratio))
            self.optimizer.param_groups[0]['lr'] = learning_rate

        # Rollouts are written time-major. Stage that contiguous allocation
        # once, then expose a strided agent-major view. Indexing the view below
        # materializes only the selected minibatch instead of a second full
        # observation horizon (6.59 GiB at 16,384x128x843 float32 values).
        obs = agent_major_rollout(self.observations, device)
        act = agent_major_rollout(self.actions, device)
        val = self.values.T.contiguous().to(device)
        lp = agent_major_rollout(self.logprobs, device)
        rew = self.rewards.T.contiguous()
        rew.clamp_(-1, 1)
        rew = rew.to(device)
        ter = self.terminals.T.contiguous().to(device)

        P = Profile
        prof.mark(0)
        num_minibatches = int(config['replay_ratio'] * self.batch_size / config['minibatch_size'])
        if num_minibatches < 1:
            raise ValueError(
                'train.replay_ratio and train.minibatch_size produce zero '
                'minibatches per epoch')
        for mb in range(num_minibatches):
            advantages = self.advantages
            advantages[:, -1].zero_()
            advantages = compute_puff_advantage(val, rew,
                ter, self.ratio, advantages, config['gamma'],
                config['gae_lambda'], config['vtrace_rho_clip'], config['vtrace_c_clip'],
                dispatch=self.advantage_dispatch)

            adv = advantages.abs().sum(axis=1)
            prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
            prio_probs = (prio_weights + 1e-6)/(prio_weights.sum() + 1e-6)
            idx = torch.multinomial(prio_probs,
                self.minibatch_segments, replacement=True)
            mb_prio = (self.total_agents*prio_probs[idx, None])**-anneal_beta

            mb_obs = obs[idx]
            mb_actions = act[idx]
            mb_logprobs = lp[idx]
            mb_values = val[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]

            prof.mark(1)
            if self.compiled_ppo is None:
                with self._autocast(device):
                    logits, newvalue = self.policy(mb_obs)
                logits = _float_policy_output(logits)
                newvalue = newvalue.float()
                _, newlogprob, entropy = sample_logits(
                    logits, action=mb_actions)
                newlogprob = newlogprob.reshape(mb_logprobs.shape)
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()
            else:
                (_, newvalue, _, _, logratio, ratio,
                 pg_loss, v_loss, entropy_loss, loss) = self.compiled_ppo(
                    mb_obs,
                    mb_actions,
                    mb_logprobs,
                    mb_values,
                    mb_returns,
                    mb_advantages,
                    mb_prio,
                    clip_coef,
                    vf_clip,
                    config['vf_coef'],
                    config['ent_coef'],
                )
            prof.mark(2)
            prof.elapsed(P.TRAIN_FORWARD, 1, 2)

            self.ratio[idx] = ratio.detach()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > config['clip_coef']).float().mean()

            if self.compiled_ppo is None:
                adv = mb_advantages
                adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(mb_returns.shape)
                v_clipped = mb_values + torch.clamp(
                    newvalue - mb_values, -vf_clip, vf_clip)
                v_loss_unclipped = (newvalue - mb_returns) ** 2
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5*torch.max(
                    v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss + config['vf_coef']*v_loss
                    - config['ent_coef']*entropy_loss)

            val[idx] = newvalue.detach().float()

            # Metrics must not retain a chain of completed autograd graphs
            # across replay minibatches. Keep them on-device until the single
            # end-of-train synchronization, but sever graph ownership here.
            losses['policy_loss'] += pg_loss.detach()
            losses['value_loss'] += v_loss.detach()
            losses['entropy'] += entropy_loss.detach()
            losses['old_approx_kl'] += old_approx_kl.detach()
            losses['approx_kl'] += approx_kl.detach()
            losses['clipfrac'] += clipfrac.detach()
            losses['importance'] += ratio.mean().detach()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config['max_grad_norm'])
            self.optimizer.step()
            self.optimizer.zero_grad()

        synchronize(device)
        prof.mark(1)
        prof.elapsed(P.TRAIN, 0, 1)
        prof.resolve_pending()

        losses = {k: v.item() / num_minibatches for k, v in losses.items()}
        y_pred = val.flatten()
        y_true = advantages.flatten() + val.flatten()
        var_y = y_true.var()
        explained_var = torch.nan if var_y == 0 else (1 - (y_true - y_pred).var() / var_y).item()
        losses['explained_variance'] = explained_var

        self.losses = losses
        self.epoch += 1

    def log(self):
        P = Profile
        perf = self.profile.read_and_reset()
        logs = {
            'SPS': self.sps * self.world_size,
            'agent_steps': self.global_step * self.world_size,
            'uptime': time.time() - self.start_time,
            'epoch': self.epoch,
            'env': dict(getattr(self, 'env_logs', {})),
            'loss': dict(getattr(self, 'losses', {})),
            'perf': {
                'rollout': perf[P.ROLLOUT],
                'eval_gpu': perf[P.EVAL_GPU],
                'eval_env': perf[P.EVAL_ENV],
                'train': perf[P.TRAIN],
                'train_misc': perf[P.TRAIN_MISC],
                'train_forward': perf[P.TRAIN_FORWARD],
            },
            'util': dict(_C.get_utilization(self.args.get('gpu_id', 0))) if self.gpu else {},
        }
        self.last_log_time = time.time()
        self.last_log_step = self.global_step
        return logs

    eval_log = log

    def save_weights(self, path):
        policy = pufferlib.models._unwrap_parallel_policy(self.policy)
        torch.save(policy.state_dict(), path)

    def load_weights(self, path):
        state_dict = torch.load(path, map_location=self.device)
        parameter_ids = tuple(id(value) for value in self.policy.parameters())
        pufferlib.models.load_compatible_state_dict(
            self.policy, state_dict, checkpoint_path=path)
        if self.rollout_policy is not self.policy:
            pufferlib.models.load_compatible_state_dict(
                self.rollout_policy, state_dict, checkpoint_path=path,
                warn=False)

        # The Impulse Wars legacy fallback replaces only the encoder module.
        # Rebind the optimizer so load_weights remains safe for callers that
        # resume training after constructing PuffeRL (eval callers also use the
        # separately loaded rollout actor above).
        if parameter_ids != tuple(id(value) for value in self.policy.parameters()):
            self.optimizer = Muon(
                self.policy.parameters(),
                lr=self.config['learning_rate'],
                momentum=self.config['beta1'],
                eps=self.config['eps'],
            )

    def render(self, env_id=0):
        self._vec.render(env_id)

    def close(self):
        self.vec_obs = None
        self.vec_rewards = None
        self.vec_terminals = None
        self._vec.close()
        if self._owns_process_group and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    @classmethod
    def create_pufferl(cls, args):
        '''Matches _C.create_pufferl(args) interface.'''
        torch_config = args.get('torch', {})
        device = resolve_device(torch_config.get('device', 'auto'),
            native_cuda=bool(_C.gpu))

        external = all(
            name in os.environ for name in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
        if external:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            cuda_index = int(os.environ['LOCAL_RANK'])
        else:
            rank = int(args.get('rank', 0))
            world_size = int(args.get('world_size', 1))
            cuda_index = int(args.get('gpu_id', 0))

        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError('invalid Torch rank/world_size configuration')
        distributed = world_size > 1
        if distributed and device.type != 'cuda':
            raise RuntimeError('Torch DDP is currently supported only on CUDA')
        if distributed and args.get('load_id') is not None:
            raise RuntimeError(
                'Torch DDP cannot download a W&B load_id independently on '
                'each rank; download it once and use load_model_path')
        if device.type == 'cuda':
            # Internal jobs select args.gpu_id; torchrun selects LOCAL_RANK.
            # A single explicit cuda:N request remains authoritative.
            if not distributed and not external and device.index is not None:
                cuda_index = device.index
            torch.cuda.set_device(cuda_index)
            device = torch.device('cuda', cuda_index)

        args['rank'] = rank
        args['world_size'] = world_size
        args['gpu_id'] = cuda_index

        args['vec']['num_buffers'] = 1
        vec = _C.create_vec(args, _C.gpu)
        rollout_device = resolve_rollout_device(
            torch_config.get('rollout_device', 'auto'), device,
            vec_gpu=bool(vec.gpu), total_agents=vec.total_agents,
            mps_threshold=torch_config.get('mps_rollout_threshold', -1))
        if rollout_device.type == 'cuda' and rollout_device.index is None:
            rollout_device = device
        if vec.gpu and rollout_device != device:
            vec.close()
            raise ValueError(
                'native CUDA vector buffers and rollout policy must use the '
                f'same device (vec={device}, rollout={rollout_device})')
        try:
            policy = load_policy(args, vec, device=rollout_device)
            trainer = cls(args, vec, policy,
                device=device, rollout_device=rollout_device)
        except Exception:
            vec.close()
            raise

        if distributed:
            owns_process_group = False
            if not torch.distributed.is_initialized():
                init_kwargs = {
                    'backend': 'nccl',
                    'rank': rank,
                    'world_size': world_size,
                }
                if not external:
                    init_method = args.get('torch_dist_init_method')
                    if not init_method:
                        trainer.close()
                        raise RuntimeError(
                            'internal Torch DDP requires a shared rendezvous')
                    init_kwargs['init_method'] = init_method
                torch.distributed.init_process_group(**init_kwargs)
                owns_process_group = True
            elif (torch.distributed.get_rank() != rank
                    or torch.distributed.get_world_size() != world_size):
                trainer.close()
                raise RuntimeError(
                    'existing Torch process group does not match requested '
                    'rank/world_size')

            learner = trainer.policy
            try:
                model = torch.nn.parallel.DistributedDataParallel(
                    learner, device_ids=[cuda_index], output_device=cuda_index)
            except Exception:
                if owns_process_group and torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
                trainer.close()
                raise
            if hasattr(learner, 'hidden_size'):
                model.hidden_size = learner.hidden_size
            model.forward_eval = learner.forward_eval
            model.initial_state = learner.initial_state
            trainer.policy = model
            if trainer.rollout_policy is learner:
                trainer.rollout_policy = model
            trainer._owns_process_group = owns_process_group

        return trainer

def compute_puff_advantage(values, rewards, terminals,
        ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip,
        dispatch=None):
    num_steps, horizon = values.shape
    if values.device.type == 'cpu':
        _C.puff_advantage_cpu(
            values.data_ptr(), rewards.data_ptr(), terminals.data_ptr(),
            ratio.data_ptr(), advantages.data_ptr(),
            num_steps, horizon,
            gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip)
        return advantages
    if values.device.type == 'cuda' and hasattr(_C, 'puff_advantage'):
        _C.puff_advantage(
            values.data_ptr(), rewards.data_ptr(), terminals.data_ptr(),
            ratio.data_ptr(), advantages.data_ptr(),
            num_steps, horizon,
            gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip)
        return advantages
    if values.device.type == 'mps' and values.dtype == torch.float32:
        from pufferlib.mps_kernels import puff_advantage
        return puff_advantage(
            values, rewards, terminals, ratio, advantages,
            gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip,
            dispatch=dispatch)

    # Portable accelerator fallback. The time dependency is sequential, but
    # each timestep operates across every agent in parallel on the device.
    with torch.no_grad():
        last = torch.zeros(num_steps, dtype=values.dtype, device=values.device)
        for t in range(horizon - 2, -1, -1):
            next_nonterminal = 1.0 - terminals[:, t + 1]
            importance = ratio[:, t]
            rho_t = importance.clamp(max=vtrace_rho_clip)
            c_t = importance.clamp(max=vtrace_c_clip)
            delta = rho_t * (
                rewards[:, t + 1]
                + gamma * values[:, t + 1] * next_nonterminal
                - values[:, t])
            last = (delta + gamma * gae_lambda * c_t
                * last * next_nonterminal)
            advantages[:, t] = last
    return advantages

class Profile:
    '''Matches pufferlib.cu profiling: accumulate ms, report seconds.'''
    ROLLOUT, EVAL_GPU, EVAL_ENV, TRAIN, TRAIN_MISC, TRAIN_FORWARD, NUM = range(7)

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.accum = [0.0] * Profile.NUM
        self.device_type = 'cpu'
        self._host_marks = {}
        self._event_marks = {}
        self._pending = []

    def set_device(self, device):
        if not self.enabled:
            return
        self.device_type = torch.device(device).type

    def _new_event(self):
        if self.device_type == 'cuda':
            return torch.cuda.Event(enable_timing=True)
        if self.device_type == 'mps':
            return torch.mps.Event(enable_timing=True)
        return None

    def mark(self, idx):
        if not self.enabled:
            return
        self._host_marks[idx] = time.perf_counter()
        event = self._new_event()
        if event is not None:
            event.record()
            self._event_marks[idx] = event

    def elapsed(self, idx, start_ev, end_ev):
        if not self.enabled:
            return
        accelerator_interval = idx in (Profile.EVAL_GPU, Profile.TRAIN_FORWARD)
        if accelerator_interval and self.device_type in ('cuda', 'mps'):
            # Defer synchronization until the whole phase is complete. This
            # keeps profiling from inserting a barrier after every forward.
            self._pending.append(
                (idx, self._event_marks[start_ev], self._event_marks[end_ev]))
        else:
            self.accum[idx] += (
                self._host_marks[end_ev] - self._host_marks[start_ev]) * 1000.0

    def resolve_pending(self):
        if not self.enabled:
            return
        for idx, start_event, end_event in self._pending:
            end_event.synchronize()
            self.accum[idx] += start_event.elapsed_time(end_event)
        self._pending.clear()

    def read_and_reset(self):
        self.resolve_pending()
        out = [v / 1000.0 for v in self.accum]
        self.accum = [0.0] * Profile.NUM
        return out

def load_policy(args, vec, device=None):
    import inspect
    import pufferlib.models
    policy_kwargs = args['policy']
    network_cls = getattr(pufferlib.models, args['torch']['network'])
    encoder_cls = getattr(pufferlib.models, args['torch']['encoder'])
    decoder_cls = getattr(pufferlib.models, args['torch']['decoder'])

    network = network_cls(**policy_kwargs)
    encoder_parameters = inspect.signature(encoder_cls.__init__).parameters
    encoder_kwargs = {
        key: value for key, value in policy_kwargs.items()
        if key in encoder_parameters and key not in ('self', 'obs_size', 'env')
    }
    encoder = encoder_cls(vec.obs_size, **encoder_kwargs)
    decoder = decoder_cls(vec.act_sizes, policy_kwargs['hidden_size'])
    policy = pufferlib.models.Policy(encoder, decoder, network)

    device = torch.device(device or resolve_device(
        args.get('torch', {}).get('device', 'auto'),
        native_cuda=bool(_C.gpu)))
    policy = policy.to(device)

    load_id = args['load_id']
    if load_id is not None:
        if args['wandb']:
            import wandb
            artifact = wandb.use_artifact(f'{load_id}:latest')
            data_dir = artifact.download()
            path = f'{data_dir}/{max(os.listdir(data_dir))}'
        else:
            raise ValueError('load_id requires --wandb')

        state_dict = torch.load(path, map_location=device)
        pufferlib.models.load_compatible_state_dict(
            policy, state_dict, checkpoint_path=path)

    load_path = args['load_model_path']
    if load_path == 'latest':
        pattern = os.path.join(args['checkpoint_dir'], args['env_name'], '**', '*.bin')
        candidates = glob.glob(pattern, recursive=True)
        load_path = max(candidates, key=os.path.getctime)

    if load_path is not None:
        state_dict = torch.load(load_path, map_location=device)
        pufferlib.models.load_compatible_state_dict(
            policy, state_dict, checkpoint_path=load_path)

    return policy
