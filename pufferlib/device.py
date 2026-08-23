"""Torch accelerator selection shared by PufferLib's portable backends."""

import torch


def mps_available():
    """Return whether this PyTorch build can execute on Apple Metal."""
    return bool(
        hasattr(torch.backends, 'mps')
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def resolve_device(requested='auto', native_cuda=False):
    """Resolve and validate a policy/training device.

    ``native_cuda`` means the compiled vector backend exposes CUDA buffers, so
    CUDA takes precedence in auto mode. Otherwise Apple Silicon prefers MPS.
    """
    requested = str(requested or 'auto').lower()
    if requested == 'auto':
        if native_cuda and torch.cuda.is_available():
            return torch.device('cuda')
        if mps_available():
            return torch.device('mps')
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    device = torch.device(requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    if device.type == 'mps' and not mps_available():
        raise RuntimeError('MPS was requested but is not available')
    if device.type not in ('cpu', 'cuda', 'mps'):
        raise ValueError(f'Unsupported Torch device: {device}')
    return device


def resolve_rollout_device(requested, training_device, vec_gpu=False,
        total_agents=None, mps_threshold=-1):
    """Choose where timestep-at-a-time inference and rollout storage live.

    CPU native environments pair best with CPU rollout inference: a completed
    horizon is copied to the accelerator once instead of copying observations
    and actions across the device boundary at every environment step.
    """
    requested = str(requested or 'auto').lower()
    if requested == 'auto':
        if vec_gpu:
            return torch.device('cuda')
        training_device = torch.device(training_device)
        threshold = int(mps_threshold)
        if (training_device.type == 'mps' and threshold > 0
                and total_agents is not None
                and int(total_agents) >= threshold):
            return training_device
        return torch.device('cpu')
    return resolve_device(requested, native_cuda=vec_gpu)


def synchronize(device):
    """Wait for queued work on an accelerator, if any."""
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def mps_host_alias(tensor):
    """Return a CPU tensor aliasing an MPS tensor's shared Metal buffer.

    PyTorch 2.13 exposes this advanced interop primitive for Apple unified
    memory. Callers remain responsible for synchronizing MPS before and after
    host access. The trainer capability-checks it and falls back to ordinary
    copies on runtimes where the private API is absent or rejects the storage.
    """
    if tensor.device.type != 'mps' or not tensor.is_contiguous():
        raise ValueError('an MPS host alias requires a contiguous MPS tensor')
    alias_storage_fn = getattr(torch.mps, '_host_alias_storage', None)
    if alias_storage_fn is None:
        raise RuntimeError('this PyTorch runtime does not expose MPS host aliases')
    storage = alias_storage_fn(tensor.untyped_storage())
    return torch.empty(0, dtype=tensor.dtype, device='cpu').set_(
        storage, tensor.storage_offset(), tensor.size(), tensor.stride())


def agent_major_rollout(tensor, device):
    """Stage a time-major rollout once and expose an agent-major view.

    Rollouts are written contiguously as ``[horizon, agents, ...]``, while PPO
    samples complete agent trajectories as ``[agents, horizon, ...]``. Moving
    the time-major tensor before transposing avoids materializing a second
    full-horizon tensor when both devices are the same, and avoids a redundant
    source-device contiguous copy when they differ. Advanced indexing this
    strided view still returns a compact contiguous minibatch.
    """
    if tensor.ndim < 2:
        raise ValueError('a rollout tensor must have horizon and agent axes')
    return tensor.to(torch.device(device)).transpose(0, 1)
