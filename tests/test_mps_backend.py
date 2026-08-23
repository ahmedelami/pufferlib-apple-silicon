"""Correctness coverage for the portable Torch/MPS training path.

The device-policy tests run anywhere Torch is installed. Tests which exercise
Metal or the C extension skip independently so a CPU-only source checkout still
gets useful coverage instead of skipping this whole module.
"""

import ctypes
from copy import deepcopy
import math

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from pufferlib import device as device_utils
from pufferlib import models


def _require_mps():
    if not device_utils.mps_available():
        pytest.skip("MPS is not available in this PyTorch runtime")


def _require_mps_host_alias():
    _require_mps()
    if not hasattr(torch.mps, "_host_alias_storage"):
        pytest.skip("this PyTorch MPS runtime has no private host-alias API")


def _torch_pufferl():
    """Import the Torch backend only for tests which need the compiled ABI."""
    try:
        from pufferlib import _C
    except Exception as exc:
        pytest.skip(f"pufferlib._C is not built: {exc}")

    if getattr(_C, "precision_bytes", None) != 4:
        pytest.skip("the Torch backend requires a float32 pufferlib._C build")

    try:
        from pufferlib import torch_pufferl
    except Exception as exc:
        pytest.skip(f"pufferlib.torch_pufferl is unavailable: {exc}")
    return torch_pufferl


def _churn_mps_allocator():
    """Queue work in several allocator bins and release the Python owners."""
    live = []
    for repeat in range(3):
        for size in (257, 1_031, 4_099, 16_411, 32_771):
            tensor = torch.empty(size + repeat * 13, device="mps")
            tensor.fill_(repeat + size % 17)
            live.append(tensor)
        del live[:]


@pytest.mark.parametrize(
    ("native_cuda", "has_mps", "has_cuda", "expected"),
    [
        (True, True, True, "cuda"),
        (False, True, True, "mps"),
        (False, False, True, "cuda"),
        (True, False, False, "cpu"),
    ],
)
def test_resolve_device_auto_priority(
        monkeypatch, native_cuda, has_mps, has_cuda, expected):
    monkeypatch.setattr(device_utils, "mps_available", lambda: has_mps)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: has_cuda)

    resolved = device_utils.resolve_device("auto", native_cuda=native_cuda)

    assert resolved == torch.device(expected)


def test_explicit_device_validation_and_rollout_policy(monkeypatch):
    monkeypatch.setattr(device_utils, "mps_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert device_utils.resolve_device("cpu") == torch.device("cpu")
    with pytest.raises(RuntimeError, match="MPS was requested"):
        device_utils.resolve_device("mps")
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        device_utils.resolve_device("cuda")
    with pytest.raises(ValueError, match="Unsupported Torch device"):
        device_utils.resolve_device("meta")

    # Native CPU environments stage a completed horizon to the training
    # accelerator instead of crossing the device boundary on every step.
    assert device_utils.resolve_rollout_device(
        "auto", torch.device("mps"), vec_gpu=False) == torch.device("cpu")
    assert device_utils.resolve_rollout_device(
        "auto", torch.device("mps"), vec_gpu=False,
        total_agents=4_096) == torch.device("cpu")
    assert device_utils.resolve_rollout_device(
        "auto", torch.device("mps"), vec_gpu=False,
        total_agents=4_095, mps_threshold=4_096) == torch.device("cpu")
    assert device_utils.resolve_rollout_device(
        "auto", torch.device("mps"), vec_gpu=False,
        total_agents=4_096, mps_threshold=4_096) == torch.device("mps")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert device_utils.resolve_rollout_device(
        "auto", torch.device("cuda"), vec_gpu=True) == torch.device("cuda")


def test_agent_major_rollout_is_zero_copy_and_gathers_contiguous_minibatches():
    horizon, agents, obs_size = 5, 7, 11
    rollout = torch.arange(
        horizon * agents * obs_size, dtype=torch.float32,
    ).reshape(horizon, agents, obs_size)

    staged = device_utils.agent_major_rollout(rollout, "cpu")

    assert staged.shape == (agents, horizon, obs_size)
    assert not staged.is_contiguous()
    assert staged.untyped_storage().data_ptr() == rollout.untyped_storage().data_ptr()

    indices = torch.tensor([6, 1, 6, 0])
    minibatch = staged[indices]
    expected = rollout[:, indices].transpose(0, 1).contiguous()
    assert minibatch.is_contiguous()
    torch.testing.assert_close(minibatch, expected, rtol=0, atol=0)


def test_agent_major_rollout_mps_storage_staging_and_strided_indexing():
    _require_mps()
    horizon, agents, obs_size = 5, 7, 11
    cpu_rollout = torch.arange(
        horizon * agents * obs_size, dtype=torch.float32,
    ).reshape(horizon, agents, obs_size)
    mps_rollout = cpu_rollout.to("mps")
    torch.mps.synchronize()

    allocated_before = torch.mps.current_allocated_memory()
    staged = device_utils.agent_major_rollout(mps_rollout, "mps")
    torch.mps.synchronize()
    allocated_after = torch.mps.current_allocated_memory()

    assert staged.untyped_storage().data_ptr() == mps_rollout.untyped_storage().data_ptr()
    assert allocated_after == allocated_before
    assert not staged.is_contiguous()

    indices = torch.tensor([6, 1, 6, 0], device="mps")
    legacy = mps_rollout.transpose(0, 1).contiguous()
    minibatch = staged[indices]
    assert minibatch.is_contiguous()
    torch.testing.assert_close(minibatch.cpu(), legacy[indices].cpu(), rtol=0, atol=0)

    staged_from_cpu = device_utils.agent_major_rollout(cpu_rollout, "mps")
    assert not staged_from_cpu.is_contiguous()
    torch.testing.assert_close(
        staged_from_cpu.cpu(), cpu_rollout.transpose(0, 1), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", (torch.float32, torch.uint8))
def test_mps_host_alias_is_bidirectionally_coherent_under_allocator_churn(dtype):
    _require_mps_host_alias()
    source = torch.zeros(4_099, dtype=dtype, device="mps")
    alias = device_utils.mps_host_alias(source)
    assert alias.device.type == "cpu"
    assert alias.dtype == dtype
    assert alias.shape == source.shape

    for repeat in range(8):
        _churn_mps_allocator()
        torch.mps.synchronize()
        host_value = repeat + 3
        alias.fill_(host_value)
        torch.mps.synchronize()
        torch.testing.assert_close(
            source.cpu(), torch.full_like(alias, host_value), rtol=0, atol=0
        )

        source.fill_(host_value + 11)
        torch.mps.synchronize()
        torch.testing.assert_close(
            alias, torch.full_like(alias, host_value + 11), rtol=0, atol=0
        )
        torch.mps.synchronize()


def test_strided_rollout_minibatch_matches_legacy_training_layout():
    generator = torch.Generator().manual_seed(9187)
    horizon, agents, obs_size = 6, 9, 13
    rollout = torch.randn(
        horizon, agents, obs_size, generator=generator,
        dtype=torch.float32,
    )
    indices = torch.tensor([8, 2, 8, 4])

    legacy = rollout.transpose(0, 1).contiguous()[indices]
    strided = device_utils.agent_major_rollout(rollout, "cpu")[indices]
    assert strided.is_contiguous()
    torch.testing.assert_close(strided, legacy, rtol=0, atol=0)

    with torch.random.fork_rng():
        torch.manual_seed(7721)
        reference = models.Policy(
            models.DefaultEncoder(obs_size, hidden_size=16),
            models.DefaultDecoder([5, 3], hidden_size=16),
            models.MinGRU(hidden_size=16, num_layers=1),
        )
    candidate = deepcopy(reference)

    reference_logits, reference_values = reference(legacy)
    candidate_logits, candidate_values = candidate(strided)
    reference_loss = reference_values.square().mean() + sum(
        logits.square().mean() for logits in reference_logits)
    candidate_loss = candidate_values.square().mean() + sum(
        logits.square().mean() for logits in candidate_logits)
    reference_loss.backward()
    candidate_loss.backward()

    for actual, expected in zip(candidate_logits, reference_logits):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(candidate_values, reference_values, rtol=0, atol=0)
    torch.testing.assert_close(candidate_loss, reference_loss, rtol=0, atol=0)
    for (actual_name, actual), (expected_name, expected) in zip(
            candidate.named_parameters(), reference.named_parameters()):
        assert actual_name == expected_name
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)


def _python_advantage(values, rewards, terminals, ratio,
        gamma, gae_lambda, rho_clip, c_clip):
    result = torch.zeros_like(values)
    for row in range(values.shape[0]):
        last = 0.0
        for t in range(values.shape[1] - 2, -1, -1):
            next_nonterminal = 1.0 - float(terminals[row, t + 1])
            importance = float(ratio[row, t])
            rho_t = min(importance, rho_clip)
            c_t = min(importance, c_clip)
            delta = rho_t * (
                float(rewards[row, t + 1])
                + gamma * float(values[row, t + 1]) * next_nonterminal
                - float(values[row, t])
            )
            last = (
                delta
                + gamma * gae_lambda * c_t * last * next_nonterminal
            )
            result[row, t] = last
    return result


def test_portable_mps_advantage_matches_cpu_extension_and_reference():
    _require_mps()
    torch_pufferl = _torch_pufferl()

    generator = torch.Generator().manual_seed(20260822)
    shape = (7, 13)
    values = torch.randn(shape, generator=generator, dtype=torch.float32)
    rewards = torch.randn(shape, generator=generator, dtype=torch.float32)
    terminals = torch.zeros(shape, dtype=torch.float32)
    terminals[0, 4] = 1
    terminals[2, 8] = 1
    terminals[5, 1] = 1
    ratio = 0.05 + 2.5 * torch.rand(
        shape, generator=generator, dtype=torch.float32)
    gamma, gae_lambda = 0.973, 0.87
    rho_clip, c_clip = 1.15, 0.83

    reference = _python_advantage(
        values, rewards, terminals, ratio,
        gamma, gae_lambda, rho_clip, c_clip,
    )
    cpu_result = torch_pufferl.compute_puff_advantage(
        values.clone(), rewards.clone(), terminals.clone(), ratio.clone(),
        torch.zeros_like(values),
        gamma, gae_lambda, rho_clip, c_clip,
    )

    _churn_mps_allocator()
    mps_result = torch_pufferl.compute_puff_advantage(
        values.to("mps"), rewards.to("mps"), terminals.to("mps"),
        ratio.to("mps"), torch.zeros_like(values, device="mps"),
        gamma, gae_lambda, rho_clip, c_clip,
    )
    _churn_mps_allocator()
    mps_result = mps_result.cpu()

    torch.testing.assert_close(cpu_result, reference, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(mps_result, cpu_result, rtol=2e-5, atol=2e-5)
    # The final bootstrap slot is intentionally not populated by the kernel.
    assert torch.count_nonzero(mps_result[:, -1]) == 0


def test_mps_advantage_matches_cpu_at_training_shape_under_allocator_churn():
    _require_mps()
    torch_pufferl = _torch_pufferl()

    generator = torch.Generator().manual_seed(196883)
    shape = (4_096, 64)
    values = torch.randn(shape, generator=generator, dtype=torch.float32)
    rewards = torch.randn(shape, generator=generator, dtype=torch.float32)
    terminals = (torch.rand(shape, generator=generator) < 0.015).float()
    ratio = 0.1 + 1.8 * torch.rand(shape, generator=generator)
    cpu_result = torch_pufferl.compute_puff_advantage(
        values, rewards, terminals, ratio, torch.zeros_like(values),
        0.995, 0.90, 1.0, 1.0,
    )

    mps_inputs = tuple(
        tensor.to("mps") for tensor in (values, rewards, terminals, ratio)
    )
    dispatch = torch.empty(shape[0], dtype=torch.float32, device="mps")
    for _ in range(3):
        _churn_mps_allocator()
        actual = torch_pufferl.compute_puff_advantage(
            *mps_inputs, torch.zeros(shape, dtype=torch.float32, device="mps"),
            0.995, 0.90, 1.0, 1.0, dispatch=dispatch,
        )
        _churn_mps_allocator()
        torch.testing.assert_close(
            actual.cpu(), cpu_result, rtol=2e-5, atol=2e-5,
        )


@pytest.mark.cuda
def test_cuda_horizon64_advantage_matches_canonical_nonunit_ratio():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch_pufferl = _torch_pufferl()
    from pufferlib import _C
    if not hasattr(_C, "puff_advantage"):
        pytest.skip("the compiled extension has no native CUDA advantage kernel")

    generator = torch.Generator().manual_seed(88217)
    shape = (17, 64)
    values = torch.randn(shape, generator=generator)
    rewards = torch.randn(shape, generator=generator)
    terminals = (torch.rand(shape, generator=generator) < 0.07).float()
    ratio = 0.05 + 2.4 * torch.rand(shape, generator=generator)
    expected = _python_advantage(
        values, rewards, terminals, ratio, 0.987, 0.91, 1.2, 0.8)
    actual = torch_pufferl.compute_puff_advantage(
        values.cuda(), rewards.cuda(), terminals.cuda(), ratio.cuda(),
        torch.zeros(shape, device="cuda"),
        0.987, 0.91, 1.2, 0.8,
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-5)


def test_allocator_perturbed_mps_matmul_matches_cpu():
    _require_mps()
    generator = torch.Generator().manual_seed(6174)
    left = torch.randn((37, 113), generator=generator, dtype=torch.float32)
    right = torch.randn((113, 61), generator=generator, dtype=torch.float32)
    expected = left @ right

    def queue_matmul_from_temporaries():
        # The inputs lose their Python owners when this function returns. The
        # MPS allocator must not recycle them while the matmul is still queued.
        return left.to("mps") @ right.to("mps")

    for _ in range(3):
        actual = queue_matmul_from_temporaries()
        _churn_mps_allocator()
        torch.testing.assert_close(
            actual.cpu(), expected, rtol=3e-4, atol=2e-4)


def _make_policy():
    return models.Policy(
        models.DefaultEncoder(obs_size=19, hidden_size=32),
        models.DefaultDecoder(nvec=[5, 3], hidden_size=32),
        models.MLP(hidden_size=32, num_layers=2),
    )


def _policy_loss(logits, values, logit_targets, value_target):
    loss = (values - value_target).square().mean()
    for output, target in zip(logits, logit_targets):
        loss = loss + (output - target).square().mean()
    return loss


def test_allocator_perturbed_mps_model_forward_backward_matches_cpu():
    _require_mps()
    with torch.random.fork_rng():
        torch.manual_seed(99173)
        cpu_policy = _make_policy()
    mps_policy = deepcopy(cpu_policy).to("mps")

    generator = torch.Generator().manual_seed(7103)
    observations = torch.randn(
        (6, 7, 19), generator=generator, dtype=torch.float32)
    logit_targets = [
        torch.randn((42, width), generator=generator, dtype=torch.float32)
        for width in (5, 3)
    ]
    value_target = torch.randn((6, 7), generator=generator, dtype=torch.float32)

    cpu_logits, cpu_values = cpu_policy(observations)
    cpu_loss = _policy_loss(
        cpu_logits, cpu_values, logit_targets, value_target)
    cpu_loss.backward()

    _churn_mps_allocator()
    mps_logits, mps_values = mps_policy(observations.to("mps"))
    mps_loss = _policy_loss(
        mps_logits,
        mps_values,
        [target.to("mps") for target in logit_targets],
        value_target.to("mps"),
    )
    _churn_mps_allocator()
    mps_loss.backward()
    _churn_mps_allocator()
    torch.mps.synchronize()

    for actual, expected in zip(mps_logits, cpu_logits):
        torch.testing.assert_close(
            actual.cpu(), expected.detach(), rtol=8e-4, atol=3e-4)
    torch.testing.assert_close(
        mps_values.cpu(), cpu_values.detach(), rtol=8e-4, atol=3e-4)
    torch.testing.assert_close(
        mps_loss.cpu(), cpu_loss.detach(), rtol=8e-4, atol=3e-4)

    for (cpu_name, cpu_parameter), (mps_name, mps_parameter) in zip(
            cpu_policy.named_parameters(), mps_policy.named_parameters()):
        assert cpu_name == mps_name
        assert cpu_parameter.grad is not None
        assert mps_parameter.grad is not None
        torch.testing.assert_close(
            mps_parameter.grad.cpu(), cpu_parameter.grad,
            rtol=3e-3, atol=5e-4,
            msg=lambda message: f"gradient mismatch for {cpu_name}: {message}",
        )


def test_state_mapping_preserves_non_floating_state_and_hybrid_amp_is_rejected():
    torch_pufferl = _torch_pufferl()
    state = {
        "hidden": (torch.ones(2, dtype=torch.float32),),
        "counter": torch.tensor([3], dtype=torch.int64),
        "mask": torch.tensor([True], dtype=torch.bool),
    }
    mapped = torch_pufferl._map_state(
        state,
        lambda value: value.to(torch.bfloat16)
            if value.is_floating_point() else value,
    )
    assert mapped["hidden"][0].dtype == torch.bfloat16
    assert mapped["counter"].dtype == torch.int64
    assert mapped["mask"].dtype == torch.bool

    vec = _FakeCPUVec()
    args = _hybrid_args()
    args["torch"]["amp_dtype"] = "bfloat16"
    policy = models.Policy(
        models.DefaultEncoder(vec.obs_size, hidden_size=16),
        models.DefaultDecoder(vec.act_sizes, hidden_size=16),
        models.MLP(hidden_size=16, num_layers=1),
    )
    with pytest.raises(ValueError, match="requires rollout_device == device"):
        torch_pufferl.PuffeRL(
            args, vec, policy, verbose=False,
            device="mps", rollout_device="cpu",
        )


class _FakeCPUVec:
    """Small raw-pointer vec implementing the subset PuffeRL consumes."""

    gpu = False
    total_agents = 8
    obs_size = 11
    num_atns = 1
    act_sizes = [4]
    obs_dtype = "FloatTensor"

    def __init__(self):
        self.observations = np.zeros(
            (self.total_agents, self.obs_size), dtype=np.float32)
        self.rewards = np.zeros(self.total_agents, dtype=np.float32)
        self.terminals = np.zeros(self.total_agents, dtype=np.float32)
        self.obs_ptr = self.observations.ctypes.data
        self.rewards_ptr = self.rewards.ctypes.data
        self.terminals_ptr = self.terminals.ctypes.data
        self.steps = 0
        self.last_actions = None
        self.action_history = []
        self.action_pointers = []
        self.observation_history = []
        self.reward_history = []
        self.terminal_history = []
        self.closed = False

    def reset(self):
        self.steps = 0
        self.action_history.clear()
        self.action_pointers.clear()
        self.observation_history.clear()
        self.reward_history.clear()
        self.terminal_history.clear()
        self.observations[:] = np.linspace(
            -0.5, 0.5, self.observations.size, dtype=np.float32,
        ).reshape(self.observations.shape)
        self.observation_history.append(self.observations.copy())
        self.rewards.fill(0)
        self.terminals.fill(0)

    def cpu_step(self, actions_ptr):
        self.action_pointers.append(actions_ptr)
        raw_type = ctypes.c_float * (self.total_agents * self.num_atns)
        actions = np.ctypeslib.as_array(
            raw_type.from_address(actions_ptr),
        ).reshape(self.total_agents, self.num_atns)
        self.last_actions = actions.copy()
        self.action_history.append(self.last_actions)
        self.steps += 1

        assert np.isfinite(actions).all()
        assert np.equal(actions, np.floor(actions)).all()
        assert ((0 <= actions) & (actions < self.act_sizes[0])).all()

        self.rewards[:] = (actions[:, 0] == (self.steps % 4)).astype(np.float32)
        self.terminals.fill(0)
        if self.steps % 3 == 0:
            self.terminals[::3] = 1
        self.reward_history.append(self.rewards.copy())
        self.terminal_history.append(self.terminals.copy())
        self.observations[:] = np.roll(self.observations, 1, axis=1)
        self.observations[:, 0] = actions[:, 0] / 3.0
        self.observation_history.append(self.observations.copy())

    def log(self):
        return {"fake_steps": self.steps}

    def close(self):
        self.closed = True


def _hybrid_args():
    return {
        "world_size": 1,
        "torch": {"device": "mps", "rollout_device": "cpu"},
        "train": {
            "horizon": 4,
            "minibatch_size": 32,
            "total_timesteps": 32,
            "learning_rate": 0.005,
            "beta1": 0.9,
            "eps": 1e-8,
            "prio_beta0": 0.2,
            "prio_alpha": 0.8,
            "clip_coef": 0.2,
            "vf_clip_coef": 0.2,
            "anneal_lr": False,
            "min_lr_ratio": 0.0,
            "replay_ratio": 1.0,
            "gamma": 0.99,
            "gae_lambda": 0.9,
            "vtrace_rho_clip": 1.0,
            "vtrace_c_clip": 1.0,
            "vf_coef": 1.0,
            "ent_coef": 0.01,
            "max_grad_norm": 1.0,
        },
    }


class _CountingStateNetwork(torch.nn.Module):
    def initial_state(self, batch_size, device):
        return (torch.zeros(batch_size, 1, device=device),)

    def forward_eval(self, hidden, state):
        return hidden, (state[0] + 1,)

    def forward_train(self, hidden):
        return hidden


@pytest.mark.parametrize(("reset_state", "expected"), [(True, 4), (False, 8)])
def test_torch_rollout_honors_reset_state(reset_state, expected):
    torch_pufferl = _torch_pufferl()
    vec = _FakeCPUVec()
    args = _hybrid_args()
    args["reset_state"] = reset_state
    args["torch"] = {"device": "cpu", "rollout_device": "cpu"}
    policy = models.Policy(
        models.DefaultEncoder(vec.obs_size, hidden_size=16),
        models.DefaultDecoder(vec.act_sizes, hidden_size=16),
        _CountingStateNetwork(),
    )
    trainer = torch_pufferl.PuffeRL(
        args, vec, policy, verbose=False, device="cpu", rollout_device="cpu"
    )
    try:
        trainer.rollouts()
        trainer.rollouts()
        torch.testing.assert_close(
            trainer.state[0], torch.full_like(trainer.state[0], expected)
        )
    finally:
        trainer.close()


def test_short_hybrid_cpu_rollout_mps_train_step():
    _require_mps()
    torch_pufferl = _torch_pufferl()
    vec = _FakeCPUVec()
    with torch.random.fork_rng():
        torch.manual_seed(1859)
        policy = models.Policy(
            models.DefaultEncoder(vec.obs_size, hidden_size=16),
            models.DefaultDecoder(vec.act_sizes, hidden_size=16),
            models.MLP(hidden_size=16, num_layers=1),
        )

    trainer = torch_pufferl.PuffeRL(
        _hybrid_args(), vec, policy, verbose=False,
        device="mps", rollout_device="cpu",
    )
    try:
        assert trainer.device == torch.device("mps")
        assert trainer.rollout_device == torch.device("cpu")
        assert not trainer.host_horizon_io
        assert trainer.actions.device.type == "cpu"
        assert trainer.rewards.device.type == "cpu"
        assert trainer.terminals.device.type == "cpu"
        assert trainer.policy is not trainer.rollout_policy
        assert next(trainer.policy.parameters()).device.type == "mps"
        assert next(trainer.rollout_policy.parameters()).device.type == "cpu"

        parameters_before = [
            parameter.detach().cpu().clone()
            for parameter in trainer.policy.parameters()
        ]
        trainer.rollouts()
        assert trainer.global_step == 32
        assert vec.steps == 4
        assert vec.last_actions is not None
        assert vec.last_actions.dtype == np.float32

        trainer.train()
        assert trainer.epoch == 1
        assert trainer.losses
        assert all(math.isfinite(float(value)) for value in trainer.losses.values())
        assert any(
            not torch.equal(before, after.detach().cpu())
            for before, after in zip(parameters_before, trainer.policy.parameters())
        )

        trainer._sync_rollout_policy()
        for training_value, rollout_value in zip(
                trainer.policy.state_dict().values(),
                trainer.rollout_policy.state_dict().values()):
            torch.testing.assert_close(
                training_value.detach().cpu(), rollout_value,
                rtol=0, atol=0,
            )
    finally:
        trainer.close()
        assert vec.closed


@pytest.mark.parametrize("amp_dtype", ["float32", "bfloat16"])
def test_short_direct_mps_rollout_and_train_step(amp_dtype):
    _require_mps()
    torch_pufferl = _torch_pufferl()
    vec = _FakeCPUVec()
    args = _hybrid_args()
    args["torch"]["rollout_device"] = "mps"
    args["torch"]["amp_dtype"] = amp_dtype
    with torch.random.fork_rng():
        torch.manual_seed(3491)
        policy = models.Policy(
            models.DefaultEncoder(vec.obs_size, hidden_size=16),
            models.DefaultDecoder(vec.act_sizes, hidden_size=16),
            models.MLP(hidden_size=16, num_layers=1),
        )

    trainer = torch_pufferl.PuffeRL(
        args, vec, policy, verbose=False,
        device="mps", rollout_device="mps",
    )
    try:
        assert trainer.policy is trainer.rollout_policy
        assert trainer.host_horizon_io
        assert trainer.observations.device.type == "mps"
        assert trainer.values.device.type == "mps"
        assert trainer.logprobs.device.type == "mps"
        assert trainer.rewards.device.type == "cpu"
        assert trainer.terminals.device.type == "cpu"
        assert trainer.vec_actions is None
        if trainer.mps_host_alias_io:
            assert trainer.actions.device.type == "mps"
            assert trainer.host_observations.device.type == "cpu"
            assert trainer.host_actions.device.type == "cpu"
        else:
            assert trainer.actions.device.type == "cpu"
        advantage_ptr = trainer.advantages.data_ptr()
        trainer.advantages.fill_(torch.nan)
        trainer.rollouts()
        np.testing.assert_array_equal(
            trainer.actions.cpu().numpy(), np.stack(vec.action_history))
        expected_action_storage = (
            trainer.host_actions
            if trainer.mps_host_alias_io else trainer.actions
        )
        assert vec.action_pointers == [
            expected_action_storage[t].data_ptr()
            for t in range(trainer.config["horizon"])
        ]
        np.testing.assert_array_equal(
            trainer.observations.cpu().numpy(),
            np.stack(vec.observation_history[:-1]),
        )
        np.testing.assert_array_equal(
            trainer.rewards[0].numpy(), np.zeros(vec.total_agents))
        np.testing.assert_array_equal(
            trainer.terminals[0].numpy(), np.zeros(vec.total_agents))
        np.testing.assert_array_equal(
            trainer.rewards[1:].numpy(), np.stack(vec.reward_history[:-1]))
        np.testing.assert_array_equal(
            trainer.terminals[1:].numpy(), np.stack(vec.terminal_history[:-1]))
        trainer.train()
        assert trainer.advantages.data_ptr() == advantage_ptr
        assert torch.isfinite(trainer.advantages).all().item()
        assert torch.count_nonzero(trainer.advantages[:, -1]).item() == 0
        assert trainer.global_step == 32
        assert trainer.epoch == 1
        assert vec.steps == 4
        assert vec.last_actions is not None
        assert all(math.isfinite(float(value)) for value in trainer.losses.values())
        assert all(
            torch.isfinite(parameter).all().item()
            for parameter in trainer.policy.parameters()
        )
    finally:
        trainer.close()
        assert vec.closed


def test_host_alias_and_staged_mps_epochs_are_bitwise_equivalent():
    _require_mps_host_alias()
    torch_pufferl = _torch_pufferl()
    alias_vec = _FakeCPUVec()
    staged_vec = _FakeCPUVec()
    args = _hybrid_args()
    args["torch"]["rollout_device"] = "mps"
    args["torch"]["amp_dtype"] = "float32"
    staged_args = deepcopy(args)
    staged_args["torch"]["mps_host_alias"] = "off"

    with torch.random.fork_rng():
        torch.manual_seed(772901)
        policy = models.Policy(
            models.DefaultEncoder(alias_vec.obs_size, hidden_size=16),
            models.DefaultDecoder(alias_vec.act_sizes, hidden_size=16),
            models.MinGRU(hidden_size=16, num_layers=2),
        )
    alias_trainer = torch_pufferl.PuffeRL(
        args, alias_vec, deepcopy(policy), verbose=False,
        device="mps", rollout_device="mps",
    )
    staged_trainer = torch_pufferl.PuffeRL(
        staged_args, staged_vec, deepcopy(policy), verbose=False,
        device="mps", rollout_device="mps",
    )
    try:
        assert alias_trainer.mps_host_alias_io
        assert not staged_trainer.mps_host_alias_io

        torch.manual_seed(93317)
        torch.mps.manual_seed(93317)
        alias_trainer.rollouts()
        torch.manual_seed(93317)
        torch.mps.manual_seed(93317)
        staged_trainer.rollouts()

        for name in (
            "observations", "actions", "rewards", "terminals",
            "values", "logprobs",
        ):
            torch.testing.assert_close(
                getattr(alias_trainer, name).cpu(),
                getattr(staged_trainer, name).cpu(),
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            alias_trainer.state[0].cpu(), staged_trainer.state[0].cpu(),
            rtol=0, atol=0,
        )

        torch.manual_seed(44519)
        torch.mps.manual_seed(44519)
        alias_trainer.train()
        torch.manual_seed(44519)
        torch.mps.manual_seed(44519)
        staged_trainer.train()

        for alias_parameter, staged_parameter in zip(
            alias_trainer.policy.parameters(), staged_trainer.policy.parameters()
        ):
            torch.testing.assert_close(
                alias_parameter, staged_parameter, rtol=0, atol=0
            )
        assert alias_trainer.losses == staged_trainer.losses
    finally:
        alias_trainer.close()
        staged_trainer.close()


def test_mps_host_alias_failure_uses_staged_fallback(monkeypatch):
    _require_mps()
    torch_pufferl = _torch_pufferl()
    vec = _FakeCPUVec()
    args = _hybrid_args()
    args["torch"]["rollout_device"] = "mps"

    def unavailable(_tensor):
        raise RuntimeError("simulated unavailable shared Metal storage")

    monkeypatch.setattr(torch_pufferl, "mps_host_alias", unavailable)
    policy = models.Policy(
        models.DefaultEncoder(vec.obs_size, hidden_size=16),
        models.DefaultDecoder(vec.act_sizes, hidden_size=16),
        models.MLP(hidden_size=16, num_layers=1),
    )
    trainer = torch_pufferl.PuffeRL(
        args, vec, policy, verbose=False,
        device="mps", rollout_device="mps",
    )
    try:
        assert trainer.host_horizon_io
        assert not trainer.mps_host_alias_io
        assert trainer.actions.device.type == "cpu"
        trainer.rollouts()
        trainer.train()
        assert all(math.isfinite(value) for value in trainer.losses.values())
    finally:
        trainer.close()


def test_second_mps_host_alias_failure_clears_partial_state(monkeypatch):
    _require_mps()
    torch_pufferl = _torch_pufferl()
    vec = _FakeCPUVec()
    args = _hybrid_args()
    args["torch"]["rollout_device"] = "mps"
    calls = 0

    def fail_for_actions(tensor):
        nonlocal calls
        calls += 1
        if calls == 1:
            return torch.empty_like(tensor, device="cpu")
        raise RuntimeError("simulated action host-alias failure")

    monkeypatch.setattr(torch_pufferl, "mps_host_alias", fail_for_actions)
    policy = models.Policy(
        models.DefaultEncoder(vec.obs_size, hidden_size=16),
        models.DefaultDecoder(vec.act_sizes, hidden_size=16),
        models.MLP(hidden_size=16, num_layers=1),
    )
    trainer = torch_pufferl.PuffeRL(
        args, vec, policy, verbose=False,
        device="mps", rollout_device="mps",
    )
    try:
        assert calls == 2
        assert trainer.host_horizon_io
        assert not trainer.mps_host_alias_io
        assert trainer.host_observations is None
        assert trainer.host_actions is None
        assert trainer.actions.device.type == "cpu"
        trainer.rollouts()
        trainer.train()
        assert all(math.isfinite(value) for value in trainer.losses.values())
    finally:
        trainer.close()


def test_invalid_mps_host_alias_mode_fails_actionably():
    torch_pufferl = _torch_pufferl()
    vec = _FakeCPUVec()
    args = _hybrid_args()
    args["torch"] = {
        "device": "cpu",
        "rollout_device": "cpu",
        "mps_host_alias": "sometimes",
    }
    policy = models.Policy(
        models.DefaultEncoder(vec.obs_size, hidden_size=16),
        models.DefaultDecoder(vec.act_sizes, hidden_size=16),
        models.MLP(hidden_size=16, num_layers=1),
    )
    with pytest.raises(ValueError, match="must be auto, on, or off"):
        torch_pufferl.PuffeRL(
            args, vec, policy, verbose=False,
            device="cpu", rollout_device="cpu",
        )
    vec.close()
