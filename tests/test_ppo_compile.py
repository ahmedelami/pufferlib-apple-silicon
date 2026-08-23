from contextlib import contextmanager
import math
import os
import sys

import pytest
import torch

from pufferlib import models
from pufferlib import torch_pufferl


@contextmanager
def _clean_argv():
    previous = sys.argv
    sys.argv = [previous[0]]
    try:
        yield
    finally:
        sys.argv = previous


def _policy():
    return models.Policy(
        models.DefaultEncoder(118, hidden_size=64),
        models.DefaultDecoder([3], hidden_size=64),
        models.MinGRU(hidden_size=64, num_layers=2),
    )


def _args(policy_mode="inductor", ppo_mode="inductor"):
    return {
        "torch": {
            "compile_policy": policy_mode,
            "compile_ppo": ppo_mode,
        },
        "train": {
            "total_timesteps": 94_000_000,
            "horizon": 64,
            "minibatch_size": 65_536,
            "clip_coef": 0.31,
            "vf_clip_coef": 0.73,
            "vf_coef": 1.17,
            "ent_coef": 0.019,
            "max_grad_norm": 0.91,
        },
    }


def _dynamo_wrapper(target, implementation=None):
    implementation = target if implementation is None else implementation

    def wrapper(*args, **kwargs):
        return implementation(*args, **kwargs)

    wrapper._torchdynamo_orig_callable = target
    return wrapper


def _fixtures(segments=2, horizon=3):
    generator = torch.Generator().manual_seed(817)
    observations = torch.randn(
        segments, horizon, 118, generator=generator)
    actions = torch.arange(segments * horizon).remainder(3) \
        .reshape(segments, horizon, 1).float()
    old_logprobs = torch.full((segments, horizon), -4.0)
    old_values = torch.randn(
        segments, horizon, generator=generator) * 0.2
    returns = old_values + 0.5
    advantages = torch.linspace(-1.0, 1.0, segments * horizon) \
        .reshape(segments, horizon)
    priority = torch.linspace(0.8, 1.2, segments).unsqueeze(1)
    return (
        observations, actions, old_logprobs, old_values,
        returns, advantages, priority)


def test_ppo_train_outputs_use_supplied_actions_and_runtime_coefficients():
    torch.manual_seed(31)
    policy = _policy()
    fixtures = _fixtures()
    rng_before = torch.random.get_rng_state().clone()

    first = torch_pufferl._ppo_train_outputs(
        policy.forward, *fixtures, 0.1, 0.2, 0.7, 0.03)
    second = torch_pufferl._ppo_train_outputs(
        policy.forward, *fixtures, 0.8, 1.1, 1.9, 0.2)

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert torch.equal(first[2], second[2])
    assert not torch.equal(first[6], second[6])
    assert not torch.equal(first[7], second[7])
    torch.testing.assert_close(
        first[-1], first[6] + 0.7 * first[7] - 0.03 * first[8])
    torch.testing.assert_close(
        second[-1], second[6] + 1.9 * second[7] - 0.2 * second[8])

    changed_actions = list(fixtures)
    changed_actions[1] = (changed_actions[1] + 1).remainder(3)
    changed = torch_pufferl._ppo_train_outputs(
        policy.forward, *changed_actions, 0.1, 0.2, 0.7, 0.03)
    assert not torch.equal(first[2], changed[2])


def test_ppo_compile_preflight_preserves_parameters_mode_and_rng():
    torch.manual_seed(67)
    policy = _policy()
    policy.train(False)
    parameters = {
        name: value.detach().clone()
        for name, value in policy.named_parameters()
    }
    parameter_ids = [id(value) for value in policy.parameters()]
    rng_before = torch.random.get_rng_state().clone()

    def candidate(*inputs):
        return torch_pufferl._ppo_train_outputs(policy.forward, *inputs)

    torch_pufferl._preflight_ppo_compile(
        candidate,
        policy.forward,
        policy,
        "cpu",
        2,
        4,
        118,
        (0.2, 0.4, 1.3, 0.01),
        0.5,
    )

    assert policy.training is False
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert [id(value) for value in policy.parameters()] == parameter_ids
    assert all(torch.equal(parameters[name], value)
        for name, value in policy.named_parameters())
    assert all(value.grad is None for value in policy.parameters())


def test_configure_ppo_compile_preserves_policy_and_passes_dynamic_coefficients(
        monkeypatch):
    policy = _policy()
    state_keys = tuple(policy.state_dict())
    parameter_ids = [id(value) for value in policy.parameters()]
    policy.forward = _dynamo_wrapper(policy.forward)
    calls = []
    preflight = []

    def compile_stub(target, **kwargs):
        calls.append((target, kwargs))
        return _dynamo_wrapper(target)

    def preflight_stub(*args):
        preflight.append(args)

    monkeypatch.setattr(torch, "compile", compile_stub)
    monkeypatch.setattr(
        torch_pufferl, "_preflight_ppo_compile", preflight_stub)
    elapsed = iter((10.0, 10.5))
    monkeypatch.setattr(
        torch_pufferl.time, "perf_counter", lambda: next(elapsed))

    (compiled, requested, effective, reason, startup,
     checked, wrapper_verified) = torch_pufferl.configure_ppo_compile(
        _args(), policy, "mps", None, "inductor")

    assert compiled is not None
    assert requested == "inductor"
    assert effective == "inductor"
    assert "trainer coefficients supplied as graph inputs" in reason
    assert startup == 0.5
    assert checked is True
    assert wrapper_verified is True
    assert len(calls) == 1
    assert calls[0][1] == {
        "backend": "inductor",
        "fullgraph": True,
        "dynamic": False,
        "options": {"layout_optimization": False},
    }
    assert len(preflight) == 1
    assert preflight[0][-2] == (0.31, 0.73, 1.17, 0.019)
    assert preflight[0][-1] == 0.91
    assert tuple(policy.state_dict()) == state_keys
    assert [id(value) for value in policy.parameters()] == parameter_ids


@pytest.mark.parametrize(
    ("amp_dtype", "core_effective", "reason_fragment"),
    [
        (None, "off", "compiled-policy path is inactive"),
        (torch.bfloat16, "inductor", "validated only for FP32"),
    ],
)
def test_ppo_compile_falls_back_without_touching_policy(
        monkeypatch, amp_dtype, core_effective, reason_fragment):
    policy = _policy()
    parameter_ids = [id(value) for value in policy.parameters()]
    monkeypatch.setattr(
        torch, "compile",
        lambda *args, **kwargs: pytest.fail("compiler should not be called"),
    )

    result = torch_pufferl.configure_ppo_compile(
        _args(ppo_mode="auto"), policy, "mps", amp_dtype, core_effective)

    assert result[0] is None
    assert result[2] == "off"
    assert reason_fragment in result[3]
    assert result[4] == 0.0
    assert result[5:] == (False, False)
    assert [id(value) for value in policy.parameters()] == parameter_ids


def test_ppo_compile_setup_failure_keeps_compiled_policy_fallback(monkeypatch):
    policy = _policy()
    compiled_forward = _dynamo_wrapper(policy.forward)
    policy.forward = compiled_forward
    parameter_ids = [id(value) for value in policy.parameters()]
    monkeypatch.setattr(
        torch, "compile",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic PPO compile failure")),
    )
    elapsed = iter((20.0, 20.25))
    monkeypatch.setattr(
        torch_pufferl.time, "perf_counter", lambda: next(elapsed))

    result = torch_pufferl.configure_ppo_compile(
        _args(ppo_mode="auto"), policy, "mps", None, "inductor")

    assert result[0] is None
    assert result[2] == "off"
    assert "synthetic PPO compile failure" in result[3]
    assert result[4] == 0.25
    assert policy.forward is compiled_forward
    assert [id(value) for value in policy.parameters()] == parameter_ids


def test_ppo_compile_auto_respects_measured_break_even_but_explicit_forces(
        monkeypatch):
    policy = _policy()
    policy.forward = _dynamo_wrapper(policy.forward)
    args = _args(ppo_mode="auto")
    args["train"]["total_timesteps"] = (
        torch_pufferl._VALIDATED_PPO_COMPILE_BREAK_EVEN_TIMESTEPS - 1)
    monkeypatch.setattr(
        torch, "compile",
        lambda *args, **kwargs: pytest.fail("auto must avoid cold setup"),
    )

    result = torch_pufferl.configure_ppo_compile(
        args, policy, "mps", None, "inductor")

    assert result[2] == "off"
    assert "below the measured 54800000-step" in result[3]
    assert result[4:] == (0.0, False, False)

    args["torch"]["compile_ppo"] = "inductor"
    calls = []

    def compile_stub(target, **kwargs):
        calls.append(target)
        return _dynamo_wrapper(target)

    monkeypatch.setattr(torch, "compile", compile_stub)
    monkeypatch.setattr(
        torch_pufferl, "_preflight_ppo_compile", lambda *args: None)
    result = torch_pufferl.configure_ppo_compile(
        args, policy, "mps", None, "inductor")
    assert result[2] == "inductor"
    assert len(calls) == 1


def test_explicit_ppo_compile_fails_closed(monkeypatch):
    policy = _policy()
    policy.forward = _dynamo_wrapper(policy.forward)
    monkeypatch.setattr(
        torch, "compile",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic explicit failure")),
    )

    with pytest.raises(RuntimeError, match="validated FP32 compiled PPO"):
        torch_pufferl.configure_ppo_compile(
            _args(ppo_mode="inductor"),
            policy,
            "mps",
            None,
            "inductor",
        )

    with pytest.raises(ValueError, match="requires the validated FP32"):
        torch_pufferl.configure_ppo_compile(
            _args(ppo_mode="inductor"),
            policy,
            "mps",
            torch.bfloat16,
            "inductor",
        )


def test_explicit_ppo_compile_rejects_missing_original_wrapper():
    with pytest.raises(RuntimeError, match="does not expose its original"):
        torch_pufferl.configure_ppo_compile(
            _args(ppo_mode="inductor"),
            _policy(),
            "mps",
            None,
            "inductor",
        )


def test_ppo_compile_mode_validation_and_config_scope():
    with pytest.raises(ValueError, match="must be off, auto, or inductor"):
        torch_pufferl.configure_ppo_compile(
            _args(ppo_mode="fastest"), _policy(), "cpu", None, "off")

    from pufferlib.pufferl import load_config

    with _clean_argv():
        breakout = load_config("breakout")
        snake = load_config("snake")
    # The speed path remains explicit after its combined preregistered
    # learning-quality holdout failed.
    assert breakout["torch"]["compile_ppo"] == "off"
    assert snake["torch"]["compile_ppo"] == "off"


def test_production_fp32_ppo_compile_executes_without_recompile():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    if not hasattr(torch.mps, "_host_alias_storage"):
        pytest.skip("private MPS host-alias API is unavailable")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        pytest.skip("requires PYTORCH_ENABLE_MPS_FALLBACK=0 before import")

    from pufferlib import _C
    from pufferlib.pufferl import load_config

    if getattr(_C, "env_name", None) != "breakout":
        pytest.skip("requires the Breakout native extension")
    with _clean_argv():
        args = load_config("breakout")
    args["slowly"] = True
    args["profile"] = False
    args["rank"] = 0
    args["world_size"] = 1
    args["gpu_id"] = 0
    args["torch"]["device"] = "mps"
    args["torch"]["rollout_device"] = "mps"
    args["torch"]["amp_dtype"] = "float32"
    args["torch"]["mps_host_alias"] = "on"
    args["torch"]["compile_policy"] = "inductor"
    args["torch"]["compile_ppo"] = "inductor"
    args["vec"]["total_agents"] = 4096
    args["vec"]["num_buffers"] = 1
    args["vec"]["num_threads"] = 18
    args["train"]["horizon"] = 64
    args["train"]["minibatch_size"] = 65_536
    # Keep the shipped 1.424... replay ratio: this exercises five minibatches,
    # including the changing priority/annealing inputs used after minibatch 1.
    args["train"]["total_timesteps"] = 4096 * 64

    vec = _C.create_vec(args, 0)
    try:
        policy = torch_pufferl.load_policy(args, vec, device="mps")
        state_keys = tuple(policy.state_dict())
        parameter_ids = [id(value) for value in policy.parameters()]
        trainer = torch_pufferl.PuffeRL(
            args, vec, policy, verbose=False,
            device="mps", rollout_device="mps")
    except Exception:
        vec.close()
        raise
    try:
        assert trainer.policy_compile_effective == "inductor"
        assert trainer.ppo_compile_requested == "inductor"
        assert trainer.ppo_compile_effective == "inductor"
        assert trainer.ppo_compile_preflight is True
        assert trainer.ppo_compile_wrapper_verified is True
        assert trainer.ppo_compile_startup_seconds > 0.0
        assert trainer.compiled_ppo is not None
        assert tuple(trainer.policy.state_dict()) == state_keys
        assert [id(value) for value in trainer.policy.parameters()] == parameter_ids
        assert [id(value) for group in trainer.optimizer.param_groups
            for value in group["params"]] == parameter_ids

        from torch._dynamo.utils import counters
        counters.clear()
        trainer.rollouts()
        trainer.train()
        torch.mps.synchronize()
        assert counters["frames"]["total"] == 0
        assert counters["stats"]["unique_graphs"] == 0
        assert all(math.isfinite(float(value))
            for value in trainer.losses.values())
        assert all(torch.isfinite(value).all()
            for value in trainer.policy.parameters())
        assert all(
            state["momentum_buffer"].dtype == torch.float32
            and torch.isfinite(state["momentum_buffer"]).all()
            for state in trainer.optimizer.state.values())
    finally:
        trainer.close()
