from contextlib import contextmanager
import math
import os
import sys
from types import SimpleNamespace

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


def _args(mode="inductor"):
    return {
        "env_name": "breakout",
        "world_size": 1,
        "torch": {"compile_policy": mode},
        "train": {"horizon": 64, "minibatch_size": 65_536},
        "policy": {"hidden_size": 64, "num_layers": 2},
    }


def _vec():
    return SimpleNamespace(
        total_agents=4096,
        obs_size=118,
        act_sizes=[3],
        gpu=0,
    )


def _state():
    return (torch.zeros(2, 4096, 64),)


def _dynamo_wrapper(target, implementation=None):
    implementation = target if implementation is None else implementation

    def wrapper(*args, **kwargs):
        return implementation(*args, **kwargs)

    wrapper._torchdynamo_orig_callable = target
    return wrapper


def _mock_validated_host_and_geometry(monkeypatch):
    monkeypatch.setattr(
        torch_pufferl,
        "_VALIDATED_COMPILE_TORCH_VERSION",
        str(torch.__version__).split("+", 1)[0],
    )
    monkeypatch.setattr(torch_pufferl, "_compile_system_identity", lambda: {
        "system": "Darwin",
        "machine": "arm64",
        "hardware": "Mac17,8",
        "chip": "Apple M5 Pro",
        "gpu_model": "Apple M5 Pro",
        "gpu_cores": 20,
        "memory_bytes": 24 * 2**30,
        "macos_version": "27.0",
        "macos_build": "26A5378j",
        "torch_git": torch_pufferl._VALIDATED_COMPILE_TORCH_GIT,
    })
    monkeypatch.setattr(
        torch_pufferl, "_policy_geometry_mismatches",
        lambda policy, device: [],
    )


def test_validated_policy_compile_preserves_parameters_and_passes_options(
        monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    _mock_validated_host_and_geometry(monkeypatch)
    monkeypatch.setattr(
        torch_pufferl, "_preflight_policy_compile",
        lambda *args, **kwargs: None,
    )
    elapsed = iter((10.0, 10.25))
    monkeypatch.setattr(
        torch_pufferl.time, "perf_counter", lambda: next(elapsed))
    calls = []

    def compile_stub(target, **kwargs):
        calls.append((target, kwargs))
        return _dynamo_wrapper(target)

    monkeypatch.setattr(torch, "compile", compile_stub)
    policy = _policy()
    parameter_ids = [id(param) for param in policy.parameters()]

    requested, effective, reason, startup, preflight = (
        torch_pufferl.configure_policy_compile(
            _args(), _vec(), policy, "mps", "mps", None, True, _state()))

    assert requested == "inductor"
    assert effective == "inductor"
    assert "validated Breakout" in reason
    assert "Dynamo wrappers verified" in reason
    assert startup == 0.25
    assert preflight is True
    assert [id(param) for param in policy.parameters()] == parameter_ids
    assert len(calls) == 2
    assert all(kwargs == {
        "backend": "inductor",
        "fullgraph": True,
        "dynamic": False,
        "options": {"layout_optimization": False},
    } for _, kwargs in calls)


def test_policy_compile_auto_falls_back_outside_validated_shape(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    _mock_validated_host_and_geometry(monkeypatch)
    monkeypatch.setattr(
        torch, "compile",
        lambda *args, **kwargs: pytest.fail("compiler should not be called"),
    )
    vec = _vec()
    vec.total_agents = 1024
    elapsed = iter((11.0, 11.25))
    monkeypatch.setattr(
        torch_pufferl.time, "perf_counter", lambda: next(elapsed))

    requested, effective, reason, startup, preflight = (
        torch_pufferl.configure_policy_compile(
            _args("auto"), vec, _policy(), "mps", "mps", None, True,
            _state()))

    assert requested == "auto"
    assert effective == "off"
    assert "total_agents is not 4096" in reason
    assert startup == 0.25
    assert preflight is False


def test_policy_compile_auto_stays_eager_on_other_apple_hardware(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    _mock_validated_host_and_geometry(monkeypatch)
    identity = torch_pufferl._compile_system_identity()
    monkeypatch.setattr(
        torch_pufferl, "_compile_system_identity",
        lambda: {**identity, "hardware": "Mac14,5"},
    )
    monkeypatch.setattr(
        torch, "compile",
        lambda *args, **kwargs: pytest.fail("compiler should not be called"),
    )
    elapsed = iter((12.0, 12.5))
    monkeypatch.setattr(
        torch_pufferl.time, "perf_counter", lambda: next(elapsed))

    requested, effective, reason, startup, preflight = (
        torch_pufferl.configure_policy_compile(
            _args("auto"), _vec(), _policy(), "mps", "mps", None, True,
            _state()))

    assert requested == "auto"
    assert effective == "off"
    assert "hardware is not Mac17,8" in reason
    assert startup == 0.5
    assert preflight is False


def test_explicit_policy_compile_fails_closed_outside_validated_shape(
        monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    _mock_validated_host_and_geometry(monkeypatch)
    args = _args()
    args["train"]["horizon"] = 32

    with pytest.raises(ValueError, match="horizon is not 64"):
        torch_pufferl.configure_policy_compile(
            args, _vec(), _policy(), "mps", "mps", None, True, _state())


def test_policy_compile_setup_failure_does_not_half_mutate_policy(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    _mock_validated_host_and_geometry(monkeypatch)
    policy = _policy()
    original_eval = policy.forward_eval.__func__
    original_train = policy.forward.__func__
    calls = 0

    def fail_second(target, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic compile setup failure")
        return lambda *args, **kwargs: target(*args, **kwargs)

    monkeypatch.setattr(torch, "compile", fail_second)
    with pytest.raises(RuntimeError, match="failed to configure"):
        torch_pufferl.configure_policy_compile(
            _args(), _vec(), policy, "mps", "mps", None, True, _state())

    assert policy.forward_eval.__func__ is original_eval
    assert policy.forward.__func__ is original_train


def test_policy_compile_auto_setup_failure_stays_eager(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    _mock_validated_host_and_geometry(monkeypatch)
    policy = _policy()
    original_eval = policy.forward_eval.__func__
    original_train = policy.forward.__func__
    parameter_ids = [id(param) for param in policy.parameters()]
    calls = 0

    def fail_second(target, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic compile setup failure")
        return lambda *args, **kwargs: target(*args, **kwargs)

    elapsed = iter((20.0, 20.5))
    monkeypatch.setattr(
        torch_pufferl.time, "perf_counter", lambda: next(elapsed))
    monkeypatch.setattr(torch, "compile", fail_second)
    requested, effective, reason, startup, preflight = (
        torch_pufferl.configure_policy_compile(
            _args("auto"), _vec(), policy, "mps", "mps", None, True,
            _state()))

    assert requested == "auto"
    assert effective == "off"
    assert "compile/preflight failed: RuntimeError" in reason
    assert startup == 0.5
    assert preflight is False
    assert policy.forward_eval.__func__ is original_eval
    assert policy.forward.__func__ is original_train
    assert [id(param) for param in policy.parameters()] == parameter_ids
    assert all(param.grad is None for param in policy.parameters())


def test_invalid_policy_compile_mode_is_actionable():
    with pytest.raises(ValueError, match="must be off, auto, or inductor"):
        torch_pufferl.configure_policy_compile(
            _args("fastest"), _vec(), _policy(), "mps", "mps", None, True,
            _state())


def test_policy_compile_config_is_scoped_to_breakout():
    from pufferlib.pufferl import load_config

    with _clean_argv():
        breakout = load_config("breakout")
        snake = load_config("snake")
    assert breakout["torch"]["compile_policy"] == "auto"
    assert snake["torch"]["compile_policy"] == "off"


def test_lazy_compile_failure_is_caught_before_policy_mutation(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    _mock_validated_host_and_geometry(monkeypatch)
    policy = _policy()
    original_eval = policy.forward_eval.__func__
    original_train = policy.forward.__func__

    def lazy_failure(target, **kwargs):
        def fail_when_materialized(*args, **kwargs):
            raise RuntimeError("synthetic lazy backend failure")
        return _dynamo_wrapper(target, fail_when_materialized)

    monkeypatch.setattr(torch, "compile", lazy_failure)
    monkeypatch.setattr(
        torch_pufferl, "_preflight_policy_compile",
        lambda compiled_eval, *args, **kwargs: compiled_eval(None, None),
    )
    with pytest.raises(RuntimeError, match="failed to configure"):
        torch_pufferl.configure_policy_compile(
            _args(), _vec(), policy, "mps", "mps", None, True, _state())

    assert policy.forward_eval.__func__ is original_eval
    assert policy.forward.__func__ is original_train


def test_lazy_compile_failure_in_auto_stays_eager(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    _mock_validated_host_and_geometry(monkeypatch)
    policy = _policy()
    original_eval = policy.forward_eval.__func__
    original_train = policy.forward.__func__
    parameter_ids = [id(param) for param in policy.parameters()]

    def lazy_failure(target, **kwargs):
        def fail_when_materialized(*args, **kwargs):
            raise RuntimeError("synthetic lazy backend failure")
        return _dynamo_wrapper(target, fail_when_materialized)

    monkeypatch.setattr(torch, "compile", lazy_failure)
    monkeypatch.setattr(
        torch_pufferl, "_preflight_policy_compile",
        lambda compiled_eval, *args, **kwargs: compiled_eval(None, None),
    )
    elapsed = iter((30.0, 30.75))
    monkeypatch.setattr(
        torch_pufferl.time, "perf_counter", lambda: next(elapsed))
    requested, effective, reason, startup, preflight = (
        torch_pufferl.configure_policy_compile(
            _args("auto"), _vec(), policy, "mps", "mps", None, True,
            _state()))

    assert requested == "auto"
    assert effective == "off"
    assert "synthetic lazy backend failure" in reason
    assert startup == 0.75
    assert preflight is False
    assert policy.forward_eval.__func__ is original_eval
    assert policy.forward.__func__ is original_train
    assert [id(param) for param in policy.parameters()] == parameter_ids
    assert all(param.grad is None for param in policy.parameters())


def test_policy_compile_auto_rejects_custom_policy_without_attributes(
        monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    _mock_validated_host_and_geometry(monkeypatch)
    elapsed = iter((40.0, 40.125))
    monkeypatch.setattr(
        torch_pufferl.time, "perf_counter", lambda: next(elapsed))
    requested, effective, reason, startup, preflight = (
        torch_pufferl.configure_policy_compile(
            _args("auto"), _vec(), torch.nn.Identity(), "mps", "mps",
            None, True, _state()))

    assert requested == "auto"
    assert effective == "off"
    assert "policy class is not Policy" in reason
    assert "encoder is not DefaultEncoder" in reason
    assert startup == 0.125
    assert preflight is False


def test_policy_compile_auto_rejects_disabled_dynamo(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "1")
    _mock_validated_host_and_geometry(monkeypatch)
    monkeypatch.setattr(
        torch, "compile",
        lambda *args, **kwargs: pytest.fail("compiler should not be called"),
    )
    elapsed = iter((50.0, 50.25))
    monkeypatch.setattr(
        torch_pufferl.time, "perf_counter", lambda: next(elapsed))

    requested, effective, reason, startup, preflight = (
        torch_pufferl.configure_policy_compile(
            _args("auto"), _vec(), _policy(), "mps", "mps", None, True,
            _state()))

    assert requested == "auto"
    assert effective == "off"
    assert "TORCHDYNAMO_DISABLE is enabled" in reason
    assert startup == 0.25
    assert preflight is False


def test_policy_compile_auto_rejects_noop_compile_wrapper(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    monkeypatch.delenv("TORCHDYNAMO_DISABLE", raising=False)
    _mock_validated_host_and_geometry(monkeypatch)
    monkeypatch.setattr(torch, "compile", lambda target, **kwargs: target)
    elapsed = iter((60.0, 60.5))
    monkeypatch.setattr(
        torch_pufferl.time, "perf_counter", lambda: next(elapsed))

    requested, effective, reason, startup, preflight = (
        torch_pufferl.configure_policy_compile(
            _args("auto"), _vec(), _policy(), "mps", "mps", None, True,
            _state()))

    assert requested == "auto"
    assert effective == "off"
    assert "did not produce a Dynamo forward_eval wrapper" in reason
    assert startup == 0.5
    assert preflight is False


def test_actual_policy_geometry_rejects_caller_supplied_mismatch():
    policy = models.Policy(
        models.DefaultEncoder(118, hidden_size=32),
        models.DefaultDecoder([4], hidden_size=32),
        models.MinGRU(hidden_size=32, num_layers=1),
    )
    mismatches = torch_pufferl._policy_geometry_mismatches(policy, "cpu")
    assert "actual encoder geometry is not 118x64" in mismatches
    assert "actual MinGRU geometry is not 2x64" in mismatches
    assert "actual decoder geometry is not discrete [3] at width 64" in mismatches


def test_actual_policy_geometry_rejects_instance_method_override():
    policy = _policy()
    policy.forward = policy.forward
    policy.forward_eval = policy.forward_eval

    mismatches = torch_pufferl._policy_geometry_mismatches(policy, "cpu")

    assert "actual policy forward implementation is not validated" in mismatches
    assert (
        "actual policy forward_eval implementation is not validated"
        in mismatches)


def test_policy_compile_preflight_preserves_policy_state_and_rng():
    policy = _policy()
    policy.train(False)
    state = _state()
    parameters = {
        name: value.detach().clone()
        for name, value in policy.named_parameters()
    }
    state_before = tuple(value.clone() for value in state)
    rng_before = torch.random.get_rng_state().clone()

    torch_pufferl._preflight_policy_compile(
        policy.forward_eval,
        policy.forward,
        policy,
        state,
        "cpu",
        4096,
        64,
        65_536,
        118,
    )

    assert policy.training is False
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert all(torch.equal(before, after)
        for before, after in zip(state_before, state))
    assert all(torch.equal(parameters[name], value)
        for name, value in policy.named_parameters())
    assert all(value.grad is None for value in policy.parameters())


def test_production_policy_compile_executes_one_mps_epoch(tmp_path):
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
    args["vec"]["total_agents"] = 4096
    args["vec"]["num_buffers"] = 1
    args["vec"]["num_threads"] = 18
    args["train"]["horizon"] = 64
    args["train"]["minibatch_size"] = 65_536
    args["train"]["replay_ratio"] = 1.0
    args["train"]["total_timesteps"] = 4096 * 64

    vec = _C.create_vec(args, 0)
    try:
        policy = torch_pufferl.load_policy(args, vec, device="mps")
        state_keys = tuple(policy.state_dict())
        parameter_ids = [id(param) for param in policy.parameters()]
        trainer = torch_pufferl.PuffeRL(
            args, vec, policy, verbose=False,
            device="mps", rollout_device="mps")
    except Exception:
        vec.close()
        raise
    try:
        assert trainer.policy_compile_effective == "inductor"
        assert trainer.policy_compile_preflight is True
        assert trainer.policy_compile_wrapper_verified is True
        assert trainer.policy_compile_startup_seconds > 0.0
        assert tuple(trainer.policy.state_dict()) == state_keys
        assert [id(param) for param in trainer.policy.parameters()] == parameter_ids
        checkpoint_clone = _policy().to("mps")
        checkpoint_clone.load_state_dict(trainer.policy.state_dict(), strict=True)
        checkpoint_path = tmp_path / "compiled-policy.pt"
        trainer.save_weights(checkpoint_path)
        trainer.rollouts()
        trainer.train()
        assert all(math.isfinite(float(value)) for value in trainer.losses.values())
        assert all(torch.isfinite(param).all() for param in trainer.policy.parameters())
        assert any(not torch.equal(value, checkpoint_clone.state_dict()[name])
            for name, value in trainer.policy.state_dict().items())

        trainer.load_weights(checkpoint_path)
        assert [id(param) for param in trainer.policy.parameters()] == parameter_ids
        assert all(torch.equal(value, checkpoint_clone.state_dict()[name])
            for name, value in trainer.policy.state_dict().items())
        observations = torch.zeros(4096, 118, device="mps")
        state = checkpoint_clone.initial_state(4096, device="mps")
        with torch.no_grad():
            compiled_output = trainer.policy.forward_eval(observations, state)
            eager_output = checkpoint_clone.forward_eval(observations, state)
        for compiled, eager in zip(
                torch_pufferl._state_tensors(compiled_output),
                torch_pufferl._state_tensors(eager_output)):
            torch.testing.assert_close(compiled, eager, rtol=2e-6, atol=2e-5)
    finally:
        trainer.close()
