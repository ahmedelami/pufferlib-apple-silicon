from contextlib import contextmanager
import math
import os
import sys
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch._functorch.aot_autograd import aot_export_module
from torch._subclasses.fake_tensor import FakeTensorMode

from pufferlib import models
from pufferlib import mps_mingru
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


def _args(scan="metal", policy_compile="inductor"):
    return {
        "env_name": "breakout",
        "world_size": 1,
        "reset_state": True,
        "torch": {
            "compile_policy": policy_compile,
            "mingru_train_scan": scan,
        },
        "train": {
            "horizon": 64,
            "minibatch_size": 65_536,
            "total_timesteps": 94_000_000,
        },
        "policy": {"hidden_size": 64, "num_layers": 2},
    }


def _vec():
    return SimpleNamespace(
        total_agents=4096,
        obs_size=118,
        act_sizes=[3],
        gpu=0,
    )


def _mock_validated_target(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
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
        torch_pufferl, "_compiler_bisect_backend", lambda: None)
    original_geometry = torch_pufferl._policy_geometry_mismatches
    monkeypatch.setattr(
        torch_pufferl,
        "_policy_geometry_mismatches",
        lambda policy, device: original_geometry(policy, "cpu"),
    )


class _TwoLayerAOT(torch.nn.Module):
    def forward(self, input, first_weight, second_weight):
        hidden = mps_mingru.mingru_train_scan(
            F.linear(input, first_weight), input)
        hidden = mps_mingru.mingru_train_scan(
            F.linear(hidden, second_weight), hidden)
        return (hidden.sum(),)


def test_fake_aot_two_layer_forward_backward_is_dispatcher_visible():
    before = mps_mingru._library.cache_info()
    with FakeTensorMode():
        input = torch.empty(
            (1024, 64, 64), device="mps", requires_grad=True)
        weights = tuple(
            torch.empty(
                (192, 64), device="mps", requires_grad=True)
            for _ in range(2)
        )
        graph, _ = aot_export_module(
            _TwoLayerAOT(),
            (input, *weights),
            trace_joint=True,
            output_loss_index=0,
        )

    code = graph.code
    assert code.count("pufferlib.mingru_train_scan_forward") == 2
    assert code.count("pufferlib.mingru_train_scan_backward") == 2
    assert mps_mingru._library.cache_info() == before


def test_controller_promotes_exact_instance_without_identity_changes(
        monkeypatch):
    _mock_validated_target(monkeypatch)
    policy = _policy()
    class_forward = models.MinGRU.forward_train
    eval_forward = models.MinGRU.forward_eval
    state_keys = tuple(policy.state_dict())
    parameter_ids = tuple(id(value) for value in policy.parameters())

    monkeypatch.setattr(
        torch_pufferl,
        "configure_policy_compile",
        lambda *args: (
            "inductor", "inductor", "portable compiler passed", 0.25, True),
    )
    policy_result, scan_result = (
        torch_pufferl.configure_policy_and_mingru_train_scan(
            _args(), _vec(), policy, "mps", "mps", None, True, ()))

    assert policy_result[1] == "inductor"
    assert scan_result[0:2] == ("metal", "metal")
    assert scan_result[4] is True
    assert policy.network.forward_train.__func__ is mps_mingru.forward_train
    assert models.MinGRU.forward_train is class_forward
    assert models.MinGRU.forward_eval is eval_forward
    assert tuple(policy.state_dict()) == state_keys
    assert tuple(id(value) for value in policy.parameters()) == parameter_ids
    torch_pufferl._restore_mingru_train_scan(policy.network)


def test_auto_failure_restores_and_retries_portable_policy_once(monkeypatch):
    _mock_validated_target(monkeypatch)
    policy = _policy()
    state_keys = tuple(policy.state_dict())
    parameter_ids = tuple(id(value) for value in policy.parameters())
    calls = []

    def configure_stub(*args):
        network = args[2].network
        calls.append(getattr(network.forward_train, "__func__", None))
        if len(calls) == 1:
            raise RuntimeError("synthetic Metal trace failure")
        return (
            "inductor", "inductor", "portable compiler passed", 0.25, True)

    monkeypatch.setattr(
        torch_pufferl, "configure_policy_compile", configure_stub)
    policy_result, scan_result = (
        torch_pufferl.configure_policy_and_mingru_train_scan(
            _args("auto"), _vec(), policy,
            "mps", "mps", None, True, ()))

    assert calls == [
        mps_mingru.forward_train,
        torch_pufferl._VALIDATED_MINGRU_FORWARD_TRAIN,
    ]
    assert policy_result[1] == "inductor"
    assert policy_result[3] >= 0.25
    assert scan_result[0:2] == ("auto", "off")
    assert "portable policy retried" in scan_result[2]
    assert "forward_train" not in policy.network.__dict__
    assert tuple(policy.state_dict()) == state_keys
    assert tuple(id(value) for value in policy.parameters()) == parameter_ids


def test_explicit_metal_failure_restores_and_fails_closed(monkeypatch):
    _mock_validated_target(monkeypatch)
    policy = _policy()
    monkeypatch.setattr(
        torch_pufferl,
        "configure_policy_compile",
        lambda *args: (_ for _ in ()).throw(
            RuntimeError("synthetic explicit failure")),
    )

    with pytest.raises(RuntimeError, match="synthetic explicit failure"):
        torch_pufferl.configure_policy_and_mingru_train_scan(
            _args("metal"), _vec(), policy,
            "mps", "mps", None, True, ())
    assert "forward_train" not in policy.network.__dict__


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("bf16", "validated only for FP32"),
        ("persistent", "persistent cross-horizon"),
        ("segments", "training sequence geometry is not [1024,64]"),
    ],
)
def test_auto_guard_falls_back_without_install(monkeypatch, mutation, reason):
    _mock_validated_target(monkeypatch)
    args = _args("auto")
    amp_dtype = None
    if mutation == "bf16":
        amp_dtype = torch.bfloat16
    elif mutation == "persistent":
        args["reset_state"] = False
    else:
        args["train"]["minibatch_size"] = 32_768
    policy = _policy()

    prepared = torch_pufferl._prepare_mingru_train_scan(
        args, _vec(), policy, "mps", "mps", amp_dtype, True)

    assert prepared[0:2] == ("auto", "off")
    assert reason in prepared[2]
    assert prepared[-1] is None
    assert "forward_train" not in policy.network.__dict__


def test_policy_geometry_rejects_arbitrary_network_override_and_allows_exact(
        monkeypatch):
    policy = _policy()

    def arbitrary(network, hidden):
        return hidden

    policy.network.forward_train = MethodType(arbitrary, policy.network)
    mismatches = torch_pufferl._policy_geometry_mismatches(policy, "cpu")
    assert "actual MinGRU forward_train implementation is not validated" \
        in mismatches

    policy.network.forward_train = MethodType(
        mps_mingru.forward_train, policy.network)
    mismatches = torch_pufferl._policy_geometry_mismatches(policy, "cpu")
    assert "actual MinGRU forward_train implementation is not validated" \
        not in mismatches


def test_exact_config_key_is_scoped_to_breakout():
    from pufferlib.pufferl import load_config

    with _clean_argv():
        breakout = load_config("breakout")
        snake = load_config("snake")
    # The Metal scan remains explicit after its combined preregistered
    # learning-quality holdout failed.
    assert breakout["torch"]["mingru_train_scan"] == "off"
    assert snake["torch"]["mingru_train_scan"] == "off"
    assert "mingru_scan" not in breakout["torch"]


def test_invalid_scan_mode_is_actionable(monkeypatch):
    with pytest.raises(
            ValueError, match="mingru_train_scan must be off, auto, or metal"):
        torch_pufferl._prepare_mingru_train_scan(
            _args("fastest"), _vec(), _policy(),
            "mps", "mps", None, True)


def _portable_scan(combined, input):
    hidden, gate, projection = combined.chunk(3, dim=-1)
    log_coefficients = -F.softplus(gate)
    log_values = -F.softplus(-gate) + torch.where(
        hidden >= 0,
        (F.relu(hidden) + 0.5).log(),
        -F.softplus(-hidden),
    )
    accumulated = log_coefficients.cumsum(dim=1)
    state = (
        accumulated
        + (log_values - accumulated).logcumsumexp(dim=1)
    ).exp()
    highway = projection.sigmoid()
    return highway * state + (1.0 - highway) * input


def _relative_error(actual, expected):
    return float(
        (actual.detach().float() - expected.detach().float()).norm()
        / expected.detach().float().norm().clamp_min(1e-12))


def test_real_mps_portable_metal_forward_backward_and_allocation_parity():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        pytest.skip("requires PYTORCH_ENABLE_MPS_FALLBACK=0 before import")

    torch.manual_seed(211)
    combined = torch.randn(
        1024, 64, 192, device="mps", requires_grad=True)
    input = torch.randn(
        1024, 64, 64, device="mps", requires_grad=True)
    upstream = torch.randn_like(input)
    reference = _portable_scan(combined, input)
    reference_grads = torch.autograd.grad(
        reference, (combined, input), upstream, retain_graph=False)

    candidate = mps_mingru.mingru_train_scan(combined, input)
    candidate_grads = torch.autograd.grad(
        candidate, (combined, input), upstream, retain_graph=False)
    torch.mps.synchronize()

    assert _relative_error(candidate, reference) <= 2e-6
    assert _relative_error(candidate_grads[0], reference_grads[0]) <= 2e-5
    assert _relative_error(candidate_grads[1], reference_grads[1]) <= 2e-6
    assert candidate.is_contiguous()
    assert all(value.is_contiguous() for value in candidate_grads)

    before = torch.mps.current_allocated_memory()
    output, scan_state = mps_mingru._scan_forward(
        combined.detach(), input.detach())
    torch.mps.synchronize()
    assert torch.mps.current_allocated_memory() - before == (
        output.nbytes + scan_state.nbytes)
    before_backward = torch.mps.current_allocated_memory()
    grad_combined, grad_input = mps_mingru._scan_backward(
        combined.detach(), input.detach(), scan_state, upstream.detach())
    torch.mps.synchronize()
    assert torch.mps.current_allocated_memory() - before_backward == (
        grad_combined.nbytes + grad_input.nbytes)
    pointers = {
        value.untyped_storage().data_ptr()
        for value in (
            combined, input, output, scan_state, grad_combined, grad_input)
    }
    assert len(pointers) == 6


def test_real_mps_production_epoch_metadata_identity_and_no_recompile():
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
    args["torch"]["mingru_train_scan"] = "metal"
    args["vec"]["total_agents"] = 4096
    args["vec"]["num_buffers"] = 1
    args["vec"]["num_threads"] = 18
    args["train"]["horizon"] = 64
    args["train"]["minibatch_size"] = 65_536
    args["train"]["total_timesteps"] = 4096 * 64

    vec = _C.create_vec(args, 0)
    try:
        policy = torch_pufferl.load_policy(args, vec, device="mps")
        state_keys = tuple(policy.state_dict())
        parameter_ids = tuple(id(value) for value in policy.parameters())
        trainer = torch_pufferl.PuffeRL(
            args, vec, policy, verbose=False,
            device="mps", rollout_device="mps")
    except Exception:
        vec.close()
        raise
    try:
        assert trainer.mingru_train_scan_requested == "metal"
        assert trainer.mingru_train_scan_effective == "metal"
        assert trainer.mingru_train_scan_preflight is True
        assert trainer.mingru_train_scan_startup_seconds > 0.0
        assert trainer.policy_compile_effective == "inductor"
        assert trainer.ppo_compile_effective == "inductor"
        assert tuple(trainer.policy.state_dict()) == state_keys
        assert tuple(id(value) for value in trainer.policy.parameters()) \
            == parameter_ids
        assert tuple(
            id(value)
            for group in trainer.optimizer.param_groups
            for value in group["params"]
        ) == parameter_ids
        assert models.MinGRU.forward_train \
            is torch_pufferl._VALIDATED_MINGRU_FORWARD_TRAIN
        assert "forward_eval" not in trainer.policy.network.__dict__

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
    finally:
        trainer.close()
