"""Parity and scoping tests for the guarded fused MPS rollout sampler."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pufferlib import torch_pufferl
from pufferlib.mps_kernels import MPSCategoricalSampler, philox_rounds


def _sampler_args(mode="auto"):
    return {
        "env_name": "breakout",
        "torch": {"compile_policy": mode},
        "train": {"horizon": 64},
    }


def _sampler_vec():
    return SimpleNamespace(total_agents=4096, act_sizes=[3], gpu=0)


class _FakeSampler:
    def __init__(self):
        self.calls = 0

    def sample(self, probs, log_probs):
        self.calls += 1
        rows, categories = probs.shape
        actions = (torch.arange(rows) % categories).to(torch.int32)
        sampled = log_probs.gather(
            -1, actions.long().unsqueeze(-1)).squeeze(-1)
        return actions, sampled


def test_philox_round_accounting_matches_flat_mps_exponential():
    assert philox_rounds(4096 * 1) == 1024
    assert philox_rounds(4096 * 3) == 3072
    assert philox_rounds(4096 * 4) == 4096
    assert philox_rounds(4096 * 17) == 17_408
    with pytest.raises(ValueError, match="positive"):
        philox_rounds(0)


def test_sampler_seam_only_intercepts_discrete_no_entropy_rollout():
    sampler = _FakeSampler()
    logits = torch.linspace(-3, 3, 33).reshape(11, 3)
    action, logprob, entropy = torch_pufferl.sample_logits(
        logits, compute_entropy=False, rollout_sampler=sampler)
    expected_log_probs = logits - logits.logsumexp(-1, keepdim=True)
    expected_actions = (torch.arange(11) % 3).to(torch.int32).reshape(-1, 1)
    expected_logprob = expected_log_probs.gather(
        -1, expected_actions).squeeze(-1)
    assert sampler.calls == 1
    torch.testing.assert_close(action, expected_actions, rtol=0, atol=0)
    torch.testing.assert_close(logprob, expected_logprob, rtol=0, atol=0)
    assert entropy is None

    # Supplied actions (the PPO training path) remain on the general sampler.
    torch_pufferl.sample_logits(
        logits, action=expected_actions, rollout_sampler=sampler)
    # Multi-discrete and continuous rollout paths also remain general.
    torch.manual_seed(7)
    torch_pufferl.sample_logits(
        (logits[:, :2], logits),
        compute_entropy=False,
        rollout_sampler=sampler,
    )
    normal = torch.distributions.Normal(
        torch.zeros(11, 2), torch.ones(11, 2))
    torch_pufferl.sample_logits(
        normal, compute_entropy=False, rollout_sampler=sampler)
    assert sampler.calls == 1


def test_sampler_auto_falls_back_but_explicit_mode_fails_closed(monkeypatch):
    import pufferlib.mps_kernels as kernels

    class BrokenSampler:
        def __init__(self, *args):
            raise RuntimeError("synthetic bridge failure")

    monkeypatch.setattr(kernels, "MPSCategoricalSampler", BrokenSampler)
    configured = torch_pufferl.configure_rollout_sampler(
        _sampler_args("auto"),
        _sampler_vec(),
        "mps",
        "mps",
        None,
        True,
        "auto",
        "inductor",
    )
    sampler, requested, effective, reason, startup = configured
    assert sampler is None
    assert requested == "fused_mps_philox"
    assert effective == "torch_multinomial"
    assert "synthetic bridge failure" in reason
    assert startup >= 0

    with pytest.raises(RuntimeError, match="failed to configure"):
        torch_pufferl.configure_rollout_sampler(
            _sampler_args("inductor"),
            _sampler_vec(),
            "mps",
            "mps",
            None,
            True,
            "inductor",
            "inductor",
        )


@pytest.mark.parametrize("amp_dtype", [None, torch.bfloat16])
def test_sampler_accepts_validated_float32_and_bfloat16_amp(
        monkeypatch, amp_dtype):
    import pufferlib.mps_kernels as kernels

    sentinel = object()
    monkeypatch.setattr(
        kernels, "MPSCategoricalSampler", lambda *args: sentinel)
    sampler, requested, effective, reason, _ = (
        torch_pufferl.configure_rollout_sampler(
            _sampler_args("inductor"),
            _sampler_vec(),
            "mps",
            "mps",
            amp_dtype,
            True,
            "inductor",
            "inductor",
        )
    )
    assert sampler is sentinel
    assert requested == "fused_mps_philox"
    assert effective == "fused_mps_philox"
    assert "validated Breakout" in reason


def test_sampler_rejects_float16_amp():
    with pytest.raises(RuntimeError, match="AMP dtype is not float32 or bfloat16"):
        torch_pufferl.configure_rollout_sampler(
            _sampler_args("inductor"),
            _sampler_vec(),
            "mps",
            "mps",
            torch.float16,
            True,
            "inductor",
            "inductor",
        )


def test_sampler_is_not_initialized_outside_compiled_path(monkeypatch):
    import pufferlib.mps_kernels as kernels

    monkeypatch.setattr(
        kernels,
        "MPSCategoricalSampler",
        lambda *args: pytest.fail("sampler should remain lazy"),
    )
    sampler, requested, effective, reason, startup = (
        torch_pufferl.configure_rollout_sampler(
            _sampler_args("auto"),
            _sampler_vec(),
            "cpu",
            "cpu",
            None,
            False,
            "auto",
            "off",
        )
    )
    assert sampler is None
    assert requested == "fused_mps_philox"
    assert effective == "torch_multinomial"
    assert "inactive" in reason
    assert startup == 0


def _validated_mps_target_available():
    if not torch.backends.mps.is_available():
        return False
    if str(torch.__version__).split("+", 1)[0] != "2.13.0" \
            or torch.version.git_version \
            != "cf30153c4c131c8164ee7798e5022d810682e2cb":
        return False
    return True


def _fixture(rows, categories, fixture_seed, nonfinite=False):
    rng = np.random.default_rng(fixture_seed)
    logits = rng.normal(size=(rows, categories)).astype(np.float32)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted).astype(np.float32)
    probs /= probs.sum(axis=-1, keepdims=True)
    log_probs = (
        logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    ).astype(np.float32)
    if nonfinite:
        flat = probs.reshape(-1)
        flat[0::11] = np.nan
        flat[1::13] = np.inf
        flat[2::17] = -np.inf
        probs[:, -1] = np.maximum(probs[:, -1], np.float32(0.125))
    return torch.from_numpy(probs), torch.from_numpy(log_probs)


@pytest.mark.skipif(
    not _validated_mps_target_available(),
    reason="validated PyTorch MPS target unavailable",
)
def test_fused_sampler_matches_multinomial_rng_and_has_no_allocator_growth():
    original_state = torch.mps.get_rng_state().clone()
    case_count = 0
    try:
        cases = []
        breakout = _fixture(4096, 3, 0xB3EA)
        breakout_seeds = (
            0, 1, 2, 3, 7, 11, 23, 37, 42, 53, 71, 127, 255, 511,
            1023, 1028, 1029, 4095, 65537, 2**31 - 1, 2**32 - 1,
            2**63 - 1, 2**63, 2**64 - 1,
        )
        cases.extend(("breakout", breakout, seed) for seed in breakout_seeds)
        for categories in (1, 4, 17):
            fixture = _fixture(4096, categories, 0xC000 + categories)
            cases.extend(
                (f"width{categories}", fixture, seed)
                for seed in (0, 1, 42, 1028, 2**32 - 1)
            )
        for categories in (3, 4, 17):
            fixture = _fixture(
                4096, categories, 0xF000 + categories, nonfinite=True)
            cases.extend(
                (f"nonfinite_width{categories}", fixture, seed)
                for seed in (0, 42, 1028)
            )

        samplers = {}
        for name, (probs_cpu, log_probs_cpu), seed in cases:
            rows, categories = probs_cpu.shape
            if categories not in samplers:
                preflight_rng = torch.mps.get_rng_state().clone()
                samplers[categories] = MPSCategoricalSampler(rows, categories)
                assert torch.equal(
                    torch.mps.get_rng_state(), preflight_rng), categories
            sampler = samplers[categories]
            probs = probs_cpu.to("mps")
            log_probs = log_probs_cpu.to("mps")
            torch.mps.manual_seed(seed)
            initial_state = torch.mps.get_rng_state().clone()
            safe = torch.nan_to_num(
                probs, nan=1e-8, posinf=1e-8, neginf=1e-8)
            expected_actions64 = torch.multinomial(
                safe, 1, replacement=True).flatten()
            expected_actions = expected_actions64.int().cpu()
            expected_logprobs = log_probs.gather(
                -1, expected_actions64.unsqueeze(-1)).flatten().cpu()
            expected_state = torch.mps.get_rng_state().clone()

            torch.mps.set_rng_state(initial_state)
            actions, sampled = sampler.sample(probs, log_probs)
            actual_actions = actions.cpu()
            actual_logprobs = sampled.cpu()
            actual_state = torch.mps.get_rng_state().clone()
            assert torch.equal(actual_actions, expected_actions), (name, seed)
            assert torch.equal(
                actual_logprobs.view(torch.int32),
                expected_logprobs.view(torch.int32),
            ), (name, seed)
            assert torch.equal(actual_state, expected_state), (name, seed)
            case_count += 1
        assert case_count == 48

        sampler = samplers[3]
        probs_cpu, log_probs_cpu = breakout
        probs = probs_cpu.to("mps")
        log_probs = log_probs_cpu.to("mps")
        for _ in range(128):
            sampler.sample(probs, log_probs)
        torch.mps.synchronize()
        gc.collect()
        current_before = torch.mps.current_allocated_memory()
        driver_before = torch.mps.driver_allocated_memory()
        for _ in range(1024):
            sampler.sample(probs, log_probs)
        torch.mps.synchronize()
        gc.collect()
        assert torch.mps.current_allocated_memory() == current_before
        assert torch.mps.driver_allocated_memory() == driver_before
    finally:
        torch.mps.set_rng_state(original_state)
        torch.mps.synchronize()


def _tensor_digest(tensor):
    if tensor.device.type == "mps":
        torch.mps.synchronize()
    cpu_tensor = tensor.detach().cpu().contiguous()
    array = (
        cpu_tensor.view(torch.uint8).numpy()
        if cpu_tensor.dtype == torch.bfloat16
        else cpu_tensor.numpy()
    )
    digest = hashlib.sha256()
    digest.update(str(cpu_tensor.dtype).encode())
    digest.update(str(tuple(cpu_tensor.shape)).encode())
    digest.update(memoryview(array))
    return digest.hexdigest()


def _state_digests(value, prefix):
    if isinstance(value, torch.Tensor):
        return {prefix: _tensor_digest(value)}
    result = {}
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            result.update(_state_digests(item, f"{prefix}/{index}"))
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            result.update(_state_digests(value[key], f"{prefix}/{key}"))
    elif value is not None:
        result[prefix] = hashlib.sha256(repr(value).encode()).hexdigest()
    return result


def _trainer_digest(trainer):
    components = {}
    for name in (
        "observations", "actions", "rewards", "terminals", "values",
        "logprobs", "ratio", "advantages",
    ):
        components[name] = _tensor_digest(getattr(trainer, name))
    components.update(_state_digests(trainer.state, "state"))
    policy = torch_pufferl.pufferlib.models._unwrap_parallel_policy(
        trainer.policy)
    components.update({
        f"policy/{name}": _tensor_digest(tensor)
        for name, tensor in policy.state_dict().items()
    })
    components.update(_state_digests(
        trainer.optimizer.state_dict(), "optimizer"))
    payload = {
        "components": components,
        "losses": {key: float(value) for key, value in trainer.losses.items()},
        "env_logs": {
            key: float(value) for key, value in trainer.env_logs.items()},
        "mps_rng": bytes(torch.mps.get_rng_state().tolist()).hex(),
        "cpu_rng": bytes(torch.get_rng_state().tolist()).hex(),
        "numpy_rng": repr(np.random.get_state()),
        "python_rng": repr(random.getstate()),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


@pytest.mark.skipif(
    not _validated_mps_target_available(),
    reason="validated PyTorch MPS target unavailable",
)
@pytest.mark.parametrize("amp_dtype", ["float32", "bfloat16"])
def test_production_fused_sampler_preserves_full_seeded_compiled_epoch(
        monkeypatch, amp_dtype):
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        pytest.skip("requires PYTORCH_ENABLE_MPS_FALLBACK=0 before import")
    from benchmarks.apple_silicon import make_args
    from pufferlib import _C
    from pufferlib.device import resolve_device, resolve_rollout_device

    if getattr(_C, "env_name", None) != "breakout":
        pytest.skip("requires the Breakout native extension")

    def run(fused):
        random.seed(1028)
        np.random.seed(1028)
        torch.manual_seed(1028)
        torch.mps.manual_seed(1028)
        args = make_args(
            "breakout", 4096, 64, 65_536, "mps", 18, amp_dtype,
            "auto", "auto")
        device = resolve_device("mps")
        vec = _C.create_vec(args, 0)
        try:
            rollout_device = resolve_rollout_device(
                "mps", device, vec_gpu=False)
            policy = torch_pufferl.load_policy(
                args, vec, device=rollout_device)
            if not fused:
                monkeypatch.setattr(
                    torch_pufferl,
                    "configure_rollout_sampler",
                    lambda *args, **kwargs: (
                        None,
                        "fused_mps_philox",
                        "torch_multinomial",
                        "test baseline",
                        0.0,
                    ),
                )
            trainer = torch_pufferl.PuffeRL(
                args, vec, policy, verbose=False,
                device=device, rollout_device=rollout_device)
        except Exception:
            vec.close()
            raise
        try:
            if trainer.policy_compile_effective != "inductor":
                pytest.skip(trainer.policy_compile_reason)
            expected_sampler = (
                "fused_mps_philox" if fused else "torch_multinomial")
            assert trainer.rollout_sampler_effective == expected_sampler
            trainer.rollouts()
            trainer.train()
            return _trainer_digest(trainer)
        finally:
            trainer.close()
            del trainer
            gc.collect()
            torch.mps.empty_cache()
            torch.mps.synchronize()

    original_configure = torch_pufferl.configure_rollout_sampler
    original_mps_rng = torch.mps.get_rng_state().clone()
    original_cpu_rng = torch.get_rng_state().clone()
    original_numpy_rng = np.random.get_state()
    original_python_rng = random.getstate()
    try:
        baseline_digest, baseline = run(False)
        monkeypatch.setattr(
            torch_pufferl, "configure_rollout_sampler", original_configure)
        fused_digest, fused = run(True)
        assert fused_digest == baseline_digest
        assert fused == baseline
    finally:
        random.setstate(original_python_rng)
        np.random.set_state(original_numpy_rng)
        torch.set_rng_state(original_cpu_rng)
        torch.mps.set_rng_state(original_mps_rng)
        torch.mps.synchronize()
