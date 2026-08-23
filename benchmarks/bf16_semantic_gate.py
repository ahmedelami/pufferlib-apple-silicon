"""Independent compiled-FP32 versus compiled-BF16 semantic gate on MPS.

This is a reproducible experimental-precision gate. It deliberately compares semantic metrics
instead of exact actions or elementwise equality because BF16 and compiler
lowering are not bitwise contracts. Both policies start from identical FP32
Parameters, consume deterministic CPU-originated fixtures, use supplied
actions, and run the production Breakout policy geometry.

Run with::

    PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=0 \
      TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0 \
      .venv/bin/python \
      benchmarks/bf16_semantic_gate.py
"""

from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from pufferlib import models
from pufferlib.device import synchronize
from pufferlib.muon import Muon
from pufferlib.torch_pufferl import (
    _compiler_bisect_backend,
    _float_policy_output,
    _truthy_environment,
    sample_logits,
)


OBS_SIZE = 118
NUM_ACTIONS = 3
HIDDEN_SIZE = 64
NUM_LAYERS = 2
EVAL_BATCH = 4096
SEGMENTS = 1024
HORIZON = 64
POLICY_SEED = 1_357_911
FIXTURE_SEED = 24_681_357
OUTPUT = ROOT / "work" / "bf16_cross_precision_semantic.json"

LEARNING_RATE = 0.1
MOMENTUM = 0.7279714073125252
MUON_EPS = 8.339460257113628e-05
MAX_GRAD_NORM = 1.8109182724544075
CLIP_COEF = 0.6746497927896418
VF_CLIP = 1.2291681640124468
VF_COEF = 1.2195502588297364
ENT_COEF = 0.0033240721522812535

THRESHOLDS = {
    "categorical_kl_mean_max": 1e-3,
    "categorical_kl_p99_max": 1e-2,
    "step64_value_nrmse_max": 0.02,
    "step64_state_nrmse_max": 0.02,
    "gradient_cosine_min": 0.99,
    "gradient_norm_ratio_min": 0.90,
    "gradient_norm_ratio_max": 1.10,
    "muon_update_cosine_min": 0.99,
    "muon_update_norm_ratio_min": 0.90,
    "muon_update_norm_ratio_max": 1.10,
}


def make_policy():
    return models.Policy(
        models.DefaultEncoder(OBS_SIZE, HIDDEN_SIZE),
        models.DefaultDecoder((NUM_ACTIONS,), HIDDEN_SIZE),
        models.MinGRU(HIDDEN_SIZE, NUM_LAYERS),
    )


def _rand(shape, generator):
    return torch.rand(shape, generator=generator, dtype=torch.float32)


def cpu_fixtures():
    """Create 64 distinct, deterministic, Breakout-shaped trajectories."""
    generator = torch.Generator(device="cpu").manual_seed(FIXTURE_SEED)
    observations = torch.empty(
        HORIZON, EVAL_BATCH, OBS_SIZE, dtype=torch.float32)

    # Breakout's first ten features are normalized continuous/discrete state.
    observations[..., 0] = 0.90 * _rand((HORIZON, EVAL_BATCH), generator)
    observations[..., 1] = 0.88 + 0.04 * _rand(
        (HORIZON, EVAL_BATCH), generator)
    observations[..., 2:4] = _rand(
        (HORIZON, EVAL_BATCH, 2), generator)
    observations[..., 4:6] = 1.75 * _rand(
        (HORIZON, EVAL_BATCH, 2), generator) - 0.875
    observations[..., 6] = torch.randint(
        0, 6, (HORIZON, EVAL_BATCH), generator=generator).float() / 5.0
    observations[..., 8] = torch.randint(
        0, 5, (HORIZON, EVAL_BATCH), generator=generator).float() / 5.0
    observations[..., 9] = torch.where(
        _rand((HORIZON, EVAL_BATCH), generator) < 0.15,
        torch.tensor(0.5),
        torch.tensor(1.0),
    )

    # The final 108 features are binary brick states. Later steps contain more
    # destroyed bricks, while every time slice remains independently generated.
    destroy_probability = torch.linspace(
        0.01, 0.55, HORIZON, dtype=torch.float32).view(HORIZON, 1, 1)
    brick_states = (
        _rand((HORIZON, EVAL_BATCH, OBS_SIZE - 10), generator)
        < destroy_probability
    ).float()
    observations[..., 10:] = brick_states
    observations[..., 7] = (
        brick_states.mean(dim=-1) * 0.85
        + 0.02 * _rand((HORIZON, EVAL_BATCH), generator)
    ).clamp_max(1.0)

    train_observations = observations[:, :SEGMENTS].transpose(0, 1).contiguous()
    segment = torch.arange(SEGMENTS, dtype=torch.int64).view(SEGMENTS, 1)
    timestep = torch.arange(HORIZON, dtype=torch.int64).view(1, HORIZON)
    actions = ((segment + 2 * timestep) % NUM_ACTIONS).unsqueeze(-1).float()
    old_logprobs = -math.log(NUM_ACTIONS) + 0.05 * torch.randn(
        SEGMENTS, HORIZON, generator=generator)
    old_values = 0.25 * torch.randn(
        SEGMENTS, HORIZON, generator=generator)
    returns = old_values + 0.20 * torch.randn(
        SEGMENTS, HORIZON, generator=generator)
    advantages = torch.randn(SEGMENTS, HORIZON, generator=generator)
    priority = torch.linspace(0.75, 1.25, SEGMENTS).unsqueeze(1)

    observation_slice_digests = {
        hashlib.sha256(memoryview(observations[index].numpy())).hexdigest()
        for index in range(HORIZON)
    }
    all_slices_distinct = len(observation_slice_digests) == HORIZON
    return {
        "eval_observations": observations,
        "train_observations": train_observations,
        "actions": actions,
        "old_logprobs": old_logprobs,
        "old_values": old_values,
        "returns": returns,
        "advantages": advantages,
        "priority": priority,
    }, all_slices_distinct


def _digest(tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(memoryview(tensor.numpy()))
    return digest.hexdigest()


def _autocast(amp_dtype):
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast("mps", dtype=amp_dtype)


def _cast_state(state, amp_dtype):
    if amp_dtype is None:
        return state
    return tuple(
        value.to(amp_dtype) if value.is_floating_point() else value
        for value in state
    )


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


def compile_policy(policy):
    options = {"layout_optimization": False}
    compiled_eval = torch.compile(
        policy.forward_eval,
        backend="inductor",
        fullgraph=True,
        dynamic=False,
        options=options,
    )
    compiled_train = torch.compile(
        policy.forward,
        backend="inductor",
        fullgraph=True,
        dynamic=False,
        options=options,
    )
    if not (
            callable(compiled_eval)
            and callable(compiled_train)
            and getattr(
                compiled_eval, "_torchdynamo_orig_callable", None) is not None
            and getattr(
                compiled_train, "_torchdynamo_orig_callable", None) is not None):
        raise RuntimeError(
            "torch.compile did not return verified Dynamo wrappers")
    policy.forward_eval = compiled_eval
    policy.forward = compiled_train
    return policy


def _validate_compiler_runtime():
    """Reject compiler switches that can turn this gate into eager execution."""
    mismatches = []
    if os.environ.get("TORCHINDUCTOR_FORCE_LAYOUT_OPT", "0") == "1":
        mismatches.append("TORCHINDUCTOR_FORCE_LAYOUT_OPT is enabled")
    if bool(getattr(torch._dynamo.config, "suppress_errors", False)):
        mismatches.append("torch._dynamo.config.suppress_errors is enabled")
    if bool(getattr(torch._dynamo.config, "disable", False)):
        mismatches.append("torch._dynamo.config.disable is enabled")
    if _truthy_environment("TORCHDYNAMO_DISABLE"):
        mismatches.append("TORCHDYNAMO_DISABLE is enabled")
    if _compiler_bisect_backend() is not None:
        mismatches.append("Torch compiler bisector backend override is active")
    if mismatches:
        raise RuntimeError(
            "BF16 semantic gate requires a real Inductor execution: "
            + "; ".join(mismatches))


def warm_compile(policy, fixtures, amp_dtype):
    policy.eval()
    state = _cast_state(
        policy.initial_state(EVAL_BATCH, "mps"), amp_dtype)
    with torch.no_grad(), _autocast(amp_dtype):
        policy.forward_eval(fixtures["eval_observations"][0], state)
    policy.train()
    with _autocast(amp_dtype):
        logits, values = policy(fixtures["train_observations"])
    (logits.float().sum() + values.float().sum()).backward()
    synchronize("mps")
    policy.zero_grad(set_to_none=True)


def reset_policy(policy, initial_state_dict):
    policy.load_state_dict(initial_state_dict, strict=True)
    policy.zero_grad(set_to_none=True)


def eval_horizon(policy, observations, amp_dtype):
    """Mirror production: eval mode and a fresh autocast context per step."""
    policy.eval()
    state = _cast_state(
        policy.initial_state(EVAL_BATCH, "mps"), amp_dtype)
    initial_state_dtypes = sorted({str(value.dtype) for value in state})
    logits_horizon = []
    values_horizon = []
    raw_logits_dtypes = set()
    raw_value_dtypes = set()
    raw_state_dtypes = set()
    with torch.no_grad():
        for timestep in range(HORIZON):
            with _autocast(amp_dtype):
                logits, values, state = policy.forward_eval(
                    observations[timestep], state)
            raw_logits_dtypes.add(str(logits.dtype))
            raw_value_dtypes.add(str(values.dtype))
            raw_state_dtypes.update(
                str(value.dtype) for value in _state_tensors(state))
            logits_horizon.append(logits.float())
            values_horizon.append(values.float())
    logits_horizon = torch.stack(logits_horizon)
    values_horizon = torch.stack(values_horizon)
    synchronize("mps")
    state_cpu = tuple(
        value.detach().float().cpu() for value in _state_tensors(state))
    result = {
        "logits": logits_horizon.detach().cpu(),
        "values": values_horizon.detach().cpu(),
        "state": state_cpu,
        "dtypes": {
            "initial_state": initial_state_dtypes,
            "raw_logits": sorted(raw_logits_dtypes),
            "raw_values": sorted(raw_value_dtypes),
            "raw_state": sorted(raw_state_dtypes),
        },
    }
    result["finite"] = all(
        bool(torch.isfinite(value).all())
        for value in (
            result["logits"], result["values"], *result["state"])
    )
    return result


def ppo_loss(policy, fixtures, amp_dtype):
    with _autocast(amp_dtype):
        raw_logits, raw_values = policy(fixtures["train_observations"])
    logits = _float_policy_output(raw_logits)
    values = raw_values.float()
    _, logprobs, entropy = sample_logits(
        logits, action=fixtures["actions"], compute_entropy=True)
    logprobs = logprobs.reshape(fixtures["old_logprobs"].shape)
    logratio = logprobs - fixtures["old_logprobs"]
    ratio = logratio.exp()

    advantages = fixtures["advantages"]
    advantages = fixtures["priority"] * (
        advantages - advantages.mean()) / (advantages.std() + 1e-8)
    policy_loss = torch.maximum(
        -advantages * ratio,
        -advantages * torch.clamp(
            ratio, 1 - CLIP_COEF, 1 + CLIP_COEF),
    ).mean()

    clipped_values = fixtures["old_values"] + torch.clamp(
        values - fixtures["old_values"], -VF_CLIP, VF_CLIP)
    value_loss = 0.5 * torch.maximum(
        (values - fixtures["returns"]).square(),
        (clipped_values - fixtures["returns"]).square(),
    ).mean()
    entropy_loss = entropy.mean()
    loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy_loss
    return {
        "raw_logits": raw_logits,
        "raw_values": raw_values,
        "logits": logits,
        "values": values,
        "logprobs": logprobs,
        "entropy": entropy,
        "loss": loss,
    }


def _flatten_named_tensors(named_tensors):
    return torch.cat([
        value.detach().reshape(-1).float().cpu()
        for _, value in named_tensors
    ])


def train_and_step(policy, fixtures, amp_dtype, initial_state_dict):
    policy.train()
    optimizer = Muon(
        policy.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        eps=MUON_EPS,
    )
    outputs = ppo_loss(policy, fixtures, amp_dtype)
    outputs["loss"].backward()
    synchronize("mps")

    named_parameters = tuple(policy.named_parameters())
    gradient_dtypes = sorted({
        str(parameter.grad.dtype) for _, parameter in named_parameters
        if parameter.grad is not None
    })
    raw_gradient = _flatten_named_tensors(
        (name, parameter.grad) for name, parameter in named_parameters)
    grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
        policy.parameters(), MAX_GRAD_NORM)
    optimizer.step()
    synchronize("mps")

    updates = {}
    for name, parameter in named_parameters:
        updates[name] = (
            parameter.detach().float().cpu() - initial_state_dict[name].float())
    update_vector = _flatten_named_tensors(updates.items())
    momentum_buffers = tuple(
        state["momentum_buffer"]
        for state in optimizer.state.values()
        if "momentum_buffer" in state
    )
    parameter_dtypes = sorted({
        str(parameter.dtype) for _, parameter in named_parameters})
    momentum_dtypes = sorted({
        str(value.dtype) for value in momentum_buffers})

    finite_tensors = (
        outputs["logits"], outputs["values"], outputs["logprobs"],
        outputs["entropy"], outputs["loss"], raw_gradient, update_vector,
        *(parameter for _, parameter in named_parameters),
        *momentum_buffers,
    )
    finite = all(bool(torch.isfinite(value).all()) for value in finite_tensors)
    return {
        "gradient": raw_gradient,
        "updates": updates,
        "update_vector": update_vector,
        "finite": finite,
        "loss": float(outputs["loss"].detach().cpu()),
        "grad_norm_before_clip": float(grad_norm_before_clip.detach().cpu()),
        "zero_fraction": float((raw_gradient == 0).float().mean()),
        "dtypes": {
            "parameters": parameter_dtypes,
            "raw_logits": str(outputs["raw_logits"].dtype),
            "raw_values": str(outputs["raw_values"].dtype),
            "ppo_logits": str(outputs["logits"].dtype),
            "ppo_values": str(outputs["values"].dtype),
            "loss": str(outputs["loss"].dtype),
            "gradients": gradient_dtypes,
            "momentum": momentum_dtypes,
        },
        "optimizer": {
            "parameter_count": len(named_parameters),
            "momentum_buffer_count": len(momentum_buffers),
        },
    }


def _vector_metrics(reference, candidate):
    reference = reference.double().reshape(-1)
    candidate = candidate.double().reshape(-1)
    reference_norm = reference.norm()
    candidate_norm = candidate.norm()
    denominator = float(reference_norm)
    if denominator == 0.0 or float(candidate_norm) == 0.0:
        cosine = float("nan")
        ratio = float("nan") if denominator == 0.0 else 0.0
    else:
        cosine = float(torch.dot(reference, candidate) /
            (reference_norm * candidate_norm))
        ratio = float(candidate_norm / reference_norm)
    difference = candidate - reference
    return {
        "cosine": cosine,
        "norm_ratio_bf16_to_fp32": ratio,
        "fp32_norm": float(reference_norm),
        "bf16_norm": float(candidate_norm),
        "rmse": float(difference.square().mean().sqrt()),
        "max_abs": float(difference.abs().max()),
    }


def _nrmse(reference, candidate):
    reference = reference.double().reshape(-1)
    candidate = candidate.double().reshape(-1)
    rmse = (candidate - reference).square().mean().sqrt()
    reference_rms = reference.square().mean().sqrt()
    value = float(rmse / reference_rms.clamp_min(1e-12))
    return {
        "nrmse": value,
        "rmse": float(rmse),
        "fp32_rms": float(reference_rms),
        "max_abs": float((candidate - reference).abs().max()),
    }


def _dtype_contract(eval_result, train_result, expected_raw_dtype):
    expected = f"torch.{expected_raw_dtype}"
    checks = {
        "eval_initial_state": eval_result["dtypes"]["initial_state"] == [expected],
        "eval_raw_logits": eval_result["dtypes"]["raw_logits"] == [expected],
        "eval_raw_values": eval_result["dtypes"]["raw_values"] == [expected],
        "eval_raw_state": eval_result["dtypes"]["raw_state"] == [expected],
        "train_raw_logits": train_result["dtypes"]["raw_logits"] == expected,
        "train_raw_values": train_result["dtypes"]["raw_values"] == expected,
        "ppo_logits_fp32": train_result["dtypes"]["ppo_logits"] == "torch.float32",
        "ppo_values_fp32": train_result["dtypes"]["ppo_values"] == "torch.float32",
        "loss_fp32": train_result["dtypes"]["loss"] == "torch.float32",
        "parameters_fp32": train_result["dtypes"]["parameters"] == ["torch.float32"],
        "gradients_fp32": train_result["dtypes"]["gradients"] == ["torch.float32"],
        "momentum_fp32": train_result["dtypes"]["momentum"] == ["torch.float32"],
        "momentum_complete": (
            train_result["optimizer"]["momentum_buffer_count"]
            == train_result["optimizer"]["parameter_count"]
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def build_report(fp32_eval, bf16_eval, fp32_train, bf16_train,
        fixture_metadata):
    fp32_logprobs = F.log_softmax(fp32_eval["logits"].double(), dim=-1)
    bf16_logprobs = F.log_softmax(bf16_eval["logits"].double(), dim=-1)
    categorical_kl = (
        fp32_logprobs.exp() * (fp32_logprobs - bf16_logprobs)
    ).sum(dim=-1).clamp_min(0.0)
    kl_metrics = {
        "mean": float(categorical_kl.mean()),
        "p99": float(torch.quantile(categorical_kl, 0.99)),
        "max": float(categorical_kl.max()),
        "sample_count": categorical_kl.numel(),
    }

    value_metrics = _nrmse(
        fp32_eval["values"][-1], bf16_eval["values"][-1])
    fp32_state = torch.cat([value.reshape(-1) for value in fp32_eval["state"]])
    bf16_state = torch.cat([value.reshape(-1) for value in bf16_eval["state"]])
    state_metrics = _nrmse(fp32_state, bf16_state)
    gradient_metrics = _vector_metrics(
        fp32_train["gradient"], bf16_train["gradient"])
    gradient_metrics.update({
        "fp32_zero_fraction": fp32_train["zero_fraction"],
        "bf16_zero_fraction": bf16_train["zero_fraction"],
        "zero_fraction_delta": (
            bf16_train["zero_fraction"] - fp32_train["zero_fraction"]),
        "fp32_norm_before_clip": fp32_train["grad_norm_before_clip"],
        "bf16_norm_before_clip": bf16_train["grad_norm_before_clip"],
        "clip_max_norm": MAX_GRAD_NORM,
    })
    update_metrics = _vector_metrics(
        fp32_train["update_vector"], bf16_train["update_vector"])

    per_parameter_updates = {}
    for name in fp32_train["updates"]:
        per_parameter_updates[name] = _vector_metrics(
            fp32_train["updates"][name], bf16_train["updates"][name])

    fp32_dtype = _dtype_contract(fp32_eval, fp32_train, "float32")
    bf16_dtype = _dtype_contract(bf16_eval, bf16_train, "bfloat16")
    finite = bool(
        fp32_eval["finite"] and bf16_eval["finite"]
        and fp32_train["finite"] and bf16_train["finite"]
        and all(math.isfinite(value) for value in (
            kl_metrics["mean"], kl_metrics["p99"], value_metrics["nrmse"],
            state_metrics["nrmse"], gradient_metrics["cosine"],
            gradient_metrics["norm_ratio_bf16_to_fp32"],
            update_metrics["cosine"],
            update_metrics["norm_ratio_bf16_to_fp32"],
        ))
    )
    checks = {
        "all_finite": finite,
        "dtype_contracts": fp32_dtype["passed"] and bf16_dtype["passed"],
        "categorical_kl_mean": (
            kl_metrics["mean"] <= THRESHOLDS["categorical_kl_mean_max"]),
        "categorical_kl_p99": (
            kl_metrics["p99"] <= THRESHOLDS["categorical_kl_p99_max"]),
        "step64_value_nrmse": (
            value_metrics["nrmse"] <= THRESHOLDS["step64_value_nrmse_max"]),
        "step64_state_nrmse": (
            state_metrics["nrmse"] <= THRESHOLDS["step64_state_nrmse_max"]),
        "gradient_cosine": (
            gradient_metrics["cosine"] >= THRESHOLDS["gradient_cosine_min"]),
        "gradient_norm_ratio": (
            THRESHOLDS["gradient_norm_ratio_min"]
            <= gradient_metrics["norm_ratio_bf16_to_fp32"]
            <= THRESHOLDS["gradient_norm_ratio_max"]),
        "muon_update_cosine": (
            update_metrics["cosine"]
            >= THRESHOLDS["muon_update_cosine_min"]),
        "muon_update_norm_ratio": (
            THRESHOLDS["muon_update_norm_ratio_min"]
            <= update_metrics["norm_ratio_bf16_to_fp32"]
            <= THRESHOLDS["muon_update_norm_ratio_max"]),
    }
    return {
        "schema": "pufferlib-bf16-cross-precision-semantic-v1",
        "comparison": "compiled FP32 vs compiled BF16 autocast",
        "pass": all(checks.values()),
        "checks": checks,
        "thresholds_preregistered": THRESHOLDS,
        "fixture": fixture_metadata,
        "system": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_git": torch.version.git_version,
            "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "layout_optimization_env": os.environ.get(
                "TORCHINDUCTOR_LAYOUT_OPTIMIZATION"),
        },
        "metrics": {
            "categorical_kl": kl_metrics,
            "step64_value": value_metrics,
            "step64_recurrent_state": state_metrics,
            "flattened_gradient": gradient_metrics,
            "clipped_muon_update": update_metrics,
            "per_parameter_muon_updates": per_parameter_updates,
            "loss": {
                "fp32": fp32_train["loss"],
                "bf16": bf16_train["loss"],
                "absolute_difference": abs(
                    bf16_train["loss"] - fp32_train["loss"]),
            },
        },
        "dtype_contracts": {
            "fp32": fp32_dtype,
            "bf16": bf16_dtype,
            "observed": {
                "fp32_eval": fp32_eval["dtypes"],
                "bf16_eval": bf16_eval["dtypes"],
                "fp32_train": fp32_train["dtypes"],
                "bf16_train": bf16_train["dtypes"],
            },
        },
    }


def main():
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must be 0")
    if os.environ.get("TORCHINDUCTOR_LAYOUT_OPTIMIZATION") != "0":
        raise RuntimeError("TORCHINDUCTOR_LAYOUT_OPTIMIZATION must be 0")
    _validate_compiler_runtime()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(POLICY_SEED)
    template = make_policy()
    initial_state_dict = {
        name: value.detach().clone()
        for name, value in template.state_dict().items()
    }
    fp32_policy = compile_policy(deepcopy(template).to("mps"))
    bf16_policy = compile_policy(deepcopy(template).to("mps"))

    fixtures_cpu, all_slices_distinct = cpu_fixtures()
    fixture_metadata = {
        "policy_seed": POLICY_SEED,
        "fixture_seed": FIXTURE_SEED,
        "shape": {
            "eval_observations": list(
                fixtures_cpu["eval_observations"].shape),
            "train_observations": list(
                fixtures_cpu["train_observations"].shape),
            "actions": list(fixtures_cpu["actions"].shape),
        },
        "all_64_observation_slices_pairwise_distinct": all_slices_distinct,
        "eval_observation_sha256": _digest(
            fixtures_cpu["eval_observations"]),
        "supplied_action_sha256": _digest(fixtures_cpu["actions"]),
        "sampling_used": False,
    }
    if not all_slices_distinct:
        raise RuntimeError("the fixed observation horizon is not distinct")
    fixtures = {
        name: value.to("mps") for name, value in fixtures_cpu.items()
    }
    del fixtures_cpu

    # Materialize eval/train/backward graphs in their production autocast and
    # mode contexts, then restore exactly identical FP32 Parameters.
    warm_compile(fp32_policy, fixtures, None)
    warm_compile(bf16_policy, fixtures, torch.bfloat16)
    reset_policy(fp32_policy, initial_state_dict)
    reset_policy(bf16_policy, initial_state_dict)

    fp32_eval = eval_horizon(
        fp32_policy, fixtures["eval_observations"], None)
    bf16_eval = eval_horizon(
        bf16_policy, fixtures["eval_observations"], torch.bfloat16)
    fp32_train = train_and_step(
        fp32_policy, fixtures, None, initial_state_dict)
    bf16_train = train_and_step(
        bf16_policy, fixtures, torch.bfloat16, initial_state_dict)
    report = build_report(
        fp32_eval, bf16_eval, fp32_train, bf16_train, fixture_metadata)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUTPUT)
    print(json.dumps({
        "output": str(OUTPUT),
        "pass": report["pass"],
        "checks": report["checks"],
        "metrics": {
            "categorical_kl": report["metrics"]["categorical_kl"],
            "step64_value": report["metrics"]["step64_value"],
            "step64_recurrent_state": report["metrics"][
                "step64_recurrent_state"],
            "flattened_gradient": report["metrics"]["flattened_gradient"],
            "clipped_muon_update": report["metrics"]["clipped_muon_update"],
        },
    }, indent=2, sort_keys=True), flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
