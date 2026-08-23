import importlib.util
from pathlib import Path

import pytest
import torch


_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "bf16_semantic_gate",
    _ROOT / "benchmarks" / "bf16_semantic_gate.py",
)
gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate)


def test_bfloat16_semantic_thresholds_are_fixed():
    assert gate.THRESHOLDS == {
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
    assert gate.HORIZON == 64
    assert gate.EVAL_BATCH == 4096
    assert gate.SEGMENTS == 1024


def test_bfloat16_semantic_metric_helpers_are_directional():
    reference = torch.tensor([1.0, 2.0, 3.0])
    identical = gate._vector_metrics(reference, reference.clone())
    assert identical["cosine"] == pytest.approx(1.0)
    assert identical["norm_ratio_bf16_to_fp32"] == pytest.approx(1.0)
    assert identical["rmse"] == 0.0

    perturbed = gate._nrmse(reference, reference + 0.1)
    assert perturbed["nrmse"] > 0.0
    assert perturbed["max_abs"] == pytest.approx(0.1)


def test_bfloat16_semantic_gate_fails_before_dispatch_without_fallback_guard(
        monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    with pytest.raises(RuntimeError, match="PYTORCH_ENABLE_MPS_FALLBACK"):
        gate.main()


@pytest.mark.parametrize(
    ("environment", "value"),
    (
        ("TORCHDYNAMO_DISABLE", "1"),
        ("TORCHINDUCTOR_FORCE_LAYOUT_OPT", "1"),
    ),
)
def test_bfloat16_semantic_gate_rejects_compiler_disable_switches(
        monkeypatch, environment, value):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    monkeypatch.setenv("TORCHINDUCTOR_LAYOUT_OPTIMIZATION", "0")
    monkeypatch.setenv(environment, value)
    with pytest.raises(RuntimeError, match="real Inductor execution"):
        gate.main()


def test_bfloat16_semantic_gate_rejects_noop_compile_wrapper(monkeypatch):
    policy = gate.make_policy()
    monkeypatch.setattr(gate.torch, "compile", lambda target, **kwargs: target)
    with pytest.raises(RuntimeError, match="verified Dynamo wrappers"):
        gate.compile_policy(policy)
