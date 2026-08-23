import importlib.util
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "compile_policy_holdout",
    _ROOT / "benchmarks" / "compile_policy_holdout.py",
)
holdout = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(holdout)


def test_holdout_seeds_are_fixed_and_disjoint_from_discovery():
    assert holdout.SEEDS == (73, 79, 83, 89, 97, 101, 103, 107, 109, 113)
    assert set(holdout.SEEDS).isdisjoint({11, 23, 37, 53, 71})
    assert holdout.TIMESTEPS == 4096 * 64 * 32


def test_holdout_counterbalances_fresh_process_order():
    assert holdout._mode_order(0) == ("mps", "mps_compile")
    assert holdout._mode_order(1) == ("mps_compile", "mps")
    assert holdout._mode_order(8) == ("mps", "mps_compile")
    assert holdout._mode_order(9) == ("mps_compile", "mps")


def _compiled_run():
    return {
        "effective_policy_compile": "inductor",
        "policy_compile_preflight": True,
        "policy_compile_wrapper_verified": True,
        "requested_rollout_sampler": "fused_mps_philox",
        "effective_rollout_sampler": "fused_mps_philox",
        "mps_host_alias_io": True,
    }


def test_holdout_requires_fused_sampler_for_compiled_child():
    run = _compiled_run()
    holdout._validate_child_run("mps_compile", run)
    run["effective_rollout_sampler"] = "torch_multinomial"
    with pytest.raises(RuntimeError, match="fused MPS sampler"):
        holdout._validate_child_run("mps_compile", run)


def test_holdout_requires_wrapper_verified_compiled_child():
    run = _compiled_run()
    run["policy_compile_wrapper_verified"] = False
    with pytest.raises(RuntimeError, match="wrappers and graph preflight"):
        holdout._validate_child_run("mps_compile", run)
