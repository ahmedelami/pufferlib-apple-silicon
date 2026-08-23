import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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
    assert set(holdout.SEEDS).isdisjoint(holdout.DISCOVERY_SEEDS)
    assert holdout.TIMESTEPS == 4096 * 64 * 32


def test_bfloat16_holdout_uses_new_unobserved_seeds():
    assert holdout.BF16_SEEDS == (
        127, 131, 137, 139, 149, 151, 157, 163, 167, 173)
    assert set(holdout.BF16_SEEDS).isdisjoint(holdout.OBSERVED_SEEDS)
    assert holdout.HOLDOUTS["compile"]["seeds"] == holdout.SEEDS
    assert holdout.HOLDOUTS["bf16"]["seeds"] == holdout.BF16_SEEDS

    metadata = holdout._preregistration(
        holdout.HOLDOUTS["bf16"], holdout.BF16_SEEDS)
    assert metadata["discovery_seeds_excluded"] == list(
        holdout.DISCOVERY_SEEDS)
    assert metadata["previously_observed_seeds_excluded"] == list(
        holdout.OBSERVED_SEEDS)
    assert metadata["holdout_seeds"] == list(holdout.BF16_SEEDS)


def test_ppo_holdout_is_full_length_and_uses_fresh_fixed_seeds():
    assert holdout.PPO_SEEDS == (179, 181, 191, 193, 197)
    assert set(holdout.PPO_SEEDS).isdisjoint(
        holdout.HOLDOUTS["ppo"]["excluded_observed_seeds"])
    assert holdout.PPO_TIMESTEPS == 4096 * 64 * 358 == 93_847_552
    profile = holdout.HOLDOUTS["ppo"]
    assert profile["timesteps"] == holdout.PPO_TIMESTEPS
    assert profile["modes"] == ("mps_compile", "mps_compile_ppo")
    assert profile["min_quality_ratio"] == 0.95
    assert profile["bootstrap_guard"] == 0.90
    assert profile["max_step_ratio"] == 1.05
    registration = holdout._preregistration(profile, holdout.PPO_SEEDS)
    assert registration["declared_run_budget"] == {
        "timesteps_per_run": 93_847_552,
        "epochs": 358,
    }
    assert registration["run_budget_preregistered"] is True


def test_profile_timestep_override_is_valid_but_exploratory():
    exploratory_timesteps = 93_847_552
    assert exploratory_timesteps == 4096 * 64 * 358
    assert holdout._validated_timesteps(
        exploratory_timesteps) == exploratory_timesteps
    assert holdout.HOLDOUTS["compile"]["timesteps"] == holdout.TIMESTEPS
    assert holdout.HOLDOUTS["bf16"]["timesteps"] == holdout.TIMESTEPS


def test_bfloat16_run_budgets_are_preregistered_and_overrides_exploratory():
    short = holdout._preregistration(
        holdout.HOLDOUTS["bf16"], holdout.BF16_SEEDS)
    assert short["declared_run_budget"] == {
        "timesteps_per_run": holdout.TIMESTEPS,
        "epochs": 32,
    }
    assert short["run_budget_preregistered"] is True
    assert short["budget_status"] == "preregistered"

    exploratory = holdout._preregistration(
        holdout.HOLDOUTS["bf16"], holdout.BF16_SEEDS,
        4096 * 64 * 33)
    assert exploratory["run_budget_preregistered"] is False
    assert exploratory["budget_status"] == "exploratory CLI override"

    accepted = {"acceptance": {"passed": True}}
    assert holdout.HOLDOUTS["bf16"]["promotion_status"] == "completed_failed"
    assert holdout._promotion_eligible(
        holdout.HOLDOUTS["bf16"], holdout.TIMESTEPS, accepted) is False
    assert holdout._promotion_eligible(
        holdout.HOLDOUTS["compile"], holdout.TIMESTEPS, accepted) is True
    assert holdout._promotion_eligible(
        holdout.HOLDOUTS["bf16"], 4096 * 64 * 33, accepted) is False
    assert holdout._promotion_eligible(
        holdout.HOLDOUTS["bf16"], holdout.TIMESTEPS,
        {"acceptance": {"passed": False}}) is False


def test_bfloat16_failed_promotion_manifest_is_durable_and_locked():
    manifest = json.loads(
        (_ROOT / "benchmarks" / "bf16_holdout_result.json").read_text())
    assert manifest["schema"] == "pufferlib-bf16-promotion-result-v1"
    assert manifest["decision"] == "completed_failed"
    assert manifest["source_report"]["sha256"] == (
        "df7e50bc29de143a90805e8c0ff6943f714f9a8ec1ce05b19c83d9241acdbc40")
    assert manifest["protocol"]["paired_seeds"] == list(holdout.BF16_SEEDS)
    assert manifest["acceptance"]["passed"] is False
    assert manifest["acceptance"]["checks"][
        "tail_score_bootstrap_guard"] is False
    assert manifest["acceptance"]["checks"][
        "threshold_4.0_step_equivalence"] is False
    assert manifest["metrics"]["tail_score_ratio_bootstrap_90pct_lower"] == (
        holdout.HOLDOUTS["bf16"]["completed_result"][
            "tail_bootstrap_lower"])
    assert manifest["metrics"][
        "score_4_median_candidate_to_baseline_steps"] == (
            holdout.HOLDOUTS["bf16"]["completed_result"][
                "score4_step_ratio"])


@pytest.mark.parametrize("timesteps", [0, -1, 4096 * 64 + 1])
def test_profile_timestep_override_rejects_invalid_budget(timesteps):
    with pytest.raises(ValueError, match="positive multiple"):
        holdout._validated_timesteps(timesteps)


def test_holdout_counterbalances_fresh_process_order():
    assert holdout._mode_order(0) == ("mps", "mps_compile")
    assert holdout._mode_order(1) == ("mps_compile", "mps")
    assert holdout._mode_order(8) == ("mps", "mps_compile")
    assert holdout._mode_order(9) == ("mps_compile", "mps")


def test_bfloat16_holdout_counterbalances_compiled_precision_order():
    modes = holdout.HOLDOUTS["bf16"]["modes"]
    assert holdout._mode_order(0, modes) == (
        "mps_compile", "mps_compile_bf16")
    assert holdout._mode_order(1, modes) == (
        "mps_compile_bf16", "mps_compile")
    assert holdout.HOLDOUTS["bf16"]["baseline_mode"] == "mps_compile"
    assert holdout.HOLDOUTS["bf16"]["candidate_mode"] == (
        "mps_compile_bf16")
    assert holdout.HOLDOUTS["compile"]["output"] == holdout.OUTPUT
    assert holdout.HOLDOUTS["bf16"]["output"] == holdout.BF16_OUTPUT


def test_ppo_holdout_counterbalances_compiler_boundary_order():
    modes = holdout.HOLDOUTS["ppo"]["modes"]
    assert holdout._mode_order(0, modes) == (
        "mps_compile", "mps_compile_ppo")
    assert holdout._mode_order(1, modes) == (
        "mps_compile_ppo", "mps_compile")
    assert holdout.HOLDOUTS["ppo"]["output"] == holdout.PPO_OUTPUT


def _compiled_run():
    return {
        "requested_policy_compile": "inductor",
        "effective_policy_compile": "inductor",
        "requested_amp_dtype": "float32",
        "effective_amp_dtype": "float32",
        "policy_compile_preflight": True,
        "policy_compile_wrapper_verified": True,
        "policy_compile_startup_seconds": 0.75,
        "requested_rollout_sampler": "fused_mps_philox",
        "effective_rollout_sampler": "fused_mps_philox",
        "optimization_startup_seconds": 1.0,
        "total_seconds": 10.0,
        "mps_host_alias_io": True,
    }


def _system_provenance():
    return {
        "git_revision": "abc",
        "working_tree_patch": {"digest": "123"},
        "torch": "2.13.0",
        "torch_git_revision": "cf30153",
        "hardware_model": "Mac17,8",
        "chip": "Apple M5 Pro",
        "gpu_model": "Apple M5 Pro",
        "gpu_cores": 20,
        "memory_bytes": 24 * 2**30,
        "macos_build": "26A5378j",
        "compiled_environment": "breakout",
        "extension_gpu": False,
        "extension_precision_bytes": 4,
        "environment_variables": {
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
            "TORCHINDUCTOR_LAYOUT_OPTIMIZATION": "0",
        },
    }


def test_holdout_child_provenance_must_match_parent_and_exact_env():
    expected = _system_provenance()
    holdout._validate_child_provenance({"system": expected}, expected)

    changed = {**expected, "working_tree_patch": {"digest": "different"}}
    with pytest.raises(RuntimeError, match="working_tree_patch"):
        holdout._validate_child_provenance({"system": changed}, expected)

    changed = {
        **expected,
        "environment_variables": {
            **expected["environment_variables"],
            "TORCHINDUCTOR_LAYOUT_OPTIMIZATION": None,
        },
    }
    with pytest.raises(RuntimeError, match="LAYOUT_OPTIMIZATION"):
        holdout._validate_child_provenance({"system": changed}, expected)


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


def test_bfloat16_holdout_requires_observed_bfloat16_execution():
    run = _compiled_run()
    run.update({
        "requested_amp_dtype": "bfloat16",
        "effective_amp_dtype": "bfloat16",
    })
    holdout._validate_child_run("mps_compile_bf16", run)
    run["effective_amp_dtype"] = "float32"
    with pytest.raises(RuntimeError, match="requested bfloat16 precision"):
        holdout._validate_child_run("mps_compile_bf16", run)


def test_ppo_holdout_requires_clean_baseline_and_verified_candidate():
    baseline = _compiled_run()
    baseline.update({
        "requested_ppo_compile": "off",
        "effective_ppo_compile": "off",
        "ppo_compile_preflight": False,
        "ppo_compile_wrapper_verified": False,
        "ppo_compile_startup_seconds": 0.0,
    })
    holdout._validate_child_run("mps_compile", baseline)

    candidate = dict(baseline)
    candidate.update({
        "requested_ppo_compile": "inductor",
        "effective_ppo_compile": "inductor",
        "ppo_compile_preflight": True,
        "ppo_compile_wrapper_verified": True,
        "ppo_compile_startup_seconds": 0.5,
    })
    holdout._validate_child_run("mps_compile_ppo", candidate)
    candidate["effective_ppo_compile"] = "off"
    with pytest.raises(RuntimeError, match="PPO compiler state"):
        holdout._validate_child_run("mps_compile_ppo", candidate)


def test_compiled_holdout_requires_startup_in_total_time():
    run = _compiled_run()
    run["total_seconds"] = 0.5
    with pytest.raises(RuntimeError, match="startup in total time"):
        holdout._validate_child_run("mps_compile", run)


def test_isolated_child_uses_fresh_process_and_unique_empty_cache(monkeypatch):
    cache_dirs = []
    commands = []

    def fake_run(command, *, cwd, env, text, stdout, stderr):
        del cwd, text, stdout, stderr
        cache_dir = Path(env["TORCHINDUCTOR_CACHE_DIR"])
        assert cache_dir.is_dir()
        assert not any(cache_dir.iterdir())
        assert env["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
        assert env["TORCHINDUCTOR_LAYOUT_OPTIMIZATION"] == "0"
        cache_dirs.append(cache_dir)
        commands.append(command)
        output = Path(command[command.index("--child-output") + 1])
        mode = command[command.index("--mode") + 1]
        seed = int(command[command.index("--seed") + 1])
        timesteps = int(command[command.index("--timesteps") + 1])
        output.write_text(json.dumps({
            "mode": mode,
            "seed": seed,
            "timesteps": timesteps,
        }))
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(holdout.subprocess, "run", fake_run)
    first = holdout._run_isolated_child("mps_compile", 73)
    second = holdout._run_isolated_child(
        "mps_compile_bf16", 127, 93_847_552)

    assert first == {
        "mode": "mps_compile", "seed": 73,
        "timesteps": holdout.TIMESTEPS,
    }
    assert second == {
        "mode": "mps_compile_bf16", "seed": 127,
        "timesteps": 93_847_552,
    }
    assert len({str(path) for path in cache_dirs}) == 2
    assert all(not path.exists() for path in cache_dirs)
    assert all("--child" in command for command in commands)
