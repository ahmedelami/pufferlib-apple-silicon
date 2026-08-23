import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "learning_quality_benchmark",
    _ROOT / "benchmarks" / "learning_quality.py",
)
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def _record(epoch, score, episodes=1000, seconds=None):
    return {
        "epoch": epoch,
        "agent_steps": epoch * 100,
        "uptime_seconds": float(seconds if seconds is not None else epoch),
        "epoch_seconds": 1.0,
        "episode_count": float(episodes),
        "score": score,
        "normalized_score": None,
        "losses": {"policy_loss": 0.1, "value_loss": 0.2},
        "training_state": {"passed": True},
    }


def _run(mode, seed, scores, seconds=10.0):
    return {
        "mode": mode,
        "seed": seed,
        "total_seconds": seconds,
        "records": [_record(i + 1, score) for i, score in enumerate(scores)],
    }


def test_rolling_curve_is_episode_weighted_and_requires_sample_floor():
    records = [
        _record(1, 2.0, episodes=100),
        _record(2, 4.0, episodes=300),
        _record(3, 8.0, episodes=100),
    ]
    curve = benchmark.rolling_curve(
        records, window_epochs=2, min_window_episodes=350)
    assert curve[0]["score"] is None
    assert curve[1]["score"] == pytest.approx(3.5)
    assert curve[2]["score"] == pytest.approx(5.0)


def test_time_to_threshold_requires_sustained_windows():
    curve = [
        {"agent_steps": 100, "uptime_seconds": 1, "score": 3.0},
        {"agent_steps": 200, "uptime_seconds": 2, "score": 1.0},
        {"agent_steps": 300, "uptime_seconds": 3, "score": 3.0},
        {"agent_steps": 400, "uptime_seconds": 4, "score": 3.5},
    ]
    hit = benchmark.time_to_threshold(curve, 2.0, sustain_epochs=2)
    assert hit == {
        "agent_steps": 400,
        "uptime_seconds": 4.0,
        "score": 3.5,
    }


def test_summarize_run_reports_tail_auc_and_thresholds():
    run = _run("cpu", 11, [0.0, 1.0, 2.0, 3.0], seconds=8.0)
    run.update({
        "requested_rollout_sampler": "fused_mps_philox",
        "effective_rollout_sampler": "torch_multinomial",
        "rollout_sampler_reason": "outside exact compiled MPS guard",
        "rollout_sampler_startup_seconds": 0.25,
        "optimization_startup_seconds": 1.5,
        "timing_contract": "measured training clock",
        "validation_seconds_total": 0.5,
        "measured_loop_wall_seconds": 8.5,
    })
    summary = benchmark.summarize_run(
        run,
        thresholds=[1.0],
        window_epochs=1,
        min_window_episodes=1,
        sustain_epochs=2,
        tail_fraction=0.5,
    )
    assert summary["finite"] is True
    assert summary["steps_per_second"] == pytest.approx(50.0)
    assert summary["tail_score"] == pytest.approx(2.5)
    assert summary["mean_score_auc"] == pytest.approx(1.125)
    assert summary["time_to_threshold"]["1.0"]["agent_steps"] == 300
    assert summary["requested_rollout_sampler"] == "fused_mps_philox"
    assert summary["effective_rollout_sampler"] == "torch_multinomial"
    assert summary["rollout_sampler_reason"] == (
        "outside exact compiled MPS guard")
    assert summary["rollout_sampler_startup_seconds"] == pytest.approx(0.25)
    assert summary["optimization_startup_seconds"] == pytest.approx(1.5)
    assert summary["timing_contract"] == "measured training clock"
    assert summary["validation_seconds_total"] == pytest.approx(0.5)
    assert summary["measured_loop_wall_seconds"] == pytest.approx(8.5)


def test_compare_modes_accepts_equivalent_faster_candidate():
    summaries = []
    for seed in benchmark.DEFAULT_SEEDS:
        base = benchmark.summarize_run(
            _run("cpu", seed, [0, 1, 2, 3, 4], seconds=10),
            thresholds=[1], window_epochs=1, min_window_episodes=1)
        candidate = benchmark.summarize_run(
            _run("mps", seed, [0, 1.1, 2.1, 3.1, 4.1], seconds=5),
            thresholds=[1], window_epochs=1, min_window_episodes=1)
        summaries.extend((base, candidate))

    comparison = benchmark.compare_modes(
        summaries, thresholds=[1], baseline_mode="cpu", candidate_mode="mps")
    assert comparison["acceptance"]["passed"] is True
    assert comparison["aggregate"]["median_measured_training_speedup"] == 2.0


def test_compare_modes_rejects_learning_regression():
    summaries = []
    for seed in benchmark.DEFAULT_SEEDS:
        base = benchmark.summarize_run(
            _run("cpu", seed, [0, 2, 3, 4, 5], seconds=10),
            thresholds=[2], window_epochs=1, min_window_episodes=1)
        candidate = benchmark.summarize_run(
            _run("mps", seed, [0, 0.1, 0.2, 0.3, 0.4], seconds=5),
            thresholds=[2], window_epochs=1, min_window_episodes=1)
        summaries.extend((base, candidate))

    comparison = benchmark.compare_modes(
        summaries, thresholds=[2], baseline_mode="cpu", candidate_mode="mps")
    assert comparison["acceptance"]["passed"] is False
    assert comparison["acceptance"]["checks"][
        "median_tail_score_ratio"] is False
    assert comparison["acceptance"]["checks"][
        "threshold_2.0_reach_coverage"] is False


def test_compare_modes_can_require_optimized_mps_alias_path():
    summaries = []
    for seed in benchmark.DEFAULT_SEEDS:
        base = benchmark.summarize_run(
            _run("cpu", seed, [0, 1, 2, 3], seconds=10),
            thresholds=[1], window_epochs=1, min_window_episodes=1)
        candidate = benchmark.summarize_run(
            _run("mps", seed, [0, 1, 2, 3], seconds=5),
            thresholds=[1], window_epochs=1, min_window_episodes=1)
        candidate.update({
            "training_device": "mps",
            "rollout_device": "mps",
            "mps_host_alias_io": seed != benchmark.DEFAULT_SEEDS[-1],
        })
        summaries.extend((base, candidate))

    comparison = benchmark.compare_modes(
        summaries,
        thresholds=[1],
        baseline_mode="cpu",
        candidate_mode="mps",
        require_candidate_host_alias=True,
    )
    assert comparison["acceptance"]["passed"] is False
    assert comparison["acceptance"]["checks"][
        "candidate_mps_host_alias_active"] is False


def test_compiled_candidate_cannot_pass_without_observed_preflight():
    summaries = []
    for seed in benchmark.DEFAULT_SEEDS:
        eager = benchmark.summarize_run(
            _run("mps", seed, [0, 1, 2, 3], seconds=10),
            thresholds=[1], window_epochs=1, min_window_episodes=1)
        impostor = benchmark.summarize_run(
            _run("mps_compile", seed, [0, 1, 2, 3], seconds=5),
            thresholds=[1], window_epochs=1, min_window_episodes=1)
        impostor.update({
            "training_device": "mps",
            "rollout_device": "mps",
            "mps_host_alias_io": True,
            "requested_policy_compile": "inductor",
            "effective_policy_compile": "off",
            "policy_compile_preflight": False,
            "policy_compile_startup_seconds": 0.0,
        })
        summaries.extend((eager, impostor))

    comparison = benchmark.compare_modes(
        summaries,
        thresholds=[1],
        baseline_mode="mps",
        candidate_mode="mps_compile",
        require_candidate_host_alias=True,
    )
    assert comparison["acceptance"]["passed"] is False
    assert comparison["acceptance"]["checks"][
        "candidate_policy_compile_active"] is False


def test_compiled_candidate_cannot_pass_without_fused_rollout_sampler():
    summaries = []
    for seed in benchmark.DEFAULT_SEEDS:
        eager = benchmark.summarize_run(
            _run("mps", seed, [0, 1, 2, 3], seconds=10),
            thresholds=[1], window_epochs=1, min_window_episodes=1)
        candidate = benchmark.summarize_run(
            _run("mps_compile", seed, [0, 1, 2, 3], seconds=5),
            thresholds=[1], window_epochs=1, min_window_episodes=1)
        candidate.update({
            "training_device": "mps",
            "rollout_device": "mps",
            "mps_host_alias_io": True,
            "requested_policy_compile": "inductor",
            "effective_policy_compile": "inductor",
            "policy_compile_preflight": True,
            "policy_compile_wrapper_verified": True,
            "policy_compile_startup_seconds": 1.0,
            "requested_rollout_sampler": "fused_mps_philox",
            "effective_rollout_sampler": "torch_multinomial",
        })
        summaries.extend((eager, candidate))

    comparison = benchmark.compare_modes(
        summaries,
        thresholds=[1],
        baseline_mode="mps",
        candidate_mode="mps_compile",
        require_candidate_host_alias=True,
    )
    assert comparison["acceptance"]["passed"] is False
    assert comparison["acceptance"]["checks"][
        "candidate_policy_compile_active"] is True
    assert comparison["acceptance"]["checks"][
        "candidate_rollout_sampler_active"] is False


def test_compiled_bfloat16_candidate_cannot_pass_without_effective_amp():
    summaries = []
    for seed in benchmark.DEFAULT_SEEDS:
        baseline = benchmark.summarize_run(
            _run("mps_compile", seed, [0, 1, 2, 3], seconds=10),
            thresholds=[1], window_epochs=1, min_window_episodes=1)
        candidate = benchmark.summarize_run(
            _run("mps_compile_bf16", seed, [0, 1, 2, 3], seconds=5),
            thresholds=[1], window_epochs=1, min_window_episodes=1)
        candidate.update({
            "training_device": "mps",
            "rollout_device": "mps",
            "mps_host_alias_io": True,
            "requested_amp_dtype": "bfloat16",
            "effective_amp_dtype": "float32",
            "requested_policy_compile": "inductor",
            "effective_policy_compile": "inductor",
            "policy_compile_preflight": True,
            "policy_compile_wrapper_verified": True,
            "policy_compile_startup_seconds": 1.0,
            "requested_rollout_sampler": "fused_mps_philox",
            "effective_rollout_sampler": "fused_mps_philox",
        })
        summaries.extend((baseline, candidate))

    comparison = benchmark.compare_modes(
        summaries,
        thresholds=[1],
        baseline_mode="mps_compile",
        candidate_mode="mps_compile_bf16",
        require_candidate_host_alias=True,
    )
    assert comparison["acceptance"]["passed"] is False
    assert comparison["acceptance"]["checks"][
        "candidate_bfloat16_active"] is False


def test_compiled_bfloat16_candidate_identity_can_pass():
    summaries = []
    for seed in benchmark.DEFAULT_SEEDS:
        baseline = benchmark.summarize_run(
            _run("mps_compile", seed, [0, 1, 2, 3], seconds=10),
            thresholds=[1], window_epochs=1, min_window_episodes=1)
        candidate = benchmark.summarize_run(
            _run("mps_compile_bf16", seed, [0, 1, 2, 3], seconds=5),
            thresholds=[1], window_epochs=1, min_window_episodes=1)
        candidate.update({
            "training_device": "mps",
            "rollout_device": "mps",
            "mps_host_alias_io": True,
            "requested_amp_dtype": "bfloat16",
            "effective_amp_dtype": "bfloat16",
            "requested_policy_compile": "inductor",
            "effective_policy_compile": "inductor",
            "policy_compile_preflight": True,
            "policy_compile_wrapper_verified": True,
            "policy_compile_startup_seconds": 1.0,
            "requested_rollout_sampler": "fused_mps_philox",
            "effective_rollout_sampler": "fused_mps_philox",
        })
        summaries.extend((baseline, candidate))

    comparison = benchmark.compare_modes(
        summaries,
        thresholds=[1],
        baseline_mode="mps_compile",
        candidate_mode="mps_compile_bf16",
        require_candidate_host_alias=True,
    )
    assert comparison["acceptance"]["passed"] is True
    assert comparison["acceptance"]["checks"][
        "candidate_bfloat16_active"] is True


def test_make_args_preserves_training_hyperparameters_except_shape_and_device():
    args = benchmark.make_args(
        "breakout",
        "cpu",
        agents=4096,
        horizon=64,
        minibatch_size=65536,
        timesteps=benchmark.DEFAULT_TIMESTEPS,
        threads=18,
    )
    assert args["torch"]["device"] == "cpu"
    assert args["torch"]["rollout_device"] == "cpu"
    assert args["torch"]["amp_dtype"] == "float32"
    with benchmark._clean_argv():
        baseline = benchmark.load_config("breakout")
    assert args["train"]["learning_rate"] == pytest.approx(
        baseline["train"]["learning_rate"])
    assert args["train"]["replay_ratio"] == pytest.approx(
        baseline["train"]["replay_ratio"])
    assert args["train"]["total_timesteps"] == benchmark.DEFAULT_TIMESTEPS


def test_seed_everything_defaults_to_production_algorithms():
    benchmark.seed_everything(17)
    assert not benchmark.torch.are_deterministic_algorithms_enabled()
    benchmark.seed_everything(17, strict_determinism=True)
    assert benchmark.torch.are_deterministic_algorithms_enabled()
    benchmark.seed_everything(17)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("cpu", True), ("cuda", True), ("mps", False),
     ("mps_compile", False), ("mps_compile_bf16", False),
     ("hybrid", False)],
)
def test_deterministic_algorithms_are_enabled_only_on_supported_modes(
        mode, expected):
    assert benchmark.deterministic_algorithms_for_mode(mode) is expected
    assert benchmark.deterministic_algorithms_for_mode(
        mode, require_all=True) is True


def test_training_state_health_rejects_parameter_momentum_and_state_damage():
    policy = torch.nn.Linear(3, 2)
    momentum = {
        parameter: {"momentum_buffer": torch.zeros_like(parameter)}
        for parameter in policy.parameters()
    }
    trainer = SimpleNamespace(
        policy=policy,
        optimizer=SimpleNamespace(state=momentum),
        state=(torch.zeros(1, 4, 2),),
        amp_dtype=None,
    )
    assert benchmark._training_state_health(trainer)["passed"] is True

    with torch.no_grad():
        next(policy.parameters()).flatten()[0] = float("nan")
    assert benchmark._training_state_health(trainer)["passed"] is False
    with torch.no_grad():
        next(policy.parameters()).flatten()[0] = 0.0
        next(iter(momentum.values()))["momentum_buffer"].flatten()[0] = float("inf")
    assert benchmark._training_state_health(trainer)["passed"] is False
    next(iter(momentum.values()))["momentum_buffer"].zero_()
    trainer.state = (torch.zeros(1, 4, 2, dtype=torch.bfloat16),)
    assert benchmark._training_state_health(trainer)["passed"] is False


@pytest.mark.parametrize("timesteps", [0, benchmark.DEFAULT_TIMESTEPS + 1])
def test_make_args_rejects_non_exact_training_budget(timesteps):
    with pytest.raises(ValueError, match="multiple of batch size"):
        benchmark.make_args(
            "breakout",
            "cpu",
            agents=4096,
            horizon=64,
            minibatch_size=65536,
            timesteps=timesteps,
            threads=18,
        )


def test_compiled_mps_mode_uses_guarded_production_compiler():
    args = benchmark.make_args(
        "breakout",
        "mps_compile",
        agents=4096,
        horizon=64,
        minibatch_size=65536,
        timesteps=benchmark.DEFAULT_TIMESTEPS,
        threads=18,
    )
    assert args["torch"]["device"] == "mps"
    assert args["torch"]["rollout_device"] == "mps"
    assert args["torch"]["compile_policy"] == "inductor"


def test_compiled_bfloat16_mode_records_requested_precision():
    args = benchmark.make_args(
        "breakout",
        "mps_compile_bf16",
        agents=4096,
        horizon=64,
        minibatch_size=65536,
        timesteps=benchmark.DEFAULT_TIMESTEPS,
        threads=18,
    )
    assert args["torch"]["device"] == "mps"
    assert args["torch"]["rollout_device"] == "mps"
    assert args["torch"]["compile_policy"] == "inductor"
    assert args["torch"]["amp_dtype"] == "bfloat16"


def test_system_metadata_records_exact_torch_git_revision():
    metadata = benchmark._system_metadata()
    assert metadata["torch"] == benchmark.torch.__version__
    assert metadata["torch_git_version"] == benchmark.torch.version.git_version


def test_system_metadata_reuses_exact_apple_hardware_identity(monkeypatch):
    source = {
        "macos_build": "26A5378j",
        "machine_name": "MacBook Pro",
        "hardware_model": "Mac17,8",
        "chip": "Apple M5 Pro",
        "gpu_model": "Apple M5 Pro",
        "gpu_cores": 20,
        "memory_bytes": 24 * 2**30,
        "torch_git_revision": "exact-torch-git-revision",
    }
    monkeypatch.setattr(
        benchmark, "_apple_system_metadata", lambda: source)

    metadata = benchmark._system_metadata()

    assert {
        key: metadata[key]
        for key in (
            "macos_build",
            "machine_name",
            "hardware_model",
            "chip",
            "gpu_model",
            "gpu_cores",
            "memory_bytes",
        )
    } == {
        "macos_build": "26A5378j",
        "machine_name": "MacBook Pro",
        "hardware_model": "Mac17,8",
        "chip": "Apple M5 Pro",
        "gpu_model": "Apple M5 Pro",
        "gpu_cores": 20,
        "memory_bytes": 24 * 2**30,
    }
    assert metadata["torch_git_revision"] == "exact-torch-git-revision"
    assert metadata["torch_git_version"] == "exact-torch-git-revision"
    assert source == {
        "macos_build": "26A5378j",
        "machine_name": "MacBook Pro",
        "hardware_model": "Mac17,8",
        "chip": "Apple M5 Pro",
        "gpu_model": "Apple M5 Pro",
        "gpu_cores": 20,
        "memory_bytes": 24 * 2**30,
        "torch_git_revision": "exact-torch-git-revision",
    }
