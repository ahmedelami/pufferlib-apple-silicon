import importlib.util
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ppo_learning_quality_benchmark",
    _ROOT / "benchmarks" / "learning_quality.py",
)
learning = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(learning)


def _record(epoch, score):
    return {
        "epoch": epoch,
        "agent_steps": epoch * 100,
        "uptime_seconds": float(epoch),
        "epoch_seconds": 1.0,
        "episode_count": 1000.0,
        "score": float(score),
        "normalized_score": None,
        "losses": {"policy_loss": 0.1},
        "training_state": {"passed": True},
    }


def _summary(mode, seed, *, ppo):
    run = {
        "mode": mode,
        "seed": seed,
        "total_seconds": 5.0 if ppo else 5.2,
        "training_device": "mps",
        "rollout_device": "mps",
        "mps_host_alias_io": True,
        "requested_amp_dtype": "float32",
        "effective_amp_dtype": "float32",
        "requested_policy_compile": "inductor",
        "effective_policy_compile": "inductor",
        "policy_compile_preflight": True,
        "policy_compile_wrapper_verified": True,
        "policy_compile_startup_seconds": 1.0,
        "requested_ppo_compile": "inductor" if ppo else "off",
        "effective_ppo_compile": "inductor" if ppo else "off",
        "ppo_compile_preflight": ppo,
        "ppo_compile_wrapper_verified": ppo,
        "ppo_compile_startup_seconds": 0.5 if ppo else 0.0,
        "requested_rollout_sampler": "fused_mps_philox",
        "effective_rollout_sampler": "fused_mps_philox",
        "records": [_record(i + 1, score)
            for i, score in enumerate((0, 1, 2, 3))],
    }
    return learning.summarize_run(
        run, thresholds=[1], window_epochs=1,
        min_window_episodes=1, sustain_epochs=1)


def test_learning_modes_expose_clean_ppo_baseline_switch():
    common = {
        "agents": 4096,
        "horizon": 64,
        "minibatch_size": 65_536,
        "timesteps": learning.DEFAULT_TIMESTEPS,
        "threads": 18,
    }
    baseline = learning.make_args("breakout", "mps_compile", **common)
    candidate = learning.make_args("breakout", "mps_compile_ppo", **common)

    assert baseline["torch"]["compile_policy"] == "inductor"
    assert candidate["torch"]["compile_policy"] == "inductor"
    assert baseline["torch"]["compile_ppo"] == "off"
    assert candidate["torch"]["compile_ppo"] == "inductor"
    assert baseline["train"] == candidate["train"]


def test_learning_analyzer_requires_both_ppo_arm_identities():
    summaries = []
    for seed in learning.DEFAULT_SEEDS:
        summaries.extend((
            _summary("mps_compile", seed, ppo=False),
            _summary("mps_compile_ppo", seed, ppo=True),
        ))

    comparison = learning.compare_modes(
        summaries,
        thresholds=[1],
        baseline_mode="mps_compile",
        candidate_mode="mps_compile_ppo",
        min_ratio=0.9,
        bootstrap_guard=0.75,
        max_step_ratio=1.1,
        require_candidate_host_alias=True,
    )
    assert comparison["acceptance"]["passed"] is True
    assert comparison["acceptance"]["checks"][
        "baseline_ppo_compile_inactive"] is True
    assert comparison["acceptance"]["checks"][
        "candidate_ppo_compile_active"] is True

    summaries[-1]["effective_ppo_compile"] = "off"
    broken = learning.compare_modes(
        summaries,
        thresholds=[1],
        baseline_mode="mps_compile",
        candidate_mode="mps_compile_ppo",
    )
    assert broken["acceptance"]["passed"] is False
    assert broken["acceptance"]["checks"][
        "candidate_ppo_compile_active"] is False
