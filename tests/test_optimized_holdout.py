import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "optimized_holdout_benchmark",
    _ROOT / "benchmarks" / "optimized_holdout.py",
)
holdout = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(holdout)


def _protocol():
    return holdout.load_protocol()[0]


def _fake_run(mode, seed=233):
    protocol = _protocol()
    shape = protocol["protocol"]
    candidate = mode == holdout.CANDIDATE_MODE
    ppo_compile = "inductor" if candidate else "off"
    mingru_scan = "metal" if candidate else "off"
    batch_size = int(shape["agents"]) * int(shape["horizon"])
    records = [
        {
            "epoch": epoch,
            "agent_steps": epoch * batch_size,
            "uptime_seconds": float(epoch),
            "epoch_seconds": 1.0,
            "episode_count": 2000.0,
            "score": 3.0,
            "normalized_score": None,
            "losses": {
                "policy_loss": 0.1,
                "value_loss": 0.2,
                "entropy_loss": -0.01,
            },
            "training_state": {"passed": True},
        }
        for epoch in range(1, int(shape["epochs"]) + 1)
    ]
    run = {
        "environment": protocol["environment"],
        "mode": mode,
        "seed": int(seed),
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
        "requested_rollout_sampler": "fused_mps_philox",
        "effective_rollout_sampler": "fused_mps_philox",
        "rollout_sampler_startup_seconds": 0.25,
        "requested_ppo_compile": ppo_compile,
        "effective_ppo_compile": ppo_compile,
        "ppo_compile_preflight": candidate,
        "ppo_compile_wrapper_verified": candidate,
        "ppo_compile_startup_seconds": 0.75 if candidate else 0.0,
        "requested_mingru_train_scan": mingru_scan,
        "effective_mingru_train_scan": mingru_scan,
        "mingru_train_scan_preflight": candidate,
        "mingru_train_scan_startup_seconds": 0.5 if candidate else 0.0,
        "optimization_startup_seconds": 2.5 if candidate else 1.25,
        "post_preflight_dynamo_frames_total": 0,
        "post_preflight_dynamo_unique_graphs": 0,
        "validation_seconds_total": 0.5,
        "measured_loop_wall_seconds": 10.5,
        "total_seconds": 10.0,
        "effective_config": {
            "torch": {
                "device": "mps",
                "rollout_device": "mps",
                "amp_dtype": "float32",
                "compile_policy": "inductor",
                "compile_ppo": ppo_compile,
                "mingru_train_scan": mingru_scan,
                "mps_host_alias": "auto",
            },
            "train": {
                "horizon": int(shape["horizon"]),
                "minibatch_size": int(shape["minibatch_size"]),
                "total_timesteps": int(shape["timesteps_per_run"]),
                "learning_rate": 0.001,
            },
            "vec": {
                "total_agents": int(shape["agents"]),
                "num_threads": int(shape["environment_threads"]),
                "num_buffers": 1,
            },
            "env": {"name": "breakout"},
            "policy": {"hidden_size": 128},
        },
        "records": records,
        "summary": {"finite": True},
    }
    run["identity"] = holdout.run_identity_report(
        run, protocol, expected_mode=mode, expected_seed=seed)
    return run


def _passing_comparison():
    seeds = list(holdout.EXPECTED_SEEDS)
    return {
        "acceptance": {"passed": True},
        "aggregate": {
            "median_tail_score_ratio": 0.99,
            "tail_score_ratio_bootstrap_90pct_lower": 0.96,
            "median_mean_score_auc_ratio": 0.98,
            "mean_score_auc_ratio_bootstrap_90pct_lower": 0.95,
            "median_measured_training_speedup": 1.02,
        },
        "thresholds": {
            "2.0": {
                "gated": True,
                "baseline_reached_seeds": seeds,
                "candidate_reached_seeds": seeds[:-1],
                "median_candidate_to_baseline_steps": 1.04,
            },
            "4.0": {
                "gated": False,
                "baseline_reached_seeds": [],
                "candidate_reached_seeds": [],
                "median_candidate_to_baseline_steps": None,
            },
        },
    }


def test_canonical_protocol_digest_shape_and_abba_order_are_fixed():
    protocol, digest = holdout.load_protocol()
    assert digest == holdout.EXPECTED_PROTOCOL_SHA256
    assert digest == (
        "fd7f562c84594a6126501e636d136a0182ce357160ddef1c62ecf88362be25eb")
    assert protocol["paired_seeds"] == [233, 239, 241, 251, 257]
    assert protocol["protocol"]["epochs"] == 358
    assert protocol["protocol"]["timesteps_per_run"] == 93_847_552
    assert holdout.expected_run_order(protocol) == [
        {"seed": 233, "mode": holdout.BASELINE_MODE},
        {"seed": 233, "mode": holdout.CANDIDATE_MODE},
        {"seed": 239, "mode": holdout.CANDIDATE_MODE},
        {"seed": 239, "mode": holdout.BASELINE_MODE},
        {"seed": 241, "mode": holdout.BASELINE_MODE},
        {"seed": 241, "mode": holdout.CANDIDATE_MODE},
        {"seed": 251, "mode": holdout.CANDIDATE_MODE},
        {"seed": 251, "mode": holdout.BASELINE_MODE},
        {"seed": 257, "mode": holdout.BASELINE_MODE},
        {"seed": 257, "mode": holdout.CANDIDATE_MODE},
    ]


def test_published_result_manifest_records_known_terminal_failure():
    result = json.loads(holdout.RESULT_MANIFEST_PATH.read_text())
    assert result["schema"] == holdout.EXPECTED_RESULT_SCHEMA
    assert result["completion_status"] == "completed_failed"
    assert result["promotion_eligible"] is False
    assert result["source_report"]["sha256"] == (
        "c7bbdcfc3d00e527a693c94e1a5dc47d7499db9ab6a9d9046dc9b0b270b69873")
    assert result["protocol"]["sha256"] == (
        "fd7f562c84594a6126501e636d136a0182ce357160ddef1c62ecf88362be25eb")
    assert result["protocol"]["paired_seeds"] == [233, 239, 241, 251, 257]
    assert result["protocol"]["complete_run_count"] == 10
    assert result["execution_identity"][
        "complete_run_identity_pass_count"] == 10
    assert result["acceptance"]["passed"] is False
    assert result["acceptance"]["failed_checks"] == [
        "established_analyzer_acceptance",
        "learning_auc_bootstrap_90pct_lower",
        "median_learning_auc_ratio",
        "median_steps_to_score_ratio",
        "tail_score_bootstrap_90pct_lower",
    ]
    assert holdout.load_published_result() == result


def test_published_terminal_decision_blocks_parent_and_child_reruns(
        tmp_path):
    match = "published optimized holdout decision is immutable"
    with pytest.raises(RuntimeError, match=match):
        holdout.parent_run(
            output=tmp_path / "new-report.json",
            log_dir=tmp_path / "logs")
    with pytest.raises(RuntimeError, match=match):
        holdout.child_run(
            holdout.BASELINE_MODE, holdout.EXPECTED_SEEDS[0],
            tmp_path / "child.json")
    assert not (tmp_path / "new-report.json").exists()
    assert not (tmp_path / "child.json").exists()


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        (None, "preregistered_at", "2026-08-24"),
        (None, "paired_seeds", [233, 239, 241, 251, 259]),
        ("baseline", "ppo_compile", "inductor"),
        ("candidate", "mingru_train_scan", "off"),
        ("protocol", "epochs", 357),
        ("acceptance", "median_tail_score_ratio_min", 0.96),
        (None, "notes", ["changed", "but", "same length"]),
    ],
)
def test_protocol_semantic_drift_fails_closed(section, key, value):
    protocol = copy.deepcopy(_protocol())
    target = protocol if section is None else protocol[section]
    target[key] = value
    with pytest.raises(ValueError, match="drifted|disagree"):
        holdout.validate_protocol(protocol)


def test_protocol_unknown_field_fails_closed():
    protocol = copy.deepcopy(_protocol())
    protocol["override"] = True
    with pytest.raises(ValueError, match="schema drifted"):
        holdout.validate_protocol(protocol)


def test_validate_only_maps_exact_baseline_and_candidate(capsys):
    result = holdout.validate_only()
    assert result["valid"] is True
    assert result["mode_config"] == {
        "baseline": {
            "compile_policy": "inductor",
            "compile_ppo": "off",
            "mingru_train_scan": "off",
        },
        "candidate": {
            "compile_policy": "inductor",
            "compile_ppo": "inductor",
            "mingru_train_scan": "metal",
        },
    }
    assert result["quality_thresholds"] == [2.0, 4.0]
    assert json.loads(capsys.readouterr().out)["valid"] is True


@pytest.mark.parametrize(
    "mode", [holdout.BASELINE_MODE, holdout.CANDIDATE_MODE])
def test_exact_run_identity_accepts_complete_finite_run(mode):
    run = _fake_run(mode)
    report = holdout.validate_run_identity(
        run, _protocol(), expected_mode=mode, expected_seed=233)
    assert report["passed"] is True
    assert report["failed_checks"] == []


@pytest.mark.parametrize(
    ("field", "value", "failed_check"),
    [
        ("effective_ppo_compile", "off", "ppo_compiler"),
        ("effective_mingru_train_scan", "off", "mingru_train_scan"),
        ("post_preflight_dynamo_frames_total", 1,
         "zero_post_preflight_dynamo"),
        ("summary", {"finite": False}, "finite_training_state"),
        ("optimization_startup_seconds", 3.0,
         "optimization_startup_accounting"),
    ],
)
def test_candidate_identity_rejects_optimized_impostors(
        field, value, failed_check):
    run = _fake_run(holdout.CANDIDATE_MODE)
    run[field] = value
    report = holdout.run_identity_report(
        run, _protocol(), expected_mode=holdout.CANDIDATE_MODE,
        expected_seed=233)
    assert report["passed"] is False
    assert failed_check in report["failed_checks"]
    with pytest.raises(RuntimeError, match=failed_check):
        holdout.validate_run_identity(
            run, _protocol(), expected_mode=holdout.CANDIDATE_MODE,
            expected_seed=233)


def test_run_identity_rejects_incomplete_epoch_sequence():
    run = _fake_run(holdout.CANDIDATE_MODE)
    run["records"].pop()
    report = holdout.run_identity_report(run, _protocol())
    assert report["passed"] is False
    assert {"complete_budget", "epoch_sequence"}.issubset(
        report["failed_checks"])


def test_paired_configs_can_differ_only_in_preregistered_optimization_keys():
    runs = [
        _fake_run(mode, seed)
        for seed in holdout.EXPECTED_SEEDS
        for mode in (holdout.BASELINE_MODE, holdout.CANDIDATE_MODE)
    ]
    report = holdout.validate_paired_configs(runs, _protocol())
    assert report["passed"] is True
    assert report["allowed_torch_differences"] == [
        "compile_ppo", "mingru_train_scan"]

    candidate = next(run for run in runs
        if run["mode"] == holdout.CANDIDATE_MODE)
    candidate["effective_config"]["train"]["learning_rate"] = 0.002
    with pytest.raises(RuntimeError, match="not identical"):
        holdout.validate_paired_configs(runs, _protocol())


def test_protocol_acceptance_maps_every_preregistered_gate():
    runs = [
        _fake_run(mode, seed)
        for seed in holdout.EXPECTED_SEEDS
        for mode in (holdout.BASELINE_MODE, holdout.CANDIDATE_MODE)
    ]
    result = holdout.protocol_acceptance(
        _passing_comparison(), runs, _protocol())
    assert result["passed"] is True
    assert all(result["checks"].values())


def test_protocol_acceptance_rejects_bootstrap_or_recompile_failure():
    runs = [
        _fake_run(mode, seed)
        for seed in holdout.EXPECTED_SEEDS
        for mode in (holdout.BASELINE_MODE, holdout.CANDIDATE_MODE)
    ]
    comparison = _passing_comparison()
    comparison["aggregate"]["tail_score_ratio_bootstrap_90pct_lower"] = 0.89
    result = holdout.protocol_acceptance(comparison, runs, _protocol())
    assert result["passed"] is False
    assert result["checks"]["tail_score_bootstrap_90pct_lower"] is False

    comparison = _passing_comparison()
    candidate = next(run for run in runs
        if run["mode"] == holdout.CANDIDATE_MODE)
    candidate["post_preflight_dynamo_unique_graphs"] = 1
    result = holdout.protocol_acceptance(comparison, runs, _protocol())
    assert result["passed"] is False
    assert result["checks"][
        "candidate_zero_post_preflight_dynamo_graphs"] is False


def test_isolated_child_uses_fresh_process_and_unique_empty_cache(
        monkeypatch, tmp_path):
    calls = []

    def fake_subprocess_run(command, *, cwd, env, text, stdout, stderr,
            check):
        assert cwd == holdout.ROOT
        assert text is True
        assert check is False
        cache = Path(env["TORCHINDUCTOR_CACHE_DIR"])
        assert cache.is_dir()
        assert list(cache.iterdir()) == []
        child_output = Path(command[command.index("--child-output") + 1])
        calls.append({
            "command": list(command),
            "cache": str(cache),
            "fallback": env["PYTORCH_ENABLE_MPS_FALLBACK"],
            "layout": env["TORCHINDUCTOR_LAYOUT_OPTIMIZATION"],
        })
        child_output.write_text(json.dumps({
            "protocol_sha256": holdout.EXPECTED_PROTOCOL_SHA256,
            "child_pid": 1000 + len(calls),
            "inductor_cache_dir": str(cache),
        }))
        return SimpleNamespace(
            returncode=0, stdout=f"fresh child {len(calls)}\n")

    monkeypatch.setattr(holdout.subprocess, "run", fake_subprocess_run)
    first = holdout.run_isolated_child(
        holdout.BASELINE_MODE, 233, holdout.EXPECTED_PROTOCOL_SHA256,
        log_dir=tmp_path)
    second = holdout.run_isolated_child(
        holdout.BASELINE_MODE, 233, holdout.EXPECTED_PROTOCOL_SHA256,
        log_dir=tmp_path)

    assert calls[0]["cache"] != calls[1]["cache"]
    assert all(call["fallback"] == "0" for call in calls)
    assert all(call["layout"] == "0" for call in calls)
    assert all("--child" in call["command"] for call in calls)
    forbidden_overrides = {
        "--agents", "--horizon", "--minibatch-size", "--timesteps",
        "--threads", "--thresholds",
    }
    assert all(not forbidden_overrides.intersection(call["command"])
        for call in calls)
    assert first["isolation"]["fresh_process"] is True
    assert second["isolation"]["unique_one_use_cache"] is True
    assert first["isolation"]["log_sha256"] \
        != second["isolation"]["log_sha256"]


def test_completed_holdout_report_is_immutable(tmp_path):
    output = tmp_path / "optimized_holdout.json"
    output.write_text('{"completion_status":"completed_failed"}\n')
    with pytest.raises(RuntimeError, match="immutable"):
        holdout.parent_run(output=output, log_dir=tmp_path / "logs")
    assert json.loads(output.read_text())["completion_status"] \
        == "completed_failed"


def test_immutable_report_publish_never_replaces_existing_result(tmp_path):
    output = tmp_path / "result.json"
    holdout._immutable_json_write(output, {"status": "failed"})
    with pytest.raises(RuntimeError, match="immutable"):
        holdout._immutable_json_write(output, {"status": "passed"})
    assert json.loads(output.read_text()) == {"status": "failed"}
