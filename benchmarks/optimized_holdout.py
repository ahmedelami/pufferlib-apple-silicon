"""Execute the preregistered final M5 combined-optimization holdout.

The immutable operational inputs come from ``optimized_holdout_protocol.json``.
There are no CLI overrides for seeds, shape, budget, modes, or acceptance
criteria. Each mode/seed replicate gets a fresh process and a unique empty
Inductor cache. A completed report is never overwritten, including a failed
quality result.

Run from the repository root only after the Breakout CPU extension is built::

    PYTORCH_ENABLE_MPS_FALLBACK=0 \
      .venv/bin/python benchmarks/optimized_holdout.py

Use ``--validate-only`` for a CPU-only protocol/configuration audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import learning_quality as learning


PROTOCOL_PATH = ROOT / "benchmarks" / "optimized_holdout_protocol.json"
RESULT_MANIFEST_PATH = (
    ROOT / "benchmarks" / "optimized_holdout_result.json")
OUTPUT_PATH = ROOT / "work" / "optimized_holdout.json"
LOG_DIR = ROOT / "work" / "optimized_holdout_logs"

BASELINE_MODE = "mps_compile"
CANDIDATE_MODE = "mps_compile_ppo_mingru"
# These are the established Breakout learning-analyzer checkpoints in force at
# preregistration. Pin them here so later analyzer-default drift cannot alter a
# now-observed holdout.
QUALITY_THRESHOLDS = (2.0, 4.0)

EXPECTED_SCHEMA = "pufferlib-final-m5-optimization-holdout-v1"
EXPECTED_RESULT_SCHEMA = "pufferlib-final-m5-optimization-holdout-result-v1"
EXPECTED_PROTOCOL_SHA256 = (
    "fd7f562c84594a6126501e636d136a0182ce357160ddef1c62ecf88362be25eb")
EXPECTED_SEEDS = (233, 239, 241, 251, 257)
EXPECTED_EXCLUDED_SEEDS = (
    11, 23, 37, 42, 53, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197,
)
EXPECTED_OPERATIONAL_PROTOCOL = {
    "fresh_process_per_mode_and_seed": True,
    "alternating_pair_order": True,
    "unique_empty_inductor_cache_per_replicate": True,
    "agents": 4096,
    "horizon": 64,
    "minibatch_size": 65_536,
    "epochs": 358,
    "timesteps_per_run": 93_847_552,
    "environment_threads": 18,
    "torch_threads": 12,
    "torch_interop_threads": 1,
    "precision": "float32",
    "optimization_startup_included": True,
    "numerical_health_checks_excluded_from_acceptance_clock": True,
    "observed_loop_wall_clock_recorded_separately": True,
    "invalid_run_policy": "restart the complete holdout unchanged",
}
EXPECTED_ACCEPTANCE = {
    "median_tail_score_ratio_min": 0.95,
    "tail_score_bootstrap_90pct_lower_min": 0.90,
    "median_learning_auc_ratio_min": 0.95,
    "learning_auc_bootstrap_90pct_lower_min": 0.90,
    "median_steps_to_score_ratio_max": 1.05,
    "allowed_candidate_reach_count_deficit": 1,
    "median_measured_training_speedup_min": 1.0,
    "all_runs_finite": True,
    "all_candidate_runs_must_verify_exact_compiler_sampler_ppo_scan_identity":
        True,
    "candidate_must_show_zero_post_preflight_dynamo_graphs": True,
}
EXPECTED_BASELINE = {
    "description": "promoted FP32 compiled policy and fused Philox sampler",
    "policy_compile": "inductor",
    "ppo_compile": "off",
    "mingru_train_scan": "portable",
}
EXPECTED_CANDIDATE = {
    "description": (
        "baseline plus FP32 compiled PPO and Metal MinGRU training scan"),
    "policy_compile": "inductor",
    "ppo_compile": "inductor",
    "mingru_train_scan": "metal",
}
EXPECTED_NOTES = [
    ("The 358-epoch budget matches the 93,847,552 complete steps executed by "
     "the shipped 94,000,000-step loop."),
    ("These seeds and thresholds were fixed before any combined "
     "optimized-stack learning run."),
    ("A failed gate cannot be reversed by rerunning these now-observed "
     "seeds."),
]


def _atomic_json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _immutable_json_write(path: Path, value) -> None:
    """Atomically publish a complete report without replacing any result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RuntimeError(
                f"completed holdout output already exists and is immutable: "
                f"{path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def load_protocol(path: Path = PROTOCOL_PATH):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if path.resolve() == PROTOCOL_PATH.resolve() \
            and digest != EXPECTED_PROTOCOL_SHA256:
        raise ValueError(
            "canonical optimized holdout protocol bytes drifted: "
            f"expected {EXPECTED_PROTOCOL_SHA256}, observed {digest}")
    protocol = json.loads(raw)
    validate_protocol(protocol)
    return protocol, digest


def load_published_result(path: Path | None = None):
    """Validate the tracked terminal decision, when one has been published."""
    path = RESULT_MANIFEST_PATH if path is None else Path(path)
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"published optimized holdout result is unreadable: {path}") from exc
    if result.get("schema") != EXPECTED_RESULT_SCHEMA:
        raise RuntimeError("published optimized holdout result schema drifted")
    if result.get("completion_status") not in {
            "completed_failed", "completed_passed"}:
        raise RuntimeError(
            "published optimized holdout result is not terminal")
    if result.get("protocol", {}).get("sha256") \
            != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            "published optimized holdout protocol fingerprint drifted")
    if result["completion_status"] == "completed_failed" \
            and result.get("promotion_eligible") is not False:
        raise RuntimeError(
            "failed published holdout cannot be promotion eligible")
    return result


def refuse_published_rerun() -> None:
    result = load_published_result()
    if result is not None:
        raise RuntimeError(
            "published optimized holdout decision is immutable; refusing to "
            f"rerun {result['completion_status']} observed seeds")


def validate_protocol(protocol) -> None:
    if not isinstance(protocol, dict):
        raise ValueError("optimized holdout protocol must be a JSON object")
    required_top_level = {
        "schema", "preregistered_at", "status", "environment",
        "baseline", "candidate", "paired_seeds",
        "previously_observed_seeds_excluded", "protocol", "acceptance",
        "notes",
    }
    if set(protocol) != required_top_level:
        raise ValueError("optimized holdout protocol top-level schema drifted")
    if protocol["schema"] != EXPECTED_SCHEMA:
        raise ValueError("optimized holdout protocol schema is not v1")
    if protocol["preregistered_at"] != "2026-08-23":
        raise ValueError("optimized holdout preregistration date drifted")
    if protocol["status"] != "preregistered_not_run":
        raise ValueError("optimized holdout protocol is no longer unrun")
    if protocol["environment"] != "breakout":
        raise ValueError("optimized holdout environment must remain breakout")
    if protocol["baseline"] != EXPECTED_BASELINE:
        raise ValueError("optimized holdout baseline identity drifted")
    if protocol["candidate"] != EXPECTED_CANDIDATE:
        raise ValueError("optimized holdout candidate identity drifted")
    if tuple(protocol["paired_seeds"]) != EXPECTED_SEEDS:
        raise ValueError("optimized holdout paired seeds drifted")
    excluded = tuple(int(seed)
        for seed in protocol["previously_observed_seeds_excluded"])
    if excluded != EXPECTED_EXCLUDED_SEEDS:
        raise ValueError("previously observed seed exclusion list drifted")
    if set(EXPECTED_SEEDS) & set(excluded):
        raise ValueError("holdout seeds overlap previously observed seeds")
    if protocol["protocol"] != EXPECTED_OPERATIONAL_PROTOCOL:
        raise ValueError("optimized holdout operational protocol drifted")
    if protocol["acceptance"] != EXPECTED_ACCEPTANCE:
        raise ValueError("optimized holdout acceptance criteria drifted")
    if protocol["notes"] != EXPECTED_NOTES:
        raise ValueError("optimized holdout preregistration notes drifted")
    if tuple(float(value) for value in learning.DEFAULT_THRESHOLDS) \
            != QUALITY_THRESHOLDS:
        raise ValueError("learning analyzer score checkpoints drifted")

    shape = protocol["protocol"]
    batch_size = int(shape["agents"]) * int(shape["horizon"])
    if int(shape["minibatch_size"]) % int(shape["horizon"]):
        raise ValueError("minibatch size must be a multiple of horizon")
    if int(shape["epochs"]) * batch_size != int(shape["timesteps_per_run"]):
        raise ValueError("holdout epoch and timestep budgets disagree")


def mode_order(seed_index: int):
    if int(seed_index) % 2 == 0:
        return (BASELINE_MODE, CANDIDATE_MODE)
    return (CANDIDATE_MODE, BASELINE_MODE)


def expected_run_order(protocol):
    return [
        {"seed": int(seed), "mode": mode}
        for index, seed in enumerate(protocol["paired_seeds"])
        for mode in mode_order(index)
    ]


def _mode_role(mode: str):
    try:
        return {
            BASELINE_MODE: "baseline",
            CANDIDATE_MODE: "candidate",
        }[mode]
    except KeyError as exc:
        raise ValueError(f"mode {mode!r} is outside the final holdout") from exc


def _finite_losses(record):
    losses = record.get("losses")
    return isinstance(losses, dict) and bool(losses) and all(
        math.isfinite(float(value)) for value in losses.values())


def run_identity_report(run, protocol, *, expected_mode=None,
        expected_seed=None):
    """Return exact execution-identity and numerical-health checks."""
    mode = str(run.get("mode"))
    role = _mode_role(mode)
    shape = protocol["protocol"]
    definition = protocol[role]
    expected_ppo = definition["ppo_compile"]
    expected_scan = (
        "off" if definition["mingru_train_scan"] == "portable"
        else definition["mingru_train_scan"])
    expected_steps = int(shape["timesteps_per_run"])
    expected_epochs = int(shape["epochs"])
    batch_size = int(shape["agents"]) * int(shape["horizon"])
    records = run.get("records") or []
    summary = run.get("summary") or {}
    config = run.get("effective_config") or {}
    torch_config = config.get("torch") or {}
    train_config = config.get("train") or {}
    vec_config = config.get("vec") or {}

    checks = {
        "mode": expected_mode is None or mode == expected_mode,
        "seed": expected_seed is None or int(run.get("seed", -1))
            == int(expected_seed),
        "environment": run.get("environment") == protocol["environment"],
        "direct_mps": (
            run.get("training_device") == "mps"
            and run.get("rollout_device") == "mps"
            and run.get("mps_host_alias_io") is True),
        "float32_precision": (
            run.get("requested_amp_dtype") == shape["precision"]
            and run.get("effective_amp_dtype") == shape["precision"]),
        "policy_compiler": (
            run.get("requested_policy_compile")
                == definition["policy_compile"]
            and run.get("effective_policy_compile")
                == definition["policy_compile"]
            and run.get("policy_compile_preflight") is True
            and run.get("policy_compile_wrapper_verified") is True
            and float(run.get("policy_compile_startup_seconds", 0.0)) > 0.0),
        "fused_sampler": (
            run.get("requested_rollout_sampler") == "fused_mps_philox"
            and run.get("effective_rollout_sampler") == "fused_mps_philox"
            and float(run.get("rollout_sampler_startup_seconds", 0.0)) > 0.0),
        "ppo_compiler": (
            run.get("requested_ppo_compile") == expected_ppo
            and run.get("effective_ppo_compile") == expected_ppo
            and run.get("ppo_compile_preflight")
                is (expected_ppo == "inductor")
            and run.get("ppo_compile_wrapper_verified")
                is (expected_ppo == "inductor")
            and (
                float(run.get("ppo_compile_startup_seconds", -1.0)) > 0.0
                if expected_ppo == "inductor"
                else float(run.get("ppo_compile_startup_seconds", -1.0))
                    == 0.0)),
        "mingru_train_scan": (
            run.get("requested_mingru_train_scan") == expected_scan
            and run.get("effective_mingru_train_scan") == expected_scan
            and run.get("mingru_train_scan_preflight")
                is (expected_scan == "metal")
            and (
                float(run.get("mingru_train_scan_startup_seconds", -1.0)) > 0.0
                if expected_scan == "metal"
                else float(run.get(
                    "mingru_train_scan_startup_seconds", -1.0)) == 0.0)),
        "zero_post_preflight_dynamo": (
            int(run.get("post_preflight_dynamo_frames_total", -1)) == 0
            and int(run.get(
                "post_preflight_dynamo_unique_graphs", -1)) == 0),
        "effective_torch_config": (
            torch_config.get("device") == "mps"
            and torch_config.get("rollout_device") == "mps"
            and torch_config.get("amp_dtype") == shape["precision"]
            and torch_config.get("compile_policy")
                == definition["policy_compile"]
            and torch_config.get("compile_ppo") == expected_ppo
            and torch_config.get("mingru_train_scan") == expected_scan),
        "effective_training_shape": (
            int(train_config.get("horizon", -1)) == int(shape["horizon"])
            and int(train_config.get("minibatch_size", -1))
                == int(shape["minibatch_size"])
            and int(train_config.get("total_timesteps", -1))
                == expected_steps
            and int(vec_config.get("total_agents", -1))
                == int(shape["agents"])
            and int(vec_config.get("num_threads", -1))
                == int(shape["environment_threads"])),
        "complete_budget": (
            len(records) == expected_epochs
            and int(records[-1].get("epoch", -1)) == expected_epochs
            and int(records[-1].get("agent_steps", -1)) == expected_steps
            if records else False),
        "epoch_sequence": (
            len(records) == expected_epochs
            and all(
                int(record.get("epoch", -1)) == index
                and int(record.get("agent_steps", -1)) == index * batch_size
                for index, record in enumerate(records, start=1))),
        "finite_training_state": (
            summary.get("finite") is True
            and all(_finite_losses(record)
                and record.get("training_state", {}).get("passed") is True
                for record in records)),
        "timing_contract": (
            math.isfinite(float(run.get("total_seconds", float("nan"))))
            and float(run.get("total_seconds", 0.0))
                >= float(run.get("optimization_startup_seconds", 0.0)) > 0.0
            and math.isfinite(float(run.get(
                "measured_loop_wall_seconds", float("nan"))))
            and float(run.get("measured_loop_wall_seconds", 0.0))
                >= float(run.get("total_seconds", 0.0))
            and float(run.get("validation_seconds_total", -1.0)) >= 0.0),
        "optimization_startup_accounting": math.isclose(
            float(run.get("optimization_startup_seconds", float("nan"))),
            sum(float(run.get(field, float("nan"))) for field in (
                "policy_compile_startup_seconds",
                "rollout_sampler_startup_seconds",
                "ppo_compile_startup_seconds",
                "mingru_train_scan_startup_seconds",
            )),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
    }
    return {
        "role": role,
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items()
            if not passed],
    }


def validate_run_identity(run, protocol, *, expected_mode=None,
        expected_seed=None):
    report = run_identity_report(
        run, protocol,
        expected_mode=expected_mode,
        expected_seed=expected_seed)
    if not report["passed"]:
        raise RuntimeError(
            f"{report['role']} run identity failed: "
            + ", ".join(report["failed_checks"]))
    return report


def validate_paired_configs(runs, protocol):
    """Prove each seed changed only the two preregistered optimization keys."""
    by_seed = {}
    for run in runs:
        by_seed.setdefault(int(run["seed"]), {})[run["mode"]] = run
    allowed_torch_differences = {"compile_ppo", "mingru_train_scan"}
    checks = {}
    for seed in protocol["paired_seeds"]:
        pair = by_seed.get(int(seed), {})
        if set(pair) != {BASELINE_MODE, CANDIDATE_MODE}:
            checks[str(seed)] = False
            continue
        baseline = pair[BASELINE_MODE]["effective_config"]
        candidate = pair[CANDIDATE_MODE]["effective_config"]
        baseline_torch = dict(baseline["torch"])
        candidate_torch = dict(candidate["torch"])
        for key in allowed_torch_differences:
            baseline_torch.pop(key, None)
            candidate_torch.pop(key, None)
        checks[str(seed)] = (
            baseline_torch == candidate_torch
            and all(baseline.get(section) == candidate.get(section)
                for section in ("train", "vec", "env", "policy")))
    report = {
        "passed": all(checks.values()),
        "per_seed": checks,
        "allowed_torch_differences": sorted(allowed_torch_differences),
    }
    if not report["passed"]:
        raise RuntimeError("paired holdout configurations are not identical")
    return report


def _validate_child_provenance(
        run, expected_system, *, expected_environment="breakout"):
    system = run.get("system")
    if not isinstance(system, dict):
        raise RuntimeError("holdout child did not record system provenance")
    exact_fields = (
        "git_revision", "working_tree_patch", "torch",
        "torch_git_revision", "hardware_model", "chip", "gpu_model",
        "gpu_cores", "memory_bytes", "macos_build", "compiled_environment",
        "extension_gpu", "extension_precision_bytes",
    )
    mismatches = [field for field in exact_fields
        if system.get(field) != expected_system.get(field)]
    environment = system.get("environment_variables", {})
    if system.get("compiled_environment") != expected_environment:
        mismatches.append("compiled_environment_expected_environment")
    if environment.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        mismatches.append("PYTORCH_ENABLE_MPS_FALLBACK")
    if environment.get("TORCHINDUCTOR_LAYOUT_OPTIMIZATION") != "0":
        mismatches.append("TORCHINDUCTOR_LAYOUT_OPTIMIZATION")
    if int(system.get("torch_threads", -1)) != int(
            EXPECTED_OPERATIONAL_PROTOCOL["torch_threads"]):
        mismatches.append("torch_threads")
    if int(system.get("torch_interop_threads", -1)) != int(
            EXPECTED_OPERATIONAL_PROTOCOL["torch_interop_threads"]):
        mismatches.append("torch_interop_threads")
    if mismatches:
        raise RuntimeError(
            "holdout child provenance differs: " + ", ".join(mismatches))


def child_run(mode: str, seed: int, output: Path):
    refuse_published_rerun()
    protocol, protocol_digest = load_protocol()
    _mode_role(mode)
    if int(seed) not in protocol["paired_seeds"]:
        raise ValueError("child seed is outside the preregistered holdout")
    compiled_environment = getattr(learning._C, "env_name", None)
    if compiled_environment != protocol["environment"]:
        raise RuntimeError(
            "native extension environment differs from holdout: "
            f"expected {protocol['environment']!r}, "
            f"observed {compiled_environment!r}")
    shape = protocol["protocol"]
    learning.torch.set_num_threads(int(shape["torch_threads"]))
    learning.torch.set_num_interop_threads(int(shape["torch_interop_threads"]))
    run = learning.run_one(
        protocol["environment"],
        mode,
        int(seed),
        agents=int(shape["agents"]),
        horizon=int(shape["horizon"]),
        minibatch_size=int(shape["minibatch_size"]),
        timesteps=int(shape["timesteps_per_run"]),
        threads=int(shape["environment_threads"]),
        mps_host_alias="auto",
        strict_determinism=False,
    )
    run["summary"] = learning.summarize_run(
        run,
        thresholds=QUALITY_THRESHOLDS,
        window_epochs=4,
        min_window_episodes=2000,
        sustain_epochs=2,
        tail_fraction=0.25,
    )
    run["protocol_sha256"] = protocol_digest
    run["child_pid"] = os.getpid()
    run["inductor_cache_dir"] = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    run["identity"] = validate_run_identity(
        run, protocol, expected_mode=mode, expected_seed=seed)
    run["system"] = learning._system_metadata()
    _atomic_json_write(output, run)
    print(json.dumps({
        "event": "optimized_holdout_child_complete",
        "mode": mode,
        "seed": int(seed),
        "summary": run["summary"],
    }, sort_keys=True), flush=True)


def run_isolated_child(mode: str, seed: int, protocol_digest: str,
        *, log_dir: Path = LOG_DIR):
    """Run one replicate in a fresh process and unique empty cache."""
    with tempfile.TemporaryDirectory(
            prefix=f"pufferlib-final-cache-{mode}-{seed}-") as cache_dir, \
            tempfile.TemporaryDirectory(
                prefix=f"pufferlib-final-output-{mode}-{seed}-") as output_dir:
        cache_path = Path(cache_dir)
        child_output = Path(output_dir) / "result.json"
        if any(cache_path.iterdir()):
            raise RuntimeError("new Inductor cache was not empty")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT)
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        env["TORCHINDUCTOR_LAYOUT_OPTIMIZATION"] = "0"
        env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_path)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--mode", mode,
            "--seed", str(int(seed)),
            "--child-output", str(child_output),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"seed_{seed}_{mode}.log"
        log_path.write_text(completed.stdout)
        print(completed.stdout, end="", flush=True)
        if completed.returncode:
            raise RuntimeError(
                f"holdout child failed: mode={mode} seed={seed} "
                f"exit={completed.returncode}; see {log_path}")
        if not child_output.is_file():
            raise RuntimeError(
                f"holdout child produced no result: mode={mode} seed={seed}")
        run = json.loads(child_output.read_text())
        if run.get("protocol_sha256") != protocol_digest:
            raise RuntimeError("child used a different preregistered protocol")
        if run.get("inductor_cache_dir") != str(cache_path):
            raise RuntimeError("child did not use its assigned Inductor cache")
        if int(run.get("child_pid", -1)) == os.getpid():
            raise RuntimeError("holdout replicate did not use a fresh process")
        try:
            recorded_log_path = str(log_path.relative_to(ROOT))
        except ValueError:
            recorded_log_path = str(log_path)
        run["isolation"] = {
            "fresh_process": True,
            "unique_one_use_cache": True,
            "cache_initial_entries": 0,
            "cache_token": hashlib.sha256(
                str(cache_path).encode()).hexdigest(),
            "log": recorded_log_path,
            "log_sha256": hashlib.sha256(
                completed.stdout.encode()).hexdigest(),
        }
        return run


def protocol_acceptance(comparison, runs, protocol):
    criteria = protocol["acceptance"]
    aggregate = comparison["aggregate"]
    identity = [run["identity"] for run in runs]
    candidate_runs = [run for run in runs
        if run["mode"] == CANDIDATE_MODE]

    threshold_coverage = []
    threshold_steps = []
    for result in comparison["thresholds"].values():
        if not result["gated"]:
            continue
        baseline_count = len(result["baseline_reached_seeds"])
        candidate_count = len(result["candidate_reached_seeds"])
        paired_count = len(
            set(result["baseline_reached_seeds"])
            & set(result["candidate_reached_seeds"]))
        allowed_deficit = int(
            criteria["allowed_candidate_reach_count_deficit"])
        threshold_coverage.append(
            candidate_count >= baseline_count
                - allowed_deficit)
        ratio = result["median_candidate_to_baseline_steps"]
        threshold_steps.append(
            paired_count >= baseline_count - allowed_deficit
            and ratio is not None
            and float(ratio)
                <= float(criteria["median_steps_to_score_ratio_max"]))

    checks = {
        "established_analyzer_acceptance": (
            comparison.get("acceptance", {}).get("passed") is True),
        "median_tail_score_ratio": (
            aggregate["median_tail_score_ratio"] is not None
            and aggregate["median_tail_score_ratio"]
                >= criteria["median_tail_score_ratio_min"]),
        "tail_score_bootstrap_90pct_lower": (
            aggregate["tail_score_ratio_bootstrap_90pct_lower"] is not None
            and aggregate["tail_score_ratio_bootstrap_90pct_lower"]
                >= criteria["tail_score_bootstrap_90pct_lower_min"]),
        "median_learning_auc_ratio": (
            aggregate["median_mean_score_auc_ratio"] is not None
            and aggregate["median_mean_score_auc_ratio"]
                >= criteria["median_learning_auc_ratio_min"]),
        "learning_auc_bootstrap_90pct_lower": (
            aggregate["mean_score_auc_ratio_bootstrap_90pct_lower"] is not None
            and aggregate["mean_score_auc_ratio_bootstrap_90pct_lower"]
                >= criteria["learning_auc_bootstrap_90pct_lower_min"]),
        "threshold_reach_deficit": (
            bool(threshold_coverage) and all(threshold_coverage)),
        "median_steps_to_score_ratio": (
            bool(threshold_steps) and all(threshold_steps)),
        "median_measured_training_speedup": (
            aggregate["median_measured_training_speedup"] is not None
            and aggregate["median_measured_training_speedup"]
                >= criteria["median_measured_training_speedup_min"]),
        "all_runs_finite": all(
            run["summary"].get("finite") is True for run in runs),
        "all_baseline_and_candidate_identities": all(
            report.get("passed") is True for report in identity),
        "all_candidate_exact_optimized_identity": all(
            run["identity"].get("passed") is True
            and run["identity"].get("role") == "candidate"
            for run in candidate_runs),
        "candidate_zero_post_preflight_dynamo_graphs": all(
            int(run.get("post_preflight_dynamo_frames_total", -1)) == 0
            and int(run.get("post_preflight_dynamo_unique_graphs", -1)) == 0
            for run in candidate_runs),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "criteria": criteria,
        "quality_thresholds": list(QUALITY_THRESHOLDS),
    }


def parent_run(*, output: Path = OUTPUT_PATH, log_dir: Path = LOG_DIR,
        child_runner=run_isolated_child):
    refuse_published_rerun()
    protocol, protocol_digest = load_protocol()
    if output.exists():
        raise RuntimeError(
            f"completed holdout output already exists and is immutable: {output}")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must be 0")

    shape = protocol["protocol"]
    learning.torch.set_num_threads(int(shape["torch_threads"]))
    learning.torch.set_num_interop_threads(int(shape["torch_interop_threads"]))
    parent_system = learning._system_metadata()
    runs = []
    order = expected_run_order(protocol)
    cache_tokens = set()
    for item in order:
        run = child_runner(
            item["mode"], item["seed"], protocol_digest, log_dir=log_dir)
        validate_run_identity(
            run, protocol,
            expected_mode=item["mode"], expected_seed=item["seed"])
        _validate_child_provenance(
            run, parent_system,
            expected_environment=protocol["environment"])
        isolation = run.get("isolation") or {}
        if not (
                isolation.get("fresh_process") is True
                and isolation.get("unique_one_use_cache") is True
                and isolation.get("cache_initial_entries") == 0):
            raise RuntimeError("child did not prove fresh-process/cache isolation")
        token = isolation.get("cache_token")
        if not token:
            raise RuntimeError("child did not record its one-use cache token")
        if token in cache_tokens:
            raise RuntimeError("child isolation evidence was reused")
        cache_tokens.add(token)
        runs.append(run)

    paired_config = validate_paired_configs(runs, protocol)
    summaries = [run["summary"] for run in runs]
    comparison = learning.compare_modes(
        summaries,
        baseline_mode=BASELINE_MODE,
        candidate_mode=CANDIDATE_MODE,
        thresholds=QUALITY_THRESHOLDS,
        min_ratio=float(protocol["acceptance"][
            "median_tail_score_ratio_min"]),
        bootstrap_guard=float(protocol["acceptance"][
            "tail_score_bootstrap_90pct_lower_min"]),
        max_step_ratio=float(protocol["acceptance"][
            "median_steps_to_score_ratio_max"]),
        require_candidate_host_alias=True,
    )
    acceptance = protocol_acceptance(comparison, runs, protocol)
    report = {
        "schema": protocol["schema"],
        "completion_status": (
            "completed_passed" if acceptance["passed"]
            else "completed_failed"),
        "protocol_source": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": protocol_digest,
        "preregistered_protocol": protocol,
        "modes": {
            "baseline": BASELINE_MODE,
            "candidate": CANDIDATE_MODE,
        },
        "run_order": order,
        "system": parent_system,
        "paired_configuration_identity": paired_config,
        "runs": runs,
        "comparison": comparison,
        "acceptance": acceptance,
        "promotion_eligible": acceptance["passed"],
    }
    _immutable_json_write(output, report)
    print(json.dumps({
        "completion_status": report["completion_status"],
        "acceptance": acceptance,
        "output": str(output),
    }, indent=2, sort_keys=True), flush=True)
    return 0 if acceptance["passed"] else 1


def validate_only():
    protocol, protocol_digest = load_protocol()
    published = load_published_result()
    shape = protocol["protocol"]
    common = {
        "agents": shape["agents"],
        "horizon": shape["horizon"],
        "minibatch_size": shape["minibatch_size"],
        "timesteps": shape["timesteps_per_run"],
        "threads": shape["environment_threads"],
    }
    baseline = learning.make_args(
        protocol["environment"], BASELINE_MODE, **common)
    candidate = learning.make_args(
        protocol["environment"], CANDIDATE_MODE, **common)
    observed = {
        "baseline": {
            "compile_policy": baseline["torch"]["compile_policy"],
            "compile_ppo": baseline["torch"]["compile_ppo"],
            "mingru_train_scan": baseline["torch"]["mingru_train_scan"],
        },
        "candidate": {
            "compile_policy": candidate["torch"]["compile_policy"],
            "compile_ppo": candidate["torch"]["compile_ppo"],
            "mingru_train_scan": candidate["torch"]["mingru_train_scan"],
        },
    }
    expected = {
        "baseline": {
            "compile_policy": protocol["baseline"]["policy_compile"],
            "compile_ppo": protocol["baseline"]["ppo_compile"],
            "mingru_train_scan": "off",
        },
        "candidate": {
            "compile_policy": protocol["candidate"]["policy_compile"],
            "compile_ppo": protocol["candidate"]["ppo_compile"],
            "mingru_train_scan": protocol["candidate"]["mingru_train_scan"],
        },
    }
    if observed != expected:
        raise RuntimeError(
            f"learning mode mapping differs from protocol: {observed}")
    result = {
        "valid": True,
        "protocol_sha256": protocol_digest,
        "run_order": expected_run_order(protocol),
        "mode_config": observed,
        "quality_thresholds": list(QUALITY_THRESHOLDS),
        "published_result": (
            None if published is None else {
                "completion_status": published["completion_status"],
                "promotion_eligible": published["promotion_eligible"],
                "source_report_sha256": published[
                    "source_report"]["sha256"],
            }),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--mode", choices=(BASELINE_MODE, CANDIDATE_MODE),
        help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--child-output", type=Path, help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.child:
        if options.validate_only:
            parser.error("--child and --validate-only are mutually exclusive")
        if options.mode is None or options.seed is None \
                or options.child_output is None:
            parser.error("child mode requires --mode, --seed, and --child-output")
        child_run(options.mode, options.seed, options.child_output)
        return 0
    if any(value is not None
            for value in (options.mode, options.seed, options.child_output)):
        parser.error("child-only arguments require --child")
    if options.validate_only:
        validate_only()
        return 0
    return parent_run()


if __name__ == "__main__":
    raise SystemExit(main())
