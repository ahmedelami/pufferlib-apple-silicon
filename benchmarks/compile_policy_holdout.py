"""Isolated MPS compiler and BF16 learning-quality holdouts.

Every replicate runs in a fresh process, paired order alternates, and every
replicate gets a unique empty Inductor cache. The discovery seeds are never
pooled into the holdout result. Build Breakout first, then run the original
eager-vs-compiled FP32 holdout::

    PYTORCH_ENABLE_MPS_FALLBACK=0 python \
      benchmarks/compile_policy_holdout.py

Run the compiled-FP32-vs-compiled-BF16 validation separately with::

    PYTORCH_ENABLE_MPS_FALLBACK=0 python \
      benchmarks/compile_policy_holdout.py --comparison bf16

Run the production-length compiled-policy-only versus compiled-policy-plus-PPO
promotion gate with::

    PYTORCH_ENABLE_MPS_FALLBACK=0 python \
      benchmarks/compile_policy_holdout.py --comparison ppo

The BF16 profile is retained for reproducibility as an experimental path. Its
completed preregistered promotion run failed, so reruns cannot turn it into
promotion evidence and no production-length profile is enabled.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "work" / "compile_policy_holdout.json"
BF16_OUTPUT = ROOT / "work" / "compile_policy_bf16_holdout.json"
PPO_OUTPUT = ROOT / "work" / "compile_ppo_holdout.json"
DISCOVERY_SEEDS = (11, 23, 37, 53, 71)
SEEDS = (73, 79, 83, 89, 97, 101, 103, 107, 109, 113)
OBSERVED_SEEDS = (
    11, 23, 37, 42, 53, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113)
BF16_SEEDS = (127, 131, 137, 139, 149, 151, 157, 163, 167, 173)
PPO_SEEDS = (179, 181, 191, 193, 197)
AGENTS = 4096
HORIZON = 64
MINIBATCH_SIZE = 65_536
TIMESTEPS = AGENTS * HORIZON * 32
PPO_TIMESTEPS = AGENTS * HORIZON * 358
THREADS = 18
TORCH_THREADS = 12
TORCH_INTEROP_THREADS = 1
COMPILED_MODES = (
    "mps_compile", "mps_compile_bf16", "mps_compile_ppo")
HOLDOUTS = {
    "compile": {
        "promotion_status": "eligible",
        "seeds": SEEDS,
        "excluded_observed_seeds": DISCOVERY_SEEDS,
        "timesteps": TIMESTEPS,
        "modes": ("mps", "mps_compile"),
        "baseline_mode": "mps",
        "candidate_mode": "mps_compile",
        "schema": "pufferlib-compile-policy-holdout-v2",
        "output": OUTPUT,
        "order": "alternating eager-compiled / compiled-eager",
        "precision_by_mode": {
            "mps": "float32",
            "mps_compile": "float32",
        },
        "min_quality_ratio": 0.90,
        "bootstrap_guard": 0.75,
        "max_step_ratio": 1.10,
    },
    "bf16": {
        # This exact preregistered seed/profile set completed on 2026-08-23.
        # It failed the tail bootstrap and score-4 step-equivalence criteria.
        # Keep it reproducible, but never allow a nondeterministic rerun of the
        # now-observed seeds to overwrite that promotion decision.
        "promotion_status": "completed_failed",
        "completed_result": {
            "tail_bootstrap_lower": 0.8992735631317081,
            "required_tail_bootstrap_lower": 0.90,
            "score4_step_ratio": 1.0555555555555556,
            "maximum_score4_step_ratio": 1.05,
            "median_measured_training_speedup": 1.101012715780954,
        },
        "seeds": BF16_SEEDS,
        "excluded_observed_seeds": OBSERVED_SEEDS,
        "timesteps": TIMESTEPS,
        "modes": ("mps_compile", "mps_compile_bf16"),
        "baseline_mode": "mps_compile",
        "candidate_mode": "mps_compile_bf16",
        "schema": "pufferlib-compile-policy-bf16-holdout-v2",
        "output": BF16_OUTPUT,
        "order": "alternating compiled-FP32-compiled-BF16 / "
                 "compiled-BF16-compiled-FP32",
        "precision_by_mode": {
            "mps_compile": "float32",
            "mps_compile_bf16": "bfloat16 autocast",
        },
        # Reduced precision changes MinGRU recurrent numerics, so promotion is
        # held to a tighter preregistered quality/steps contract than the
        # original compiler-lowering comparison.
        "min_quality_ratio": 0.95,
        "bootstrap_guard": 0.90,
        "max_step_ratio": 1.05,
    },
    "ppo": {
        # Full default-run length is intentional. The fused PPO graph adds a
        # cold compiler preflight and breaks even only after about 55M steps;
        # a 32-epoch holdout would test startup rather than the shipped 94M
        # training workload. These seeds were fixed before this run.
        "promotion_status": "superseded_unrun",
        "seeds": PPO_SEEDS,
        "excluded_observed_seeds": tuple(sorted(set(
            (*OBSERVED_SEEDS, *BF16_SEEDS)))),
        "timesteps": PPO_TIMESTEPS,
        "modes": ("mps_compile", "mps_compile_ppo"),
        "baseline_mode": "mps_compile",
        "candidate_mode": "mps_compile_ppo",
        "schema": "pufferlib-compile-ppo-holdout-v1",
        "output": PPO_OUTPUT,
        "order": (
            "alternating compiled-policy-only / compiled-policy-plus-PPO "
            "and reverse"),
        "precision_by_mode": {
            "mps_compile": "float32",
            "mps_compile_ppo": "float32",
        },
        "min_quality_ratio": 0.95,
        "bootstrap_guard": 0.90,
        "max_step_ratio": 1.05,
    },
}


def _validated_timesteps(value):
    value = int(value)
    batch_size = AGENTS * HORIZON
    if value <= 0 or value % batch_size:
        raise ValueError(
            f"timesteps must be a positive multiple of {batch_size}")
    return value


def _preregistration(holdout, seeds, actual_timesteps=None):
    declared_timesteps = int(holdout["timesteps"])
    actual_timesteps = int(
        declared_timesteps if actual_timesteps is None else actual_timesteps)
    batch_size = AGENTS * HORIZON
    return {
        # Retain the original field for consumers of the FP32 report while
        # explicitly recording the broader exclusion set for new profiles.
        "discovery_seeds_excluded": list(DISCOVERY_SEEDS),
        "previously_observed_seeds_excluded": list(
            holdout["excluded_observed_seeds"]),
        "holdout_seeds": list(seeds),
        "run_isolation": "one fresh process per mode and seed",
        "order": holdout["order"],
        "compiler_cache": "unique empty cache per replicate",
        "compile_startup_included": True,
        "acceptance_clock": (
            "optimization startup plus synchronized rollout/train intervals; "
            "numerical validation and bookkeeping excluded"),
        "measured_loop_wall_clock_recorded_separately": True,
        "declared_run_budget": {
            "timesteps_per_run": declared_timesteps,
            "epochs": declared_timesteps // batch_size,
        },
        "actual_run_budget": {
            "timesteps_per_run": actual_timesteps,
            "epochs": actual_timesteps // batch_size,
        },
        "run_budget_preregistered": actual_timesteps == declared_timesteps,
        "budget_status": (
            "preregistered"
            if actual_timesteps == declared_timesteps
            else "exploratory CLI override"
        ),
        "promotion_status": holdout["promotion_status"],
        "acceptance_criteria": {
            "median_quality_ratio_min": holdout["min_quality_ratio"],
            "bootstrap_90pct_lower_guard_min": holdout["bootstrap_guard"],
            "median_steps_to_score_ratio_max": holdout["max_step_ratio"],
            "median_measured_training_speedup_min": 1.0,
        },
        "invalid_run_policy": "restart the complete holdout unchanged",
    }


def _promotion_eligible(holdout, actual_timesteps, comparison):
    """Only an accepted, exactly preregistered budget is promotion evidence."""
    return (
        holdout.get("promotion_status") == "eligible"
        and
        int(actual_timesteps) == int(holdout["timesteps"])
        and comparison.get("acceptance", {}).get("passed") is True
    )


def _mode_order(index, modes=None):
    modes = tuple(HOLDOUTS["compile"]["modes"] if modes is None else modes)
    return modes if int(index) % 2 == 0 else tuple(reversed(modes))


def _validate_child_run(mode, run):
    compiled = mode in COMPILED_MODES
    expected_compile = "inductor" if compiled else "off"
    expected_amp = "bfloat16" if mode == "mps_compile_bf16" else "float32"
    if run.get("requested_policy_compile") != expected_compile:
        raise RuntimeError(
            f"{mode} requested compiler was "
            f"{run.get('requested_policy_compile')!r}, "
            f"expected {expected_compile!r}")
    if run["effective_policy_compile"] != expected_compile:
        raise RuntimeError(
            f"{mode} effective compiler was "
            f"{run['effective_policy_compile']!r}, expected {expected_compile!r}")
    if not (
            run.get("requested_amp_dtype") == expected_amp
            and run.get("effective_amp_dtype") == expected_amp):
        raise RuntimeError(
            f"{mode} did not activate the requested {expected_amp} precision")
    if compiled and not (
            run["policy_compile_preflight"]
            and run["policy_compile_wrapper_verified"]):
        raise RuntimeError(
            f"{mode} did not verify wrappers and graph preflight")
    if compiled and not (
            run["requested_rollout_sampler"] == "fused_mps_philox"
            and run["effective_rollout_sampler"] == "fused_mps_philox"):
        raise RuntimeError(
            f"{mode} did not activate the fused MPS sampler")
    if compiled:
        compile_startup = float(run.get("policy_compile_startup_seconds", 0.0))
        optimization_startup = float(
            run.get("optimization_startup_seconds", 0.0))
        total_seconds = float(run.get("total_seconds", 0.0))
        if not (
                compile_startup > 0.0
                and optimization_startup >= compile_startup
                and total_seconds >= optimization_startup):
            raise RuntimeError(
                f"{mode} did not include compiler/sampler startup in total time")
    if not run["mps_host_alias_io"]:
        raise RuntimeError(f"{mode} did not activate MPS host aliasing")
    expected_ppo = "inductor" if mode == "mps_compile_ppo" else "off"
    if not (
            run.get("requested_ppo_compile", "off") == expected_ppo
            and run.get("effective_ppo_compile", "off") == expected_ppo):
        raise RuntimeError(
            f"{mode} did not activate the requested {expected_ppo!r} "
            "PPO compiler state")
    if expected_ppo == "inductor" and not (
            run.get("ppo_compile_preflight") is True
            and run.get("ppo_compile_wrapper_verified") is True
            and float(run.get("ppo_compile_startup_seconds", 0.0)) > 0.0):
        raise RuntimeError(
            f"{mode} did not verify its PPO wrapper and graph preflight")


def _validate_child_provenance(run, expected_system):
    system = run.get("system")
    if not isinstance(system, dict):
        raise RuntimeError("holdout child did not record system provenance")
    exact_fields = (
        "git_revision", "working_tree_patch", "torch",
        "torch_git_revision", "hardware_model", "chip", "gpu_model",
        "gpu_cores", "memory_bytes", "macos_build",
        "compiled_environment", "extension_gpu", "extension_precision_bytes",
    )
    mismatches = [
        field for field in exact_fields
        if system.get(field) != expected_system.get(field)
    ]
    environment = system.get("environment_variables", {})
    if environment.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        mismatches.append("PYTORCH_ENABLE_MPS_FALLBACK")
    if environment.get("TORCHINDUCTOR_LAYOUT_OPTIMIZATION") != "0":
        mismatches.append("TORCHINDUCTOR_LAYOUT_OPTIMIZATION")
    if mismatches:
        raise RuntimeError(
            "holdout child provenance differs from the parent protocol: "
            + ", ".join(mismatches))


def _child(mode, seed, output, timesteps=TIMESTEPS):
    from benchmarks import learning_quality as learning

    timesteps = _validated_timesteps(timesteps)
    torch = learning.torch
    torch.set_num_threads(TORCH_THREADS)
    torch.set_num_interop_threads(TORCH_INTEROP_THREADS)
    run = learning.run_one(
        "breakout",
        mode,
        seed,
        agents=AGENTS,
        horizon=HORIZON,
        minibatch_size=MINIBATCH_SIZE,
        timesteps=timesteps,
        threads=THREADS,
        mps_host_alias="auto",
    )
    run["summary"] = learning.summarize_run(
        run,
        thresholds=learning.DEFAULT_THRESHOLDS,
        window_epochs=4,
        min_window_episodes=2000,
        sustain_epochs=2,
        tail_fraction=0.25,
    )
    run["system"] = learning._system_metadata()
    _validate_child_run(mode, run)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({
        "event": "holdout_child_complete",
        "mode": mode,
        "seed": seed,
        "summary": run["summary"],
    }, sort_keys=True), flush=True)


def _run_isolated_child(mode, seed, timesteps=TIMESTEPS):
    """Run one replicate with a new process and an empty one-use cache."""
    timesteps = _validated_timesteps(timesteps)
    with tempfile.TemporaryDirectory(
            prefix=f"pufferlib-inductor-{mode}-{seed}-") as cache_dir:
        fd, path = tempfile.mkstemp(
            prefix=f"pufferlib-holdout-{mode}-{seed}-", suffix=".json")
        os.close(fd)
        child_output = Path(path)
        child_output.unlink()
        env = os.environ.copy()
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        env["TORCHINDUCTOR_LAYOUT_OPTIMIZATION"] = "0"
        env["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--mode", mode,
            "--seed", str(seed),
            "--timesteps", str(timesteps),
            "--child-output", str(child_output),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            print(completed.stdout, end="", flush=True)
            if completed.returncode:
                raise RuntimeError(
                    f"holdout child failed: mode={mode} seed={seed} "
                    f"exit={completed.returncode}")
            if not child_output.is_file():
                raise RuntimeError(
                    f"holdout child produced no result: mode={mode} seed={seed}")
            return json.loads(child_output.read_text())
        finally:
            child_output.unlink(missing_ok=True)


def _parent(comparison_name="compile", output=None, timesteps=None):
    from benchmarks import learning_quality as learning

    holdout = HOLDOUTS[comparison_name]
    modes = holdout["modes"]
    seeds = holdout["seeds"]
    timesteps = _validated_timesteps(
        holdout["timesteps"] if timesteps is None else timesteps)
    output = Path(holdout["output"] if output is None else output)
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must be 0")
    # Keep top-level metadata consistent with the fresh child processes. The
    # measured runs still set these independently inside every child.
    learning.torch.set_num_threads(TORCH_THREADS)
    learning.torch.set_num_interop_threads(TORCH_INTEROP_THREADS)
    parent_system = learning._system_metadata()
    runs = []
    run_order = []
    for index, seed in enumerate(seeds):
        for mode in _mode_order(index, modes):
            run = _run_isolated_child(mode, seed, timesteps)
            _validate_child_provenance(run, parent_system)
            runs.append(run)
            run_order.append({"mode": mode, "seed": seed})

    summaries = [run["summary"] for run in runs]
    comparison = learning.compare_modes(
        summaries,
        baseline_mode=holdout["baseline_mode"],
        candidate_mode=holdout["candidate_mode"],
        thresholds=learning.DEFAULT_THRESHOLDS,
        min_ratio=holdout["min_quality_ratio"],
        bootstrap_guard=holdout["bootstrap_guard"],
        max_step_ratio=holdout["max_step_ratio"],
        require_candidate_host_alias=True,
    )
    promotion_eligible = _promotion_eligible(
        holdout, timesteps, comparison)
    report = {
        "schema": holdout["schema"],
        "environment": "breakout",
        "preregistered_before_execution": _preregistration(
            holdout, seeds, timesteps),
        "protocol": {
            "comparison": comparison_name,
            "modes": list(modes),
            "baseline_mode": holdout["baseline_mode"],
            "candidate_mode": holdout["candidate_mode"],
            "precision_by_mode": dict(holdout["precision_by_mode"]),
            "agents": AGENTS,
            "horizon": HORIZON,
            "minibatch_size": MINIBATCH_SIZE,
            "epochs": timesteps // (AGENTS * HORIZON),
            "timesteps_per_run": timesteps,
            "threads": THREADS,
            "torch_threads": TORCH_THREADS,
            "torch_interop_threads": TORCH_INTEROP_THREADS,
            "run_order": run_order,
        },
        "system": parent_system,
        "runs": runs,
        "comparison": comparison,
        "promotion_eligible": promotion_eligible,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(comparison, indent=2, sort_keys=True), flush=True)
    print(f"wrote {output}", flush=True)
    return 0 if promotion_eligible else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument(
        "--comparison", choices=tuple(HOLDOUTS), default="compile")
    parser.add_argument(
        "--mode", choices=("mps", *COMPILED_MODES))
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--timesteps", type=_validated_timesteps,
        help=(
            "steps per child run; defaults to the selected comparison's "
            "preregistered budget; overrides are explicitly exploratory and "
            "cannot produce a promotion-eligible exit status"),
    )
    parser.add_argument("--child-output", type=Path)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    if options.child:
        if options.mode is None or options.seed is None or options.child_output is None:
            parser.error("child mode requires --mode, --seed, and --child-output")
        _child(
            options.mode,
            options.seed,
            options.child_output,
            TIMESTEPS if options.timesteps is None else options.timesteps,
        )
        return 0
    return _parent(
        options.comparison,
        output=options.output,
        timesteps=options.timesteps,
    )


if __name__ == "__main__":
    raise SystemExit(main())
