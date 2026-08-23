"""Paired learning-quality and time-to-score benchmark.

Raw steps/second does not prove that an accelerator port still learns.  This
benchmark runs the same portable Torch trainer, CPU-native simulator, model,
batch shape, and fixed seeds on two execution modes.  It records every epoch
before comparing episode-weighted learning curves at matched environment
steps.

The default Breakout gate is deliberately bounded (five paired seeds, 32
epochs, 8,388,608 steps per run).  On the Apple M5 Pro used to tune the port,
the CPU half takes about two minutes in total.  The complete CPU/MPS gate is
expected to finish in roughly three minutes.

Build the CPU-native Breakout extension once, then run::

    PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python \
      benchmarks/learning_quality.py --output work/learning_quality.json

The same harness can compare CUDA on an NVIDIA host by selecting
``--modes cpu cuda --candidate cuda`` after building the appropriate native
extension.  Float32 is intentional: reduced precision is a separate quality
experiment, not an equivalent execution-path comparison.
"""

import argparse
from contextlib import contextmanager
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np
import torch

from pufferlib import _C
from pufferlib.device import resolve_device, resolve_rollout_device, synchronize
from pufferlib.pufferl import load_config
from pufferlib.torch_pufferl import PuffeRL, load_policy


DEFAULT_SEEDS = (11, 23, 37, 53, 71)
DEFAULT_THRESHOLDS = (2.0, 4.0)
DEFAULT_AGENTS = 4096
DEFAULT_HORIZON = 64
DEFAULT_EPOCHS = 32
DEFAULT_TIMESTEPS = DEFAULT_AGENTS * DEFAULT_HORIZON * DEFAULT_EPOCHS


@contextmanager
def _clean_argv():
    previous = sys.argv
    sys.argv = [previous[0]]
    try:
        yield
    finally:
        sys.argv = previous


def make_args(env_name, mode, *, agents, horizon, minibatch_size,
        timesteps, threads, mps_host_alias="auto"):
    """Construct one effective config without changing learning semantics."""
    with _clean_argv():
        args = load_config(env_name)

    try:
        training_device, rollout_device = {
            "cpu": ("cpu", "cpu"),
            "hybrid": ("mps", "cpu"),
            "mps": ("mps", "mps"),
            "mps_compile": ("mps", "mps"),
            "cuda": ("cuda", "cuda"),
        }[mode]
    except KeyError as exc:
        raise ValueError(f"unsupported execution mode: {mode}") from exc

    batch_size = int(agents) * int(horizon)
    if int(timesteps) <= 0 or int(timesteps) % batch_size:
        raise ValueError(
            f"timesteps must be a positive multiple of batch size {batch_size}")
    if int(minibatch_size) <= 0 or int(minibatch_size) % int(horizon):
        raise ValueError("minibatch_size must be a positive multiple of horizon")
    if int(minibatch_size) > batch_size:
        raise ValueError("minibatch_size cannot exceed agents * horizon")

    args["slowly"] = True
    args["profile"] = False
    args["rank"] = 0
    args["world_size"] = 1
    args["gpu_id"] = 0
    args["nccl_id"] = b""
    args["torch"]["device"] = training_device
    args["torch"]["rollout_device"] = rollout_device
    args["torch"]["amp_dtype"] = "float32"
    args["torch"]["mps_host_alias"] = mps_host_alias
    args["torch"]["compile_policy"] = (
        "inductor" if mode == "mps_compile" else "off")
    args["vec"]["total_agents"] = int(agents)
    args["vec"]["num_buffers"] = 1
    args["vec"]["num_threads"] = int(threads)
    args["train"]["horizon"] = int(horizon)
    args["train"]["minibatch_size"] = int(minibatch_size)
    args["train"]["total_timesteps"] = int(timesteps)
    args["train"]["gpus"] = 1
    return args


def seed_everything(seed, *, strict_determinism=False):
    """Reset every Python-side RNG used by the portable trainer.

    MPS currently reports no deterministic implementation for scatter-reduce,
    which is used by the production backward pass. The default therefore uses
    fixed seeds plus paired statistical replication without changing the
    production kernels. ``strict_determinism`` remains useful for CPU-only
    diagnostics and fails closed if a selected backend cannot honor it.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(
        bool(strict_determinism), warn_only=False)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def deterministic_algorithms_for_mode(mode, *, require_all=False):
    """Use strict kernels where supported without changing the MPS path."""
    if require_all:
        return True
    return mode not in ("mps", "mps_compile", "hybrid")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _finite_losses(losses):
    return bool(losses) and all(
        math.isfinite(float(value)) for value in losses.values())


def run_one(env_name, mode, seed, *, agents, horizon, minibatch_size,
        timesteps, threads, mps_host_alias="auto", strict_determinism=False):
    """Run one isolated logical replicate and return every epoch record."""
    deterministic_algorithms = deterministic_algorithms_for_mode(
        mode, require_all=strict_determinism)
    seed_everything(seed, strict_determinism=deterministic_algorithms)
    args = make_args(
        env_name,
        mode,
        agents=agents,
        horizon=horizon,
        minibatch_size=minibatch_size,
        timesteps=timesteps,
        threads=threads,
        mps_host_alias=mps_host_alias,
    )
    device = resolve_device(args["torch"]["device"], native_cuda=False)
    vec = _C.create_vec(args, 0)
    try:
        rollout_device = resolve_rollout_device(
            args["torch"]["rollout_device"],
            device,
            vec_gpu=bool(vec.gpu),
            total_agents=vec.total_agents,
            mps_threshold=args["torch"].get("mps_rollout_threshold", -1),
        )
        policy = load_policy(args, vec, device=rollout_device)
        trainer = PuffeRL(
            args,
            vec,
            policy,
            verbose=False,
            device=device,
            rollout_device=rollout_device,
        )
    except Exception:
        vec.close()
        raise

    batch_size = int(agents) * int(horizon)
    epochs = int(timesteps) // batch_size
    records = []
    synchronize(device)
    synchronize(rollout_device)
    compile_startup_seconds = float(trainer.policy_compile_startup_seconds)
    sampler_startup_seconds = float(
        trainer.rollout_sampler_startup_seconds)
    optimization_startup_seconds = float(
        trainer.optimization_startup_seconds)
    # Policy and sampler preflight before the first epoch. Shift the logical
    # start back so every cold setup cost remains included in uptime, threshold
    # wall time, full-run wall time, and speedup acceptance.
    started = time.perf_counter() - optimization_startup_seconds
    try:
        for epoch in range(1, epochs + 1):
            synchronize(device)
            synchronize(rollout_device)
            epoch_started = time.perf_counter()
            trainer.rollouts()
            trainer.train()
            synchronize(device)
            synchronize(rollout_device)

            losses = {key: float(value) for key, value in trainer.losses.items()}
            if not _finite_losses(losses):
                raise RuntimeError(
                    f"non-finite training loss in mode={mode} seed={seed}: {losses}")
            env = dict(trainer.env_logs)
            score = env.get("score")
            episode_count = float(env.get("n", 0.0))
            records.append({
                "epoch": epoch,
                "agent_steps": int(trainer.global_step),
                "uptime_seconds": time.perf_counter() - started,
                "epoch_seconds": time.perf_counter() - epoch_started,
                "episode_count": episode_count,
                "score": None if score is None else float(score),
                "normalized_score": (
                    None if env.get("perf") is None else float(env["perf"])),
                "losses": losses,
            })
    finally:
        trainer.close()

    result = {
        "environment": env_name,
        "mode": mode,
        "seed": int(seed),
        "training_device": str(device),
        "rollout_device": str(rollout_device),
        "mps_host_alias_io": bool(trainer.mps_host_alias_io),
        "host_horizon_io": bool(trainer.host_horizon_io),
        "requested_policy_compile": trainer.policy_compile_requested,
        "effective_policy_compile": trainer.policy_compile_effective,
        "policy_compile_reason": trainer.policy_compile_reason,
        "policy_compile_preflight": bool(trainer.policy_compile_preflight),
        "policy_compile_wrapper_verified": bool(
            trainer.policy_compile_wrapper_verified),
        "policy_compile_startup_seconds": compile_startup_seconds,
        "requested_rollout_sampler": trainer.rollout_sampler_requested,
        "effective_rollout_sampler": trainer.rollout_sampler_effective,
        "rollout_sampler_reason": trainer.rollout_sampler_reason,
        "rollout_sampler_startup_seconds": sampler_startup_seconds,
        "optimization_startup_seconds": optimization_startup_seconds,
        "deterministic_algorithms": deterministic_algorithms,
        "total_seconds": time.perf_counter() - started,
        "effective_config": {
            section: _json_safe(args.get(section, {}))
            for section in ("torch", "train", "vec", "env", "policy")
        },
        "records": records,
    }
    return result


def _window_score(records, index, window_epochs, min_window_episodes):
    start = max(0, index + 1 - int(window_epochs))
    episode_count = 0.0
    score_sum = 0.0
    for record in records[start:index + 1]:
        count = float(record.get("episode_count") or 0.0)
        score = record.get("score")
        if count <= 0 or score is None or not math.isfinite(float(score)):
            continue
        episode_count += count
        score_sum += count * float(score)
    if episode_count < float(min_window_episodes):
        return None, episode_count
    return score_sum / episode_count, episode_count


def rolling_curve(records, *, window_epochs=4, min_window_episodes=2000):
    """Return a trailing, episode-weighted score for each training epoch."""
    curve = []
    for index, record in enumerate(records):
        score, episodes = _window_score(
            records, index, window_epochs, min_window_episodes)
        curve.append({
            "agent_steps": int(record["agent_steps"]),
            "uptime_seconds": float(record["uptime_seconds"]),
            "score": score,
            "window_episode_count": episodes,
        })
    return curve


def time_to_threshold(curve, threshold, *, sustain_epochs=2):
    """First checkpoint completing ``sustain_epochs`` above a threshold."""
    consecutive = 0
    for point in curve:
        score = point.get("score")
        if score is not None and float(score) >= float(threshold):
            consecutive += 1
            if consecutive >= int(sustain_epochs):
                return {
                    "agent_steps": int(point["agent_steps"]),
                    "uptime_seconds": float(point["uptime_seconds"]),
                    "score": float(score),
                }
        else:
            consecutive = 0
    return None


def _tail_score(records, tail_fraction):
    final_steps = int(records[-1]["agent_steps"])
    cutoff = final_steps * (1.0 - float(tail_fraction))
    selected = [record for record in records if record["agent_steps"] > cutoff]
    count = sum(float(record.get("episode_count") or 0.0) for record in selected)
    if count <= 0:
        return None
    return sum(
        float(record.get("episode_count") or 0.0) * float(record.get("score") or 0.0)
        for record in selected
    ) / count


def _mean_score_auc(curve, final_steps):
    """Step-normalized area under the trailing-score learning curve."""
    usable = [point for point in curve if point.get("score") is not None]
    if not usable:
        return None
    xs = [0.0]
    ys = [0.0]
    for point in usable:
        xs.append(float(point["agent_steps"]))
        ys.append(float(point["score"]))
    if xs[-1] < final_steps:
        xs.append(float(final_steps))
        ys.append(ys[-1])
    return float(np.trapezoid(ys, xs) / float(final_steps))


def summarize_run(run, *, thresholds=DEFAULT_THRESHOLDS, window_epochs=4,
        min_window_episodes=2000, sustain_epochs=2, tail_fraction=0.25):
    records = run.get("records", [])
    if not records:
        raise ValueError("run contains no epoch records")
    curve = rolling_curve(
        records,
        window_epochs=window_epochs,
        min_window_episodes=min_window_episodes,
    )
    final_steps = int(records[-1]["agent_steps"])
    total_seconds = float(run["total_seconds"])
    finite = all(
        _finite_losses(record.get("losses", {})) for record in records)
    return {
        "mode": run["mode"],
        "seed": int(run["seed"]),
        "training_device": run.get("training_device"),
        "rollout_device": run.get("rollout_device"),
        "mps_host_alias_io": bool(run.get("mps_host_alias_io", False)),
        "requested_policy_compile": run.get(
            "requested_policy_compile", "off"),
        "effective_policy_compile": run.get(
            "effective_policy_compile", "off"),
        "policy_compile_reason": run.get("policy_compile_reason"),
        "policy_compile_preflight": bool(
            run.get("policy_compile_preflight", False)),
        "policy_compile_wrapper_verified": bool(
            run.get("policy_compile_wrapper_verified", False)),
        "policy_compile_startup_seconds": float(
            run.get("policy_compile_startup_seconds", 0.0)),
        "requested_rollout_sampler": run.get(
            "requested_rollout_sampler", "fused_mps_philox"),
        "effective_rollout_sampler": run.get(
            "effective_rollout_sampler", "torch_multinomial"),
        "rollout_sampler_reason": run.get("rollout_sampler_reason"),
        "rollout_sampler_startup_seconds": float(
            run.get("rollout_sampler_startup_seconds", 0.0)),
        "optimization_startup_seconds": float(
            run.get("optimization_startup_seconds", 0.0)),
        "deterministic_algorithms": bool(
            run.get("deterministic_algorithms", False)),
        "finite": finite,
        "final_steps": final_steps,
        "total_seconds": total_seconds,
        "steps_per_second": final_steps / total_seconds,
        "tail_fraction": float(tail_fraction),
        "tail_score": _tail_score(records, tail_fraction),
        "mean_score_auc": _mean_score_auc(curve, final_steps),
        "time_to_threshold": {
            str(float(threshold)): time_to_threshold(
                curve, threshold, sustain_epochs=sustain_epochs)
            for threshold in thresholds
        },
        "curve": curve,
    }


def _median(values):
    values = [float(value) for value in values if value is not None]
    return statistics.median(values) if values else None


def _ratio(candidate, baseline):
    if candidate is None or baseline is None or float(baseline) <= 0:
        return None
    return float(candidate) / float(baseline)


def _bootstrap_median_lower(values, *, samples=10_000, percentile=5.0,
        seed=20260822):
    values = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if values.size == 0:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(int(samples), values.size))
    medians = np.median(values[indices], axis=1)
    return float(np.percentile(medians, float(percentile)))


def compare_modes(summaries, *, baseline_mode="cpu", candidate_mode="mps",
        thresholds=DEFAULT_THRESHOLDS, min_ratio=0.90,
        bootstrap_guard=0.75, max_step_ratio=1.10,
        require_candidate_host_alias=False):
    """Apply predeclared paired equivalence and time-to-score criteria."""
    baseline = {
        int(run["seed"]): run for run in summaries
        if run["mode"] == baseline_mode
    }
    candidate = {
        int(run["seed"]): run for run in summaries
        if run["mode"] == candidate_mode
    }
    seeds = sorted(set(baseline) & set(candidate))
    if not seeds:
        raise ValueError("baseline and candidate have no paired seeds")

    paired = []
    for seed in seeds:
        base = baseline[seed]
        cand = candidate[seed]
        paired.append({
            "seed": seed,
            "finite": bool(base["finite"] and cand["finite"]),
            "tail_score_ratio": _ratio(cand["tail_score"], base["tail_score"]),
            "mean_score_auc_ratio": _ratio(
                cand["mean_score_auc"], base["mean_score_auc"]),
            "wall_time_speedup": _ratio(
                base["total_seconds"], cand["total_seconds"]),
        })

    tail_ratios = [pair["tail_score_ratio"] for pair in paired]
    auc_ratios = [pair["mean_score_auc_ratio"] for pair in paired]
    speedups = [pair["wall_time_speedup"] for pair in paired]
    checks = {
        "all_runs_finite": all(pair["finite"] for pair in paired),
        "all_quality_pairs_complete": (
            all(value is not None for value in tail_ratios)
            and all(value is not None for value in auc_ratios)),
        "median_tail_score_ratio": (
            _median(tail_ratios) is not None
            and _median(tail_ratios) >= float(min_ratio)),
        "median_mean_score_auc_ratio": (
            _median(auc_ratios) is not None
            and _median(auc_ratios) >= float(min_ratio)),
        "tail_score_bootstrap_guard": (
            _bootstrap_median_lower(tail_ratios) is not None
            and _bootstrap_median_lower(tail_ratios) >= float(bootstrap_guard)),
        "mean_score_auc_bootstrap_guard": (
            _bootstrap_median_lower(auc_ratios) is not None
            and _bootstrap_median_lower(auc_ratios) >= float(bootstrap_guard)),
        "median_wall_time_not_slower": (
            _median(speedups) is not None and _median(speedups) >= 1.0),
    }
    if require_candidate_host_alias:
        checks["candidate_mps_host_alias_active"] = all(
            candidate[seed].get("training_device") == "mps"
            and candidate[seed].get("rollout_device") == "mps"
            and candidate[seed].get("mps_host_alias_io") is True
            for seed in seeds
        )
    if candidate_mode == "mps_compile":
        checks["candidate_policy_compile_active"] = all(
            candidate[seed].get("requested_policy_compile") == "inductor"
            and candidate[seed].get("effective_policy_compile") == "inductor"
            and candidate[seed].get("policy_compile_preflight") is True
            and candidate[seed].get(
                "policy_compile_wrapper_verified") is True
            and float(candidate[seed].get(
                "policy_compile_startup_seconds", 0.0)) > 0.0
            and candidate[seed].get("training_device") == "mps"
            and candidate[seed].get("rollout_device") == "mps"
            and candidate[seed].get("mps_host_alias_io") is True
            for seed in seeds
        )
        checks["candidate_rollout_sampler_active"] = all(
            candidate[seed].get(
                "requested_rollout_sampler") == "fused_mps_philox"
            and candidate[seed].get(
                "effective_rollout_sampler") == "fused_mps_philox"
            for seed in seeds
        )

    threshold_results = {}
    for threshold in thresholds:
        key = str(float(threshold))
        cpu_reached = []
        candidate_reached = []
        paired_step_ratios = []
        paired_wall_speedups = []
        for seed in seeds:
            base_hit = baseline[seed]["time_to_threshold"].get(key)
            cand_hit = candidate[seed]["time_to_threshold"].get(key)
            if base_hit is not None:
                cpu_reached.append(seed)
            if cand_hit is not None:
                candidate_reached.append(seed)
            if base_hit is not None and cand_hit is not None:
                paired_step_ratios.append(
                    cand_hit["agent_steps"] / base_hit["agent_steps"])
                paired_wall_speedups.append(
                    base_hit["uptime_seconds"] / cand_hit["uptime_seconds"])

        # A threshold is a quality gate only when a majority of baseline seeds
        # reached it. This rule is fixed up front; an unreachable threshold is
        # reported rather than silently scored as a candidate failure.
        gated = len(cpu_reached) >= math.ceil(len(seeds) / 2)
        reach_ok = (
            not gated or len(candidate_reached) >= len(cpu_reached) - 1)
        step_ok = (
            not gated or (
                len(paired_step_ratios) >= len(cpu_reached) - 1
                and _median(paired_step_ratios) <= float(max_step_ratio)))
        checks[f"threshold_{key}_reach_coverage"] = reach_ok
        checks[f"threshold_{key}_step_equivalence"] = step_ok
        threshold_results[key] = {
            "gated": gated,
            "baseline_reached_seeds": cpu_reached,
            "candidate_reached_seeds": candidate_reached,
            "median_candidate_to_baseline_steps": _median(paired_step_ratios),
            "median_wall_time_speedup": _median(paired_wall_speedups),
        }

    return {
        "baseline_mode": baseline_mode,
        "candidate_mode": candidate_mode,
        "paired_seed_count": len(seeds),
        "paired": paired,
        "aggregate": {
            "median_tail_score_ratio": _median(tail_ratios),
            "tail_score_ratio_bootstrap_90pct_lower": _bootstrap_median_lower(
                tail_ratios),
            "median_mean_score_auc_ratio": _median(auc_ratios),
            "mean_score_auc_ratio_bootstrap_90pct_lower": _bootstrap_median_lower(
                auc_ratios),
            "median_wall_time_speedup": _median(speedups),
        },
        "thresholds": threshold_results,
        "acceptance": {
            "passed": all(checks.values()),
            "checks": checks,
            "criteria": {
                "median_quality_ratio_min": float(min_ratio),
                "bootstrap_90pct_lower_guard_min": float(bootstrap_guard),
                "median_steps_to_score_ratio_max": float(max_step_ratio),
                "threshold_gate_baseline_reach_fraction": "at least half",
                "allowed_candidate_reach_count_deficit": 1,
                "median_wall_time_speedup_min": 1.0,
                "candidate_mps_host_alias_required": bool(
                    require_candidate_host_alias),
                "candidate_policy_compile_required": (
                    candidate_mode == "mps_compile"),
                "candidate_rollout_sampler_required": (
                    candidate_mode == "mps_compile"),
            },
        },
    }


def _mode_order(modes, seed_index):
    """Counterbalance thermal/order effects while retaining paired seeds."""
    modes = list(modes)
    return modes if seed_index % 2 == 0 else list(reversed(modes))


def _apple_system_metadata():
    """Load the sibling benchmark's canonical provenance implementation.

    Both benchmark files are also supported as directly executed scripts, so
    importing through a possibly shadowed ``benchmarks`` namespace is brittle.
    Loading the sibling by path keeps this a one-way dependency and avoids a
    package-level import cycle.
    """
    benchmark_path = Path(__file__).with_name("apple_silicon.py")
    spec = importlib.util.spec_from_file_location(
        "apple_silicon_source_metadata", benchmark_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load metadata source: {benchmark_path}")
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    return benchmark.system_metadata()


def _system_metadata():
    metadata = dict(_apple_system_metadata())
    metadata.update({
        # Retain the original learning-report spelling while also preserving
        # apple_silicon.py's canonical ``torch_git_revision`` field.
        "torch_git_version": metadata.get("torch_git_revision"),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "pytorch_enable_mps_fallback": os.environ.get(
            "PYTORCH_ENABLE_MPS_FALLBACK"),
    })
    return metadata


def run_suite(options):
    compiled_env = getattr(_C, "env_name", None)
    if compiled_env != options.env:
        raise RuntimeError(
            f"_C was built for {compiled_env!r}, not {options.env!r}")
    if any(mode in ("mps", "mps_compile", "hybrid")
            for mode in options.modes):
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
            raise RuntimeError(
                "MPS quality evidence requires PYTORCH_ENABLE_MPS_FALLBACK=0")
        resolve_device("mps")
    if "cuda" in options.modes:
        resolve_device("cuda")

    results = []
    for seed_index, seed in enumerate(options.seeds):
        for mode in _mode_order(options.modes, seed_index):
            result = run_one(
                options.env,
                mode,
                seed,
                agents=options.agents,
                horizon=options.horizon,
                minibatch_size=options.minibatch_size,
                timesteps=options.timesteps,
                threads=options.threads,
                mps_host_alias=options.mps_host_alias,
                strict_determinism=options.strict_determinism,
            )
            result["summary"] = summarize_run(
                result,
                thresholds=options.thresholds,
                window_epochs=options.window_epochs,
                min_window_episodes=options.min_window_episodes,
                sustain_epochs=options.sustain_epochs,
                tail_fraction=options.tail_fraction,
            )
            results.append(result)
            print(json.dumps({
                "event": "completed_run",
                "mode": mode,
                "seed": seed,
                "summary": result["summary"],
            }, sort_keys=True), flush=True)
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    summaries = [result["summary"] for result in results]
    comparison = compare_modes(
        summaries,
        baseline_mode=options.baseline,
        candidate_mode=options.candidate,
        thresholds=options.thresholds,
        min_ratio=options.min_quality_ratio,
        bootstrap_guard=options.bootstrap_guard,
        max_step_ratio=options.max_step_ratio,
        require_candidate_host_alias=(
            options.require_mps_host_alias
            and options.candidate in ("mps", "mps_compile")),
    )
    return {
        "schema": "pufferlib-learning-quality-v1",
        "environment": options.env,
        "system": _system_metadata(),
        "protocol": {
            "modes": list(options.modes),
            "seeds": list(options.seeds),
            "counterbalanced_mode_order": True,
            "agents": options.agents,
            "horizon": options.horizon,
            "minibatch_size": options.minibatch_size,
            "timesteps_per_run": options.timesteps,
            "epochs_per_run": options.timesteps // (
                options.agents * options.horizon),
            "threads": options.threads,
            "thresholds": list(options.thresholds),
            "window_epochs": options.window_epochs,
            "min_window_episodes": options.min_window_episodes,
            "sustain_epochs": options.sustain_epochs,
            "tail_fraction": options.tail_fraction,
            "precision": "float32",
            "fixed_rng_seeds": True,
            "determinism_policy": (
                "strict on every mode"
                if options.strict_determinism else
                "strict on CPU/CUDA; disabled on MPS/hybrid because the "
                "production MPS scatter_reduce backward has no deterministic "
                "implementation"),
            "strict_deterministic_algorithms_required_on_all_modes": (
                options.strict_determinism),
            "mps_determinism_limit": (
                "PyTorch MPS scatter_reduce has no deterministic "
                "implementation; paired multi-seed statistics cover the "
                "production path"
                if not options.strict_determinism else None),
        },
        "runs": results,
        "comparison": comparison,
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="breakout")
    parser.add_argument(
        "--modes", nargs="+",
        choices=("cpu", "hybrid", "mps", "mps_compile", "cuda"),
        default=("cpu", "mps"))
    parser.add_argument("--baseline", default="cpu")
    parser.add_argument("--candidate", default="mps")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--agents", type=int, default=DEFAULT_AGENTS)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--minibatch-size", type=int, default=65_536)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--threads", type=int, default=18)
    parser.add_argument("--torch-threads", type=int, default=12)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument(
        "--strict-determinism",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "require torch deterministic algorithms; unsupported by the "
            "current MPS scatter-reduce backward path"),
    )
    parser.add_argument(
        "--mps-host-alias", choices=("auto", "on", "off"), default="auto")
    parser.add_argument(
        "--require-mps-host-alias",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require the optimized direct-MPS shared-memory path to be active",
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--window-epochs", type=int, default=4)
    parser.add_argument("--min-window-episodes", type=int, default=2000)
    parser.add_argument("--sustain-epochs", type=int, default=2)
    parser.add_argument("--tail-fraction", type=float, default=0.25)
    parser.add_argument("--min-quality-ratio", type=float, default=0.90)
    parser.add_argument("--bootstrap-guard", type=float, default=0.75)
    parser.add_argument("--max-step-ratio", type=float, default=1.10)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _validate_options(parser, options):
    if options.baseline not in options.modes:
        parser.error("--baseline must be included in --modes")
    if options.candidate not in options.modes:
        parser.error("--candidate must be included in --modes")
    if options.baseline == options.candidate:
        parser.error("--baseline and --candidate must differ")
    if len(set(options.seeds)) != len(options.seeds):
        parser.error("--seeds must be unique")
    if len(options.seeds) < 3:
        parser.error("at least three paired seeds are required")
    if options.threads < 1 or options.torch_threads < 1:
        parser.error("thread counts must be positive")
    if options.torch_interop_threads < 1:
        parser.error("--torch-interop-threads must be positive")
    if options.window_epochs < 1 or options.sustain_epochs < 1:
        parser.error("window and sustain epochs must be positive")
    if options.min_window_episodes < 1:
        parser.error("--min-window-episodes must be positive")
    if not 0 < options.tail_fraction <= 1:
        parser.error("--tail-fraction must be in (0, 1]")
    if options.min_quality_ratio <= 0 or options.bootstrap_guard <= 0:
        parser.error("quality ratios must be positive")
    if options.max_step_ratio <= 0:
        parser.error("--max-step-ratio must be positive")
    try:
        make_args(
            options.env,
            options.baseline,
            agents=options.agents,
            horizon=options.horizon,
            minibatch_size=options.minibatch_size,
            timesteps=options.timesteps,
            threads=options.threads,
            mps_host_alias=options.mps_host_alias,
        )
    except (ValueError, AssertionError) as exc:
        parser.error(str(exc))


def main():
    parser = _parser()
    options = parser.parse_args()
    _validate_options(parser, options)
    torch.set_num_threads(options.torch_threads)
    torch.set_num_interop_threads(options.torch_interop_threads)
    report = run_suite(options)
    if options.output is not None:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = options.output.with_suffix(options.output.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(report, output, indent=2, sort_keys=True)
            output.write("\n")
        temporary.replace(options.output)
        print(f"wrote {options.output}", flush=True)
    print(json.dumps(report["comparison"], indent=2, sort_keys=True))
    raise SystemExit(0 if report["comparison"]["acceptance"]["passed"] else 1)


if __name__ == "__main__":
    main()
