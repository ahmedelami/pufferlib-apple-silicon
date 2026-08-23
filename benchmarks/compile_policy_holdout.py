"""Isolated eager-vs-compiled MPS learning-quality holdout.

Every replicate runs in a fresh process, eager/compiled order alternates, and
every compiled replicate gets a unique empty Inductor cache. The discovery
seeds are never pooled into the holdout result. Build Breakout first, then run::

    PYTORCH_ENABLE_MPS_FALLBACK=0 python \
      benchmarks/compile_policy_holdout.py
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
SEEDS = (73, 79, 83, 89, 97, 101, 103, 107, 109, 113)
AGENTS = 4096
HORIZON = 64
MINIBATCH_SIZE = 65_536
TIMESTEPS = AGENTS * HORIZON * 32
THREADS = 18
TORCH_THREADS = 12
TORCH_INTEROP_THREADS = 1


def _mode_order(index):
    return (
        ("mps", "mps_compile") if int(index) % 2 == 0
        else ("mps_compile", "mps"))


def _validate_child_run(mode, run):
    expected_compile = "inductor" if mode == "mps_compile" else "off"
    if run["effective_policy_compile"] != expected_compile:
        raise RuntimeError(
            f"{mode} effective compiler was "
            f"{run['effective_policy_compile']!r}, expected {expected_compile!r}")
    if mode == "mps_compile" and not (
            run["policy_compile_preflight"]
            and run["policy_compile_wrapper_verified"]):
        raise RuntimeError(
            "compiled candidate did not verify wrappers and graph preflight")
    if mode == "mps_compile" and not (
            run["requested_rollout_sampler"] == "fused_mps_philox"
            and run["effective_rollout_sampler"] == "fused_mps_philox"):
        raise RuntimeError(
            "compiled candidate did not activate the fused MPS sampler")
    if not run["mps_host_alias_io"]:
        raise RuntimeError(f"{mode} did not activate MPS host aliasing")


def _child(mode, seed, output):
    from benchmarks import learning_quality as learning

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
        timesteps=TIMESTEPS,
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


def _parent():
    from benchmarks import learning_quality as learning

    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must be 0")
    # Keep top-level metadata consistent with the fresh child processes. The
    # measured runs still set these independently inside every child.
    learning.torch.set_num_threads(TORCH_THREADS)
    learning.torch.set_num_interop_threads(TORCH_INTEROP_THREADS)
    runs = []
    run_order = []
    for index, seed in enumerate(SEEDS):
        modes = _mode_order(index)
        for mode in modes:
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
                    "--child-output", str(child_output),
                ]
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
                runs.append(json.loads(child_output.read_text()))
                child_output.unlink()
                run_order.append({"mode": mode, "seed": seed})

    summaries = [run["summary"] for run in runs]
    comparison = learning.compare_modes(
        summaries,
        baseline_mode="mps",
        candidate_mode="mps_compile",
        thresholds=learning.DEFAULT_THRESHOLDS,
        min_ratio=0.90,
        bootstrap_guard=0.75,
        max_step_ratio=1.10,
        require_candidate_host_alias=True,
    )
    report = {
        "schema": "pufferlib-compile-policy-holdout-v1",
        "environment": "breakout",
        "preregistered_before_execution": {
            "discovery_seeds_excluded": [11, 23, 37, 53, 71],
            "holdout_seeds": list(SEEDS),
            "run_isolation": "one fresh process per mode and seed",
            "order": "alternating eager-compiled / compiled-eager",
            "compiler_cache": "unique empty cache per replicate",
            "compile_startup_included": True,
            "acceptance_criteria_unchanged": True,
            "invalid_run_policy": "restart the complete holdout unchanged",
        },
        "protocol": {
            "agents": AGENTS,
            "horizon": HORIZON,
            "minibatch_size": MINIBATCH_SIZE,
            "epochs": 32,
            "timesteps_per_run": TIMESTEPS,
            "threads": THREADS,
            "torch_threads": TORCH_THREADS,
            "torch_interop_threads": TORCH_INTEROP_THREADS,
            "run_order": run_order,
        },
        "system": learning._system_metadata(),
        "runs": runs,
        "comparison": comparison,
    }
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUTPUT)
    print(json.dumps(comparison, indent=2, sort_keys=True), flush=True)
    print(f"wrote {OUTPUT}", flush=True)
    return 0 if comparison["acceptance"]["passed"] else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--mode", choices=("mps", "mps_compile"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--child-output", type=Path)
    options = parser.parse_args()
    if options.child:
        if options.mode is None or options.seed is None or options.child_output is None:
            parser.error("child mode requires --mode, --seed, and --child-output")
        _child(options.mode, options.seed, options.child_output)
        return 0
    return _parent()


if __name__ == "__main__":
    raise SystemExit(main())
