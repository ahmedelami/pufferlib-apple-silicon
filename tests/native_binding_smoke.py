#!/usr/bin/env python3
"""Build and exercise the native Ocean bindings in isolated processes.

The native extension is statically linked to one environment at a time.  Run a
single already-built binding with::

    python tests/native_binding_smoke.py breakout

Or compile and smoke every binding, continuing past native crashes so the
result is a complete matrix::

    python tests/native_binding_smoke.py --build-all --build-mode mps

This is intentionally a one-reset/one-step ABI smoke test, not an environment
correctness test.  Each vector uses one complete environment and one worker to
keep memory bounded and to avoid partial multi-agent environments.
"""

from __future__ import annotations

import argparse
import ast
import configparser
import ctypes
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

# Bindings without a one-to-one config file.
CONFIG_ALIASES = {
    "squared_continuous": "squared",
}

# These bindings use compile-time or derived agent counts rather than a scalar
# env.num_agents in their ini.  Values are the smallest complete environment
# for the checked-in configuration.
FIXED_AGENTS = {
    "boids": 64,
    "craftax_classic": 1,
    "minimal": 8,
    "scape": 8,
    "target": 8,
}

# Scape and Tactical deliberately reject vector creation: upstream has no
# observation implementation and their steps do not consume policy actions.
EXPECTED_UNSUPPORTED = {
    "scape": "upstream RL vector API is incomplete",
    "tactical": "upstream RL vector API is incomplete",
}

# Drive's generated Waymo-derived binaries are not checked into the repository.
# Import/build remains testable, but construction is not meaningful without one.
OPTIONAL_FIXTURES = {
    "drive": (ROOT / "drive_data/binaries/map_000.bin", "drive map binaries are absent"),
}


def _literal(value: str):
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _load_args(env_name: str) -> dict:
    """Load default + environment ini without importing the compiled module."""
    parser = configparser.ConfigParser()
    paths = [ROOT / "config/default.ini"]
    config_name = CONFIG_ALIASES.get(env_name, env_name)
    env_path = ROOT / f"config/{config_name}.ini"
    if env_path.exists():
        paths.append(env_path)
    elif env_name != "scape":
        raise AssertionError(f"no configuration for native binding {env_name!r}")
    parser.read([str(path) for path in paths])

    args = {
        section: {key: _literal(value) for key, value in parser[section].items()}
        for section in parser.sections()
    }
    args.setdefault("env", {})
    args.setdefault("vec", {})

    # Scape is an unconfigured tutorial binding.  These are the dimensions
    # used by its standalone example and do not affect the fixed ABI.
    if env_name == "scape":
        args["env"].update(width=1080, height=720)

    # Do not require the optional chess FEN curriculum for an ABI smoke.
    if env_name == "chess":
        args["env"]["fen_curric_pct"] = 0.0

    # Scanning one map is sufficient if the optional Drive dataset is present.
    if env_name == "drive":
        args["env"]["num_maps"] = 1

    return args


def _agents_per_env(env_name: str, env_args: dict) -> int:
    if env_name in FIXED_AGENTS:
        return FIXED_AGENTS[env_name]
    if env_name == "chess":
        return 2 if int(env_args.get("mode", 0)) == 1 else 1
    if env_name == "drone":
        return int(env_args["num_drones"])
    if env_name == "matsci":
        return int(env_args.get("num_agents", env_args.get("num_atoms", 2)))
    if env_name == "moba":
        return 5 if int(env_args.get("script_opponents", 0)) else 10
    if env_name == "shared_pool":
        # The current ini expresses a sweep list; py_dict_to_c_dict correctly
        # ignores it and the native binding uses its scalar default of eight.
        value = env_args.get("num_agents", 8)
        return int(value) if isinstance(value, (int, float)) else 8

    value = env_args.get("num_agents", 1)
    if not isinstance(value, (int, float)):
        raise AssertionError(f"{env_name}: non-scalar num_agents={value!r}")
    agents = int(value)
    if agents < 1:
        raise AssertionError(f"{env_name}: invalid num_agents={agents}")
    return agents


def _view(ptr: int, count: int, dtype: np.dtype) -> np.ndarray:
    if ptr == 0:
        raise AssertionError("native binding returned a null buffer pointer")
    ctype = np.ctypeslib.as_ctypes_type(np.dtype(dtype))
    return np.ctypeslib.as_array((ctype * count).from_address(ptr))


def _assert_finite(name: str, array: np.ndarray) -> None:
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        indices = np.argwhere(~np.isfinite(array)).reshape(-1)[:8].tolist()
        raise AssertionError(f"{name} contains non-finite values at {indices}")


def smoke_one(env_name: str) -> dict:
    fixture = OPTIONAL_FIXTURES.get(env_name)
    if fixture is not None and not fixture[0].exists():
        return {
            "env": env_name,
            "status": "skipped_optional_fixture",
            "reason": fixture[1],
        }

    args = _load_args(env_name)
    total_agents = _agents_per_env(env_name, args["env"])
    args["vec"] = {
        "total_agents": total_agents,
        "num_buffers": 1,
        "num_threads": 1,
    }

    from pufferlib import _C

    if _C.env_name != env_name:
        raise AssertionError(
            f"compiled extension is {_C.env_name!r}, requested {env_name!r}"
        )
    if _C.gpu != 0 or _C.precision_bytes != 4:
        raise AssertionError(
            f"expected CPU float32 simulator, got gpu={_C.gpu}, "
            f"precision_bytes={_C.precision_bytes}"
        )

    if env_name in EXPECTED_UNSUPPORTED:
        try:
            vec = _C.create_vec(args, 0)
        except RuntimeError:
            return {
                "env": env_name,
                "status": "expected_unsupported",
                "reason": EXPECTED_UNSUPPORTED[env_name],
            }
        else:
            vec.close()
            raise AssertionError(f"{env_name}: unsupported binding unexpectedly constructed")

    started = time.perf_counter()
    vec = _C.create_vec(args, 0)
    try:
        if vec.total_agents != total_agents:
            raise AssertionError(
                f"total_agents mismatch: {vec.total_agents} != {total_agents}"
            )
        if vec.obs_size < 1 or vec.num_atns < 1:
            raise AssertionError(
                f"invalid ABI dimensions obs={vec.obs_size}, actions={vec.num_atns}"
            )
        if len(vec.act_sizes) != vec.num_atns or any(size < 1 for size in vec.act_sizes):
            raise AssertionError(
                f"invalid action heads: num_atns={vec.num_atns}, sizes={vec.act_sizes}"
            )

        dtype_by_symbol = {"FloatTensor": np.float32, "ByteTensor": np.uint8}
        try:
            obs_dtype = np.dtype(dtype_by_symbol[vec.obs_dtype])
        except KeyError as exc:
            raise AssertionError(f"unknown observation dtype {vec.obs_dtype!r}") from exc
        if obs_dtype.itemsize != vec.obs_elem_size:
            raise AssertionError(
                f"observation element size mismatch: {obs_dtype.itemsize} "
                f"!= {vec.obs_elem_size}"
            )

        observations = _view(
            vec.obs_ptr, total_agents * vec.obs_size, obs_dtype
        ).reshape(total_agents, vec.obs_size)
        rewards = _view(vec.rewards_ptr, total_agents, np.float32)
        terminals = _view(vec.terminals_ptr, total_agents, np.float32)

        vec.reset()
        _assert_finite("reset observations", observations)
        _assert_finite("reset rewards", rewards)
        _assert_finite("reset terminals", terminals)
        if np.any(rewards != 0) or np.any(terminals != 0):
            raise AssertionError("reset did not clear rewards and terminals")

        actions = np.zeros(total_agents * vec.num_atns, dtype=np.float32)
        vec.cpu_step(actions.ctypes.data)
        _assert_finite("step observations", observations)
        _assert_finite("step rewards", rewards)
        _assert_finite("step terminals", terminals)
        if not np.all((terminals == 0) | (terminals == 1)):
            raise AssertionError("terminals contain values outside {0, 1}")

        logs = vec.log()
        for key, value in logs.items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise AssertionError(f"log {key!r} is non-finite: {value!r}")

        return {
            "env": env_name,
            "status": "passed",
            "agents": total_agents,
            "obs": [total_agents, vec.obs_size],
            "obs_dtype": vec.obs_dtype,
            "actions": [total_agents, vec.num_atns],
            "act_sizes": list(vec.act_sizes),
            "elapsed_s": round(time.perf_counter() - started, 4),
        }
    finally:
        vec.close()


def _binding_names() -> list[str]:
    return sorted(path.parent.name for path in (ROOT / "ocean").glob("*/binding.c"))


def _tail(text: str, lines: int = 30) -> str:
    return "\n".join(text.splitlines()[-lines:])


def _parse_child_result(stdout: str) -> dict:
    # Native stdio and Python stdio have separate buffers, so a late C flush can
    # legally appear after the JSON line.  Find the last actual result object.
    for line in reversed(stdout.splitlines()):
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict) and "env" in result and "status" in result:
            return result
    raise ValueError("child did not emit a JSON result object")


def build_all(
    build_mode: str,
    build_timeout: float,
    smoke_timeout: float,
    json_output: Path | None = None,
) -> int:
    # Do not resolve the interpreter symlink: uv-managed virtualenvs point at a
    # shared runtime, while dependencies such as pybind11 live in .venv/bin's
    # environment and build.sh invokes them through the `python` command.
    python_dir = str(Path(sys.executable).absolute().parent)
    child_env = os.environ.copy()
    child_env["PATH"] = python_dir + os.pathsep + child_env.get("PATH", "")

    results = []
    for env_name in _binding_names():
        print(f"[{env_name}] building", flush=True)
        build_cmd = [str(ROOT / "build.sh"), env_name]
        if build_mode != "default":
            build_cmd.append(f"--{build_mode}")
        try:
            built = subprocess.run(
                build_cmd,
                cwd=ROOT,
                env=child_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=build_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            results.append({
                "env": env_name,
                "status": "build_timeout",
                "detail": _tail(exc.stdout or ""),
            })
            print(f"[{env_name}] BUILD TIMEOUT", flush=True)
            continue
        if built.returncode != 0:
            results.append({
                "env": env_name,
                "status": "build_failed",
                "detail": _tail(built.stdout),
            })
            print(f"[{env_name}] BUILD FAILED\n{_tail(built.stdout)}", flush=True)
            continue

        try:
            smoked = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), env_name],
                cwd=ROOT,
                env=child_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=smoke_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            results.append({
                "env": env_name,
                "status": "smoke_timeout",
                "detail": _tail((exc.stderr or "") + "\n" + (exc.stdout or "")),
            })
            print(f"[{env_name}] SMOKE TIMEOUT", flush=True)
            continue

        if smoked.returncode != 0:
            results.append({
                "env": env_name,
                "status": "smoke_failed",
                "returncode": smoked.returncode,
                "detail": _tail(smoked.stderr + "\n" + smoked.stdout),
            })
            print(
                f"[{env_name}] SMOKE FAILED ({smoked.returncode})\n"
                f"{_tail(smoked.stderr + chr(10) + smoked.stdout)}",
                flush=True,
            )
            continue

        try:
            result = _parse_child_result(smoked.stdout)
        except ValueError:
            result = {
                "env": env_name,
                "status": "smoke_protocol_failed",
                "detail": _tail(smoked.stderr + "\n" + smoked.stdout),
            }
        results.append(result)
        print(f"[{env_name}] {result['status']}", flush=True)

    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    summary = {"counts": counts, "results": results}
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(rendered + "\n")
    print(rendered)
    failures = [
        result for result in results
        if result["status"] not in {
            "passed", "expected_unsupported", "skipped_optional_fixture"
        }
    ]
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env", nargs="?", help="already-built native binding")
    parser.add_argument("--build-all", action="store_true")
    parser.add_argument(
        "--build-mode", choices=("default", "cpu", "mps"), default="mps"
    )
    parser.add_argument("--build-timeout", type=float, default=300.0)
    parser.add_argument("--smoke-timeout", type=float, default=60.0)
    parser.add_argument("--json-output", type=Path)
    ns = parser.parse_args()

    if ns.build_all:
        if ns.env is not None:
            parser.error("env cannot be combined with --build-all")
        return build_all(
            ns.build_mode, ns.build_timeout, ns.smoke_timeout, ns.json_output
        )
    if ns.env is None:
        parser.error("provide an environment or use --build-all")

    try:
        result = smoke_one(ns.env)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
