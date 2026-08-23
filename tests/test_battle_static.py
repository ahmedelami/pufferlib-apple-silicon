"""Focused checks for Battle's policy-agent/static-vector contract.

Build with ``./build.sh battle --cpu`` before running this module. The project
links one Ocean environment into ``pufferlib._C`` at a time, so other builds
skip these runtime checks.
"""

import ctypes

import numpy as np
import pytest


def _battle_extension():
    try:
        from pufferlib import _C
    except ImportError as exc:
        pytest.skip(f"native extension is not built: {exc}")
    if _C.env_name != "battle":
        pytest.skip("native extension is not built for Battle")
    return _C


def _args(total_agents=64, num_buffers=1, agents_per_env=64):
    return {
        "vec": {
            "total_agents": total_agents,
            "num_buffers": num_buffers,
            "num_threads": 1,
        },
        "env": {
            "width": 1920,
            "height": 1080,
            "size_x": 2.0,
            "size_y": 1.0,
            "size_z": 2.0,
            "num_agents": agents_per_env,
            "num_armies": 2,
        },
    }


def test_every_advertised_agent_has_state_and_accepts_actions():
    cmod = _battle_extension()
    # Two 64-policy-agent environments intentionally exercise the old public
    # 128-slot shape: previously only 64 of those rows were meaningful.
    vec = cmod.create_vec(_args(total_agents=128), 0)
    try:
        assert vec.total_agents == 128
        assert vec.obs_size == 100
        assert list(vec.act_sizes) == [1, 1, 1]
        vec.reset()

        storage = (ctypes.c_float * (vec.total_agents * vec.obs_size)).from_address(
            vec.obs_ptr
        )
        observations = np.ctypeslib.as_array(storage).reshape(
            vec.total_agents, vec.obs_size
        )
        assert np.isfinite(observations).all()
        # The old binding populated only the first half of these advertised
        # rows. All policy slots must now contain their own 100-float state.
        assert np.count_nonzero(observations, axis=1).min() > 0

        # Individual velocity/orientation/position lives at [70:80]. A policy
        # step must update this state for the externally controlled army.
        before = observations[:, 70:80].copy()
        actions = np.ones((vec.total_agents, vec.num_atns), dtype=np.float32)
        vec.cpu_step(actions.ctypes.data)
        after = observations[:, 70:80]
        assert np.isfinite(after).all()
        assert np.all(np.any(np.abs(after - before) > 1e-7, axis=1))
    finally:
        vec.close()


def test_rejects_partial_policy_armies():
    cmod = _battle_extension()
    with pytest.raises(RuntimeError, match="native vector initialization failed"):
        cmod.create_vec(_args(total_agents=96), 0)
