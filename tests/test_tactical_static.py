"""Tactical must not silently advertise its unfinished RL prototype as usable."""

import pytest


def test_incomplete_tactical_vector_api_fails_closed():
    try:
        from pufferlib import _C
    except ImportError as exc:
        pytest.skip(f"native extension is not built: {exc}")
    if _C.env_name != "tactical":
        pytest.skip("native extension is not built for Tactical")

    args = {
        "vec": {"total_agents": 1, "num_buffers": 1, "num_threads": 1},
        "env": {},
    }
    with pytest.raises(RuntimeError, match="native vector initialization failed"):
        _C.create_vec(args, 0)
