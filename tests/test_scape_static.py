"""Scape must not silently advertise its unfinished RL prototype as usable."""

import pytest


def test_incomplete_scape_vector_api_fails_closed(capfd):
    try:
        from pufferlib import _C
    except ImportError as exc:
        pytest.skip(f"native extension is not built: {exc}")
    if _C.env_name != "scape":
        pytest.skip("native extension is not built for Scape")

    args = {
        "vec": {"total_agents": 8, "num_buffers": 1, "num_threads": 1},
        "env": {"width": 1080, "height": 720},
    }
    with pytest.raises(RuntimeError, match="native vector initialization failed"):
        _C.create_vec(args, 0)
    assert "scape: RL vector API is incomplete upstream" in capfd.readouterr().err
