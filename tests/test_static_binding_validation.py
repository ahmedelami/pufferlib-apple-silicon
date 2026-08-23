"""Fail-safe configuration checks for static Ocean bindings and Dict growth.

Only one Ocean environment is linked into :mod:`pufferlib._C` at a time, so
binding-specific tests skip unless their matching extension is active.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
INITIALIZATION_ERROR = "native vector initialization failed"


def _extension_for(env_name: str):
    try:
        from pufferlib import _C
    except ImportError as exc:
        pytest.skip(f"native extension is not built: {exc}")
    if _C.env_name != env_name:
        pytest.skip(f"native extension is not built for {env_name}")
    return _C


def _args(total_agents: int, num_buffers: int, **env_args):
    return {
        "vec": {
            "total_agents": total_agents,
            "num_buffers": num_buffers,
            "num_threads": 1,
        },
        "env": env_args,
    }


def _assert_rejected(cmod, args) -> None:
    with pytest.raises(RuntimeError, match=INITIALIZATION_ERROR):
        cmod.create_vec(args, 0)


def test_boids_rejects_static_shape_and_partial_flocks():
    cmod = _extension_for("boids")
    _assert_rejected(cmod, _args(64, 1, num_boids=63))
    _assert_rejected(cmod, _args(96, 1, num_boids=64))
    # The global count contains three complete flocks, but each buffer would
    # contain 96 agents and therefore split a flock across a buffer boundary.
    _assert_rejected(cmod, _args(192, 2, num_boids=64))


def test_checkers_rejects_static_shape_and_invalid_buffer_counts():
    cmod = _extension_for("checkers")
    _assert_rejected(cmod, _args(1, 1, size=7))
    _assert_rejected(cmod, _args(1, 1))
    _assert_rejected(cmod, _args(1, 0, size=8))
    _assert_rejected(cmod, _args(3, 2, size=8))


def test_convert_circle_rejects_static_shape_and_partial_environments():
    cmod = _extension_for("convert_circle")
    _assert_rejected(cmod, _args(8, 1, num_agents=8, num_resources=7))
    _assert_rejected(cmod, _args(10, 1, num_agents=4, num_resources=8))
    # Three complete environments globally, but 1.5 environments per buffer.
    _assert_rejected(cmod, _args(24, 2, num_agents=8, num_resources=8))


def test_shared_pool_rejects_static_shape_and_partial_environments():
    cmod = _extension_for("shared_pool")
    _assert_rejected(cmod, _args(8, 1, num_agents=8, vision=2))
    _assert_rejected(cmod, _args(10, 1, num_agents=4, vision=3))
    # Three complete environments globally, but 1.5 environments per buffer.
    _assert_rejected(cmod, _args(24, 2, num_agents=8, vision=3))


VALID_CONFIGS = {
    "boids": _args(64, 1, num_boids=64),
    "checkers": _args(1, 1, size=8),
    "convert_circle": _args(
        4,
        2,
        width=1920,
        height=1080,
        num_agents=2,
        num_factories=8,
        num_resources=8,
        equidistant=0,
        radius=30,
    ),
    "shared_pool": _args(
        4,
        2,
        width=32,
        height=32,
        num_agents=2,
        vision=3,
        reward_food=1.0,
        interactive_food_reward=5.0,
        reward_move=-0.01,
        food_base_spawn_rate=0.002,
    ),
}


def test_active_target_binding_still_accepts_a_valid_configuration():
    try:
        from pufferlib import _C
    except ImportError as exc:
        pytest.skip(f"native extension is not built: {exc}")
    args = VALID_CONFIGS.get(_C.env_name)
    if args is None:
        pytest.skip("active extension is not one of the targeted static bindings")

    vec = _C.create_vec(args, 0)
    vec.close()


def test_dict_grows_beyond_legacy_log_capacity(tmp_path):
    """Exercise the header's native Dict implementation without an Ocean build."""
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is required for the native Dict regression")

    source = tmp_path / "dict_growth.c"
    executable = tmp_path / "dict_growth"
    source.write_text(
        r'''
#include <stdio.h>
#include "vecenv.h"

int main(void) {
    Dict* dict = create_dict(1);
    char keys[40][16];
    for (int i = 0; i < 40; i++) {
        snprintf(keys[i], sizeof(keys[i]), "key_%d", i);
        dict_set(dict, keys[i], (double)i);
    }
    dict_set(dict, keys[7], 99.0);
    DictItem* updated = dict_get_unsafe(dict, keys[7]);
    int ok = dict->size == 40 && dict->capacity >= 40
        && updated != NULL && updated->value == 99.0;
    free_dict(dict);
    return ok ? 0 : 1;
}
''',
        encoding="utf-8",
    )
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT / "src"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)
