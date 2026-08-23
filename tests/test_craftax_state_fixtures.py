import ctypes
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.optional
pytest.importorskip("jax", reason="Craftax state fixtures require optional jax")
pytest.importorskip("craftax", reason="Craftax state fixtures require optional craftax")

import jax
from craftax.craftax_env import make_craftax_env_from_name

from tests.craftax_state_fixtures import (
    CraftaxState,
    assert_env_states_equal,
    craftax_state_to_jax,
    jax_state_to_c_state,
)


def test_ctypes_state_matches_native_c_layout():
    root = Path(__file__).resolve().parents[1]
    raylib_archive = (
        "raylib-5.5_macos"
        if platform.system() == "Darwin"
        else "raylib-5.5_linux_amd64"
    )
    raylib_include = root / raylib_archive / "include"
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is required for the native layout check")
    if not (raylib_include / "raylib.h").is_file():
        pytest.skip(f"missing Raylib headers at {raylib_include}; run build.sh first")

    field_names = [name for name, _ctype in CraftaxState._fields_]
    offset_entries = ",\n".join(
        f"    offsetof(CraftaxState, {name})" for name in field_names
    )
    source = f"""
    #include <stddef.h>
    #include "ocean/craftax/craftax.h"

    size_t craftax_state_size(void) {{
        return sizeof(CraftaxState);
    }}

    size_t craftax_state_offset(size_t index) {{
        static const size_t offsets[] = {{
    {offset_entries}
        }};
        return offsets[index];
    }}
    """

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source_path = tmp_path / "craftax_state_layout.c"
        library_path = tmp_path / "craftax_state_layout.so"
        source_path.write_text(source)
        subprocess.run(
            [
                compiler,
                "-std=c99",
                "-O0",
                "-shared",
                "-fPIC",
                "-I",
                str(root),
                "-I",
                str(raylib_include),
                str(source_path),
                "-lm",
                "-o",
                str(library_path),
            ],
            check=True,
            cwd=root,
        )
        native = ctypes.CDLL(str(library_path))
        native.craftax_state_size.argtypes = []
        native.craftax_state_size.restype = ctypes.c_size_t
        native.craftax_state_offset.argtypes = [ctypes.c_size_t]
        native.craftax_state_offset.restype = ctypes.c_size_t

        assert native.craftax_state_size() == ctypes.sizeof(CraftaxState) == 79776
        for index, name in enumerate(field_names):
            assert native.craftax_state_offset(index) == getattr(
                CraftaxState, name
            ).offset


def test_jax_ctypes_roundtrip_uses_native_light_precision():
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    _observation, jax_state = env.reset(jax.random.PRNGKey(7), env.default_params)

    c_state = jax_state_to_c_state(jax_state)
    roundtrip = craftax_state_to_jax(c_state, template=jax_state)

    assert_env_states_equal(roundtrip, jax_state, "JAX -> ctypes -> JAX")


def test_native_packer_preserves_overlap_and_late_inactive_clear():
    root = Path(__file__).resolve().parents[1]
    raylib_archive = (
        "raylib-5.5_macos"
        if platform.system() == "Darwin"
        else "raylib-5.5_linux_amd64"
    )
    raylib_include = root / raylib_archive / "include"
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is required for the native packer check")
    if not (raylib_include / "raylib.h").is_file():
        pytest.skip(f"missing Raylib headers at {raylib_include}; run build.sh first")

    source = """
    #include "ocean/craftax/worldgen.h"

    void craftax_test_encode(const CraftaxWorldState* state, float* observation) {
        craftax_encode_reset_observation(state, observation);
    }
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source_path = tmp_path / "craftax_observation_fixture.c"
        library_path = tmp_path / "craftax_observation_fixture.so"
        source_path.write_text(source)
        subprocess.run(
            [
                compiler,
                "-std=c99",
                "-O2",
                "-shared",
                "-fPIC",
                "-I",
                str(root),
                "-I",
                str(raylib_include),
                str(source_path),
                "-lm",
                "-o",
                str(library_path),
            ],
            check=True,
            cwd=root,
        )
        native = ctypes.CDLL(str(library_path))
        native.craftax_test_encode.argtypes = [
            ctypes.POINTER(CraftaxState),
            ctypes.POINTER(ctypes.c_float),
        ]
        native.craftax_test_encode.restype = None

        state = CraftaxState()
        state.player_level = 0
        state.player_position[:] = (24, 24)
        state.light_map[0][24][24] = 255
        projectiles = state.mob_projectiles
        for slot, type_id in enumerate((1, 6, 1)):
            projectiles.position[0][slot][:] = (24, 24)
            projectiles.type_id[0][slot] = type_id
        projectiles.mask[0][0] = True
        projectiles.mask[0][1] = True
        projectiles.mask[0][2] = False

        observation = np.zeros(843, dtype=np.float32)
        native.craftax_test_encode(
            ctypes.byref(state),
            observation.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

    center_cell = (9 // 2) * 11 + (11 // 2)
    mob_projectile_mask = center_cell * 8 + 3 + 3
    # Slots 0/1 set types 1 and 6; later inactive slot 2 clears only type 1.
    assert int(observation[mob_projectile_mask]) == 1 << 6


def test_processwide_reset_pool_rejects_conflicting_vector_configuration():
    try:
        from pufferlib import _C
    except ImportError as exc:
        pytest.skip(f"native extension is not built: {exc}")
    if getattr(_C, "env_name", None) != "craftax":
        pytest.skip("native extension is not built for Craftax")

    args = {
        "vec": {"total_agents": 1, "num_buffers": 1, "num_threads": 1},
        "env": {"seed_offset": 0, "reset_pool_size": 0},
    }
    vec = _C.create_vec(args, 0)
    try:
        args["env"]["reset_pool_size"] = 1
        with pytest.raises(RuntimeError, match="initialization failed"):
            _C.create_vec(args, 0)
    finally:
        vec.close()
