from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.optional
pytest.importorskip("jax", reason="Craftax parity requires the optional jax package")
pytest.importorskip("craftax", reason="Craftax parity requires the optional craftax package")

try:
    from pufferlib import _C
except Exception as exc:
    pytest.skip(f"native environment extension is not built: {exc}", allow_module_level=True)

if getattr(_C, "env_name", None) != "craftax":
    pytest.skip(
        "Craftax parity requires pufferlib._C to be built for the craftax environment",
        allow_module_level=True,
    )

from tests import craftax_parity


def test_craftax_full_native_step_parity():
    args = SimpleNamespace(
        seeds=16,
        seed_start=0,
        steps=2000,
        action_seed=0,
        atol=1e-5,
    )
    assert craftax_parity.run(args) == 0
