import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from pufferlib.models import (
    CraftaxEncoder,
    DefaultEncoder,
    load_compatible_state_dict,
)
from pufferlib.muon import Muon


pytestmark = pytest.mark.optional


class _EncoderPolicy(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, observations):
        return self.encoder(observations)


def _random_packed(batch_size, seed=0):
    rng = np.random.default_rng(seed)
    cells = np.zeros(
        (batch_size, CraftaxEncoder.NUM_CELLS, CraftaxEncoder.PACKED_CHANNELS),
        dtype=np.float32,
    )
    visible = rng.integers(
        0, 2, size=(batch_size, CraftaxEncoder.NUM_CELLS), dtype=np.int32
    ).astype(bool)
    cells[:, :, 0] = rng.integers(
        0, CraftaxEncoder.NUM_BLOCK_TYPES, size=visible.shape
    ) * visible
    cells[:, :, 1] = rng.integers(
        1, CraftaxEncoder.NUM_ITEM_TYPES + 1, size=visible.shape
    ) * visible
    cells[:, :, 2] = visible
    cells[:, :, 3:] = rng.integers(
        0,
        256,
        size=(batch_size, CraftaxEncoder.NUM_CELLS, CraftaxEncoder.NUM_MOB_CLASSES),
    ) * visible[:, :, None]
    tail = rng.normal(size=(batch_size, CraftaxEncoder.TAIL_SIZE)).astype(np.float32)
    return torch.from_numpy(np.concatenate((cells.reshape(batch_size, -1), tail), axis=1))


def _materialized_linear_weight(encoder):
    map_weight = torch.cat(
        (
            encoder.block_weight,
            encoder.item_weight,
            encoder.mob_weight.reshape(
                encoder.NUM_CELLS,
                encoder.NUM_MOB_CLASSES * encoder.NUM_MOB_TYPES,
                -1,
            ),
            encoder.visibility_weight.unsqueeze(1),
        ),
        dim=1,
    ).reshape(-1, encoder.bias.numel())
    return torch.cat((map_weight, encoder.tail_weight.t()), dim=0).t()


def test_craftax_projection_matches_full_one_hot_linear():
    observations = _random_packed(7, seed=3)
    encoder = CraftaxEncoder(CraftaxEncoder.OBS_SIZE, hidden_size=13)
    reference_encoder = deepcopy(encoder)

    expanded = reference_encoder.expand_observation(observations)
    expected = F.linear(
        expanded,
        _materialized_linear_weight(reference_encoder),
        reference_encoder.bias,
    )
    actual = encoder(observations)

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    target = torch.linspace(-0.5, 0.5, actual.numel()).reshape_as(actual)
    (actual * target).sum().backward()
    (expected * target).sum().backward()
    for (name, parameter), (reference_name, reference_parameter) in zip(
        encoder.named_parameters(), reference_encoder.named_parameters()
    ):
        assert name == reference_name
        torch.testing.assert_close(
            parameter.grad,
            reference_parameter.grad,
            atol=2e-6,
            rtol=2e-6,
            msg=lambda message: f"gradient mismatch for {name}: {message}",
        )


def test_craftax_muon_step_matches_full_one_hot_linear():
    """The compact projection must remain one logical Muon matrix."""
    torch.manual_seed(17)
    observations = _random_packed(11, seed=17)
    encoder = CraftaxEncoder(CraftaxEncoder.OBS_SIZE, hidden_size=13)
    assert set(encoder.state_dict()) == {"encoder.weight", "encoder.bias"}
    reference = nn.Linear(CraftaxEncoder.FULL_OBS_SIZE, 13)
    with torch.no_grad():
        reference.weight.copy_(encoder.weight)
        reference.bias.copy_(encoder.bias)

    target = torch.randn(11, 13)
    F.mse_loss(encoder(observations), target).backward()
    F.mse_loss(
        reference(CraftaxEncoder.expand_observation(observations)), target
    ).backward()
    torch.testing.assert_close(
        encoder.weight.grad, reference.weight.grad, atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(
        encoder.bias.grad, reference.bias.grad, atol=2e-6, rtol=2e-6
    )

    optimizer = Muon(encoder.parameters(), lr=0.0025, momentum=0.9)
    reference_optimizer = Muon(reference.parameters(), lr=0.0025, momentum=0.9)
    optimizer.step()
    reference_optimizer.step()

    torch.testing.assert_close(
        encoder.weight, reference.weight, atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(
        encoder.bias, reference.bias, atol=2e-6, rtol=2e-6
    )


def test_craftax_legacy_full_symbolic_checkpoint_loads_strictly():
    """Pre-port 8,268-input DefaultEncoder checkpoints keep exact weights."""
    observations = _random_packed(5, seed=29)
    expanded = CraftaxEncoder.expand_observation(observations)
    legacy = _EncoderPolicy(DefaultEncoder(CraftaxEncoder.FULL_OBS_SIZE, 13))
    current = _EncoderPolicy(CraftaxEncoder(CraftaxEncoder.OBS_SIZE, 13))

    load_compatible_state_dict(current, legacy.state_dict(), warn=False)
    assert isinstance(current.encoder, CraftaxEncoder)
    torch.testing.assert_close(
        current(observations), legacy(expanded), atol=2e-6, rtol=2e-6)


def test_craftax_compact_linear_checkpoint_is_migrated_exactly():
    """The transient 843-input ordinal/mask Linear has an exact lift."""
    observations = _random_packed(7, seed=31)
    legacy = _EncoderPolicy(DefaultEncoder(CraftaxEncoder.OBS_SIZE, 13))
    current = _EncoderPolicy(CraftaxEncoder(CraftaxEncoder.OBS_SIZE, 13))

    with pytest.warns(UserWarning, match='843-to-8268 exact linear migration'):
        load_compatible_state_dict(current, legacy.state_dict())
    torch.testing.assert_close(
        current(observations), legacy(observations), atol=5e-5, rtol=2e-5)


def test_craftax_intermediate_checkpoint_keys_are_migrated_strictly():
    observations = _random_packed(3, seed=37)
    source = _EncoderPolicy(CraftaxEncoder(CraftaxEncoder.OBS_SIZE, 11))
    state_dict = source.state_dict()
    state_dict['encoder.weight'] = state_dict.pop('encoder.encoder.weight')
    state_dict['encoder.bias'] = state_dict.pop('encoder.encoder.bias')
    target = _EncoderPolicy(CraftaxEncoder(CraftaxEncoder.OBS_SIZE, 11))

    with pytest.warns(UserWarning, match='canonical-key rename'):
        load_compatible_state_dict(target, state_dict)
    torch.testing.assert_close(
        target(observations), source(observations), atol=2e-6, rtol=2e-6)


def test_craftax_split_table_checkpoint_is_migrated_strictly():
    observations = _random_packed(3, seed=41)
    source = CraftaxEncoder(CraftaxEncoder.OBS_SIZE, 11)
    split_state = {
        'encoder.block_weight': source.block_weight.detach().clone(),
        'encoder.item_weight': source.item_weight.detach().clone(),
        'encoder.mob_weight': source.mob_weight.detach().clone(),
        'encoder.visibility_weight': (
            source.visibility_weight.detach().clone()),
        'encoder.tail_weight': source.tail_weight.detach().clone(),
        'encoder.bias': source.bias.detach().clone(),
    }
    target = _EncoderPolicy(CraftaxEncoder(CraftaxEncoder.OBS_SIZE, 11))

    with pytest.warns(UserWarning, match='split-table exact migration'):
        load_compatible_state_dict(target, split_state)
    torch.testing.assert_close(
        target(observations), source(observations), atol=2e-6, rtol=2e-6)


def test_craftax_expansion_retains_overlapping_mob_types():
    observations = torch.zeros(1, CraftaxEncoder.OBS_SIZE)
    cells = observations[:, :CraftaxEncoder.PACKED_MAP_SIZE].reshape(
        1, CraftaxEncoder.NUM_CELLS, CraftaxEncoder.PACKED_CHANNELS
    )
    cells[0, 0, 2] = 1
    cells[0, 0, 3] = (1 << 1) | (1 << 6)

    expanded = CraftaxEncoder.expand_observation(observations)
    first_cell = expanded[0, :CraftaxEncoder.FULL_CELL_SIZE]
    mob_start = CraftaxEncoder.NUM_BLOCK_TYPES + CraftaxEncoder.NUM_ITEM_TYPES
    assert first_cell[mob_start + 1] == 1
    assert first_cell[mob_start + 6] == 1
    assert first_cell[mob_start:mob_start + CraftaxEncoder.NUM_MOB_TYPES].sum() == 2


def test_craftax_packed_expands_to_jax_symbolic_observation():
    jax = pytest.importorskip("jax")
    pytest.importorskip("craftax")
    from craftax.craftax_env import make_craftax_env_from_name
    from tests.craftax_parity import pack_symbolic_observation

    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    full_observations = []
    for seed in range(4):
        full, _state = env.reset(jax.random.PRNGKey(seed), env.default_params)
        full_observations.append(np.asarray(full, dtype=np.float32))
    full_observations = np.stack(full_observations)
    packed = pack_symbolic_observation(full_observations)

    expanded = CraftaxEncoder.expand_observation(torch.from_numpy(packed)).numpy()
    np.testing.assert_array_equal(expanded, full_observations)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires an available MPS device"
)
def test_craftax_encoder_mps_forward_backward_matches_cpu():
    observations = _random_packed(16, seed=9)
    cpu_encoder = CraftaxEncoder(CraftaxEncoder.OBS_SIZE, hidden_size=17)
    mps_encoder = CraftaxEncoder(CraftaxEncoder.OBS_SIZE, hidden_size=17).to("mps")
    mps_encoder.load_state_dict(cpu_encoder.state_dict())

    cpu_output = cpu_encoder(observations)
    cpu_output.square().mean().backward()

    mps_output = mps_encoder(observations.to("mps"))
    mps_output.square().mean().backward()
    torch.mps.synchronize()

    torch.testing.assert_close(
        mps_output.cpu(), cpu_output, atol=2e-4, rtol=2e-4
    )
    torch.testing.assert_close(
        mps_encoder.weight.grad.cpu(),
        cpu_encoder.weight.grad,
        atol=2e-4,
        rtol=2e-4,
    )
