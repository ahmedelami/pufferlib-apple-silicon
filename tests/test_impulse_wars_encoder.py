import pytest
import torch
import torch.nn as nn

from pufferlib.models import (
    DefaultEncoder,
    DefaultDecoder,
    ImpulseWarsEncoder,
    MinGRU,
    Policy,
    load_compatible_state_dict,
)


class _NoMapFeatures(nn.Module):
    def forward(self, observations):
        return observations.new_empty((observations.shape[0], 0))


class _EncoderPolicy(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, observations):
        return self.encoder(observations)


def _packed_observations(batch_size=2):
    observations = torch.zeros(batch_size, 1_000, dtype=torch.uint8)

    # Four near-wall categories, four floating-wall categories, and thirty
    # projectile-owner categories occupy bytes 121:159.
    observations[:, 121:125] = torch.tensor([0, 1, 2, 0])
    observations[:, 125:129] = torch.tensor([0, 1, 2, 3])
    observations[:, 129:159] = (
        torch.arange(30, dtype=torch.uint8) % 3)

    # Thirty projectile weapons, three pickup weapons, and the two drone
    # weapons occupy bytes 159:194. Zero means empty and 1..10 are weapons.
    observations[:, 159:194] = (
        torch.arange(35, dtype=torch.uint8) % 11)

    continuous = torch.linspace(-3.0, 3.0, 201, dtype=torch.float32)
    observations[:, 196:] = continuous.view(torch.uint8)

    # These are alignment bytes, not either categorical or float features.
    observations[:, 194:196] = 0xFF
    return observations, continuous


def test_impulse_wars_encoder_rejects_non_static_abi():
    with pytest.raises(ValueError, match='requires 1000 observation bytes'):
        ImpulseWarsEncoder(996)


def test_impulse_wars_legacy_checkpoint_uses_explicit_exact_fallback():
    """Raw-byte Linear weights cannot be mapped into the structured encoder."""
    observations, _ = _packed_observations(batch_size=3)
    legacy = _EncoderPolicy(DefaultEncoder(1_000, hidden_size=32))
    current = _EncoderPolicy(
        ImpulseWarsEncoder(1_000, hidden_size=32, cnn_channels=8))

    with pytest.warns(UserWarning, match='raw-byte DefaultEncoder was retained'):
        load_compatible_state_dict(
            current, legacy.state_dict(), checkpoint_path='legacy.bin')

    assert isinstance(current.encoder, DefaultEncoder)
    torch.testing.assert_close(current(observations), legacy(observations))


def test_impulse_wars_unknown_checkpoint_shape_fails_actionably():
    current = _EncoderPolicy(
        ImpulseWarsEncoder(1_000, hidden_size=32, cnn_channels=8))
    incompatible = _EncoderPolicy(DefaultEncoder(999, hidden_size=32))

    with pytest.raises(
            RuntimeError, match='strict loading.*Match torch.encoder'):
        load_compatible_state_dict(
            current, incompatible.state_dict(), checkpoint_path='unknown.bin')


def test_impulse_wars_new_checkpoint_loads_strictly():
    observations, _ = _packed_observations(batch_size=3)
    source = _EncoderPolicy(
        ImpulseWarsEncoder(1_000, hidden_size=32, cnn_channels=8)).eval()
    target = _EncoderPolicy(
        ImpulseWarsEncoder(1_000, hidden_size=32, cnn_channels=8)).eval()

    load_compatible_state_dict(target, source.state_dict(), warn=False)
    torch.testing.assert_close(target(observations), source(observations))


def test_impulse_wars_native_binding_rejects_invalid_config_without_abort():
    try:
        from pufferlib import _C
    except ImportError as exc:
        pytest.skip(f'native extension is not built: {exc}')
    if getattr(_C, 'env_name', None) != 'impulse_wars':
        pytest.skip('native extension is not built for Impulse Wars')

    args = {
        'vec': {'total_agents': 1, 'num_buffers': 1, 'num_threads': 1},
        'env': {'num_drones': 3, 'num_agents': 1},
    }
    with pytest.raises(RuntimeError, match='initialization failed'):
        _C.create_vec(args, 0)


def test_impulse_wars_encoder_exact_categorical_and_float_offsets():
    observations, continuous = _packed_observations()
    encoder = ImpulseWarsEncoder(
        1_000, hidden_size=16, weapon_type_embedding_dims=1)

    # Expose the pre-projection features so this test verifies the ABI rather
    # than depending on randomly initialized CNN/linear weights.
    encoder.map_cnn = _NoMapFeatures()
    encoder.encoder = nn.Identity()
    with torch.no_grad():
        encoder.weapon_type_embedding.weight.copy_(
            torch.arange(11, dtype=torch.float32).view(11, 1))

    features = encoder(observations)
    multihot = features[:, :encoder.multihot_size]
    weapons = features[:, encoder.multihot_size:encoder.multihot_size + 35]
    unpacked_continuous = features[:, -encoder.CONTINUOUS_SIZE:]

    factors = [3] * 4 + [4] * 4 + [3] * 30
    categories = observations[0, 121:159].tolist()
    expected = torch.zeros(encoder.multihot_size)
    offset = 0
    for factor, category in zip(factors, categories):
        expected[offset + category] = 1
        offset += factor

    torch.testing.assert_close(multihot, expected.expand_as(multihot))
    torch.testing.assert_close(
        weapons, observations[:, 159:194].float())
    torch.testing.assert_close(
        unpacked_continuous, continuous.expand_as(unpacked_continuous))


def test_impulse_wars_encoder_map_bit_layout_and_training_shape():
    encoder = ImpulseWarsEncoder(1_000, hidden_size=32, cnn_channels=8)
    observations = torch.zeros(2, 3, 1_000, dtype=torch.uint8)

    # wall=3, floating=1, pickup=1, other drone=1
    observations[..., 0] = (3 << 5) | (1 << 4) | (1 << 3) | 1
    unpacked = encoder._unpack_map(observations.reshape(-1, 1_000))
    assert unpacked.shape == (6, 8, 11, 11)
    assert unpacked[:, 3, 0, 0].eq(1).all()  # wall-type channel 3
    assert unpacked[:, 4, 0, 0].eq(1).all()  # floating flag
    assert unpacked[:, 5, 0, 0].eq(1).all()  # pickup flag
    assert unpacked[:, 7, 0, 0].eq(1).all()  # drone-index channel 1

    output = encoder(observations)
    assert output.shape == (6, 32)
    output.sum().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in encoder.parameters())


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_impulse_wars_recurrent_policy_training_shape(device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS is unavailable')

    observations, _ = _packed_observations(batch_size=6)
    policy = Policy(
        ImpulseWarsEncoder(1_000, hidden_size=32, cnn_channels=8),
        DefaultDecoder((9, 17, 2, 2, 2), hidden_size=32),
        MinGRU(hidden_size=32, num_layers=2),
    ).to(device)

    logits, values = policy(observations.reshape(2, 3, 1_000).to(device))
    assert [value.shape for value in logits] == [
        (6, 9), (6, 17), (6, 2), (6, 2), (6, 2)]
    assert values.shape == (2, 3)

    loss = values.square().mean() + sum(value.square().mean() for value in logits)
    loss.backward()
    if device == 'mps':
        torch.mps.synchronize()
    assert torch.isfinite(loss).item()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in policy.parameters())


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='MPS is unavailable')
def test_impulse_wars_encoder_cpu_mps_parity_and_float_bitcast():
    torch.manual_seed(7)
    observations, continuous = _packed_observations(batch_size=4)
    cpu_encoder = ImpulseWarsEncoder(
        1_000, hidden_size=32, cnn_channels=8).eval()
    mps_encoder = ImpulseWarsEncoder(
        1_000, hidden_size=32, cnn_channels=8).eval().to('mps')
    mps_encoder.load_state_dict(cpu_encoder.state_dict())

    with torch.no_grad():
        expected = cpu_encoder(observations)
        actual = mps_encoder(observations.to('mps'))
    torch.mps.synchronize()

    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
    mps_continuous = observations.to('mps').view(torch.float32).reshape(
        4, 250)[:, 49:]
    torch.testing.assert_close(
        mps_continuous.cpu(), continuous.expand_as(mps_continuous))


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='MPS is unavailable')
@pytest.mark.parametrize('autocast', [False, True])
def test_impulse_wars_encoder_mps_backward(autocast):
    observations, _ = _packed_observations(batch_size=4)
    encoder = ImpulseWarsEncoder(
        1_000, hidden_size=32, cnn_channels=8).to('mps').train()

    with torch.autocast(
            'mps', dtype=torch.bfloat16, enabled=autocast):
        output = encoder(observations.to('mps'))
        loss = output.float().square().mean()
    loss.backward()
    torch.mps.synchronize()

    assert output.dtype == (torch.bfloat16 if autocast else torch.float32)
    assert torch.isfinite(loss).item()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in encoder.parameters())
