from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from pufferlib.models import (
    CraftaxEncoder,
    DefaultEncoder,
    ImpulseWarsEncoder,
    load_compatible_state_dict,
)
from tests.test_craftax_encoder import _random_packed
from tests.test_impulse_wars_encoder import _packed_observations


pytestmark = pytest.mark.optional


class _EncoderPolicy(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, observations):
        return self.encoder(observations)


@pytest.fixture(scope='module')
def single_rank_process_group(tmp_path_factory):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip('PyTorch Gloo distributed backend is unavailable')
    if dist.is_initialized():
        if dist.get_backend() != 'gloo':
            pytest.skip('an incompatible process group is already initialized')
        yield
        return

    store = tmp_path_factory.mktemp('checkpoint-ddp') / 'store'
    dist.init_process_group(
        backend='gloo',
        init_method=f'file://{store}',
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=30),
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


def test_ddp_compact_craftax_checkpoint_migrates_strictly(
        single_rank_process_group):
    observations = _random_packed(5, seed=47)
    legacy = DistributedDataParallel(
        _EncoderPolicy(DefaultEncoder(CraftaxEncoder.OBS_SIZE, 13)))
    current = DistributedDataParallel(
        _EncoderPolicy(CraftaxEncoder(CraftaxEncoder.OBS_SIZE, 13)))
    checkpoint = legacy.state_dict()
    assert 'module.encoder.encoder.weight' in checkpoint
    assert 'module.encoder.encoder.weight' not in current.module.state_dict()

    with pytest.warns(UserWarning, match='843-to-8268 exact linear migration'):
        load_compatible_state_dict(current, checkpoint)
    torch.testing.assert_close(
        current.module(observations),
        legacy.module(observations),
        atol=5e-5,
        rtol=2e-5,
    )


def test_ddp_new_impulse_checkpoint_loads_strictly(
        single_rank_process_group):
    observations, _ = _packed_observations(batch_size=3)
    source = DistributedDataParallel(_EncoderPolicy(
        ImpulseWarsEncoder(1_000, hidden_size=32, cnn_channels=8))).eval()
    target = DistributedDataParallel(_EncoderPolicy(
        ImpulseWarsEncoder(1_000, hidden_size=32, cnn_channels=8))).eval()

    load_compatible_state_dict(target, source.state_dict(), warn=False)
    torch.testing.assert_close(
        target.module(observations), source.module(observations))


def test_ddp_legacy_impulse_fallback_must_be_selected_before_wrapping(
        single_rank_process_group):
    legacy = _EncoderPolicy(DefaultEncoder(1_000, hidden_size=32))
    current = DistributedDataParallel(_EncoderPolicy(
        ImpulseWarsEncoder(1_000, hidden_size=32, cnn_channels=8)))

    with pytest.raises(
            RuntimeError, match='loaded before DDP construction'):
        load_compatible_state_dict(
            current, legacy.state_dict(), checkpoint_path='legacy.bin')
