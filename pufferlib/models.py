from collections import OrderedDict
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, encoder, decoder, network):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.network = network

    def initial_state(self, batch_size, device):
        return self.network.initial_state(batch_size, device)

    def forward_eval(self, x, state):
        h = self.encoder(x)
        h, state = self.network.forward_eval(h, state)
        logits, values = self.decoder(h)
        return logits, values, state

    def forward(self, x):
        B, TT = x.shape[:2]
        h = self.encoder(x.reshape(B*TT, *x.shape[2:]))
        h = self.network.forward_train(h.reshape(B, TT, -1))
        logits, values = self.decoder(h.reshape(B*TT, -1))
        return logits, values.reshape(B, TT)

class DefaultEncoder(nn.Module):
    def __init__(self, obs_size, hidden_size=128):
        super().__init__()
        self.encoder = nn.Linear(obs_size, hidden_size)

    def forward(self, observations):
        return self.encoder(observations.view(observations.shape[0], -1).float())


class CraftaxEncoder(nn.Module):
    """Lossless projection of Craftax's compact symbolic observation.

    The native environment transports 99 cells with eight compact values per
    cell, followed by the original 51 scalar features. This layer is exactly
    equivalent to applying one Linear layer to Craftax's 8,268-value symbolic
    one-hot observation, but it sums position-specific categorical embeddings
    instead of materializing the large one-hot tensor on every policy step.
    """

    OBS_ROWS = 9
    OBS_COLS = 11
    NUM_CELLS = OBS_ROWS * OBS_COLS
    PACKED_CHANNELS = 8
    PACKED_MAP_SIZE = NUM_CELLS * PACKED_CHANNELS
    TAIL_SIZE = 51
    OBS_SIZE = PACKED_MAP_SIZE + TAIL_SIZE
    NUM_BLOCK_TYPES = 37
    NUM_ITEM_TYPES = 5
    NUM_MOB_CLASSES = 5
    NUM_MOB_TYPES = 8
    FULL_CELL_SIZE = (
        NUM_BLOCK_TYPES
        + NUM_ITEM_TYPES
        + NUM_MOB_CLASSES * NUM_MOB_TYPES
        + 1
    )
    FULL_OBS_SIZE = NUM_CELLS * FULL_CELL_SIZE + TAIL_SIZE

    def __init__(self, obs_size, hidden_size=128):
        super().__init__()
        if int(obs_size) != self.OBS_SIZE:
            raise ValueError(
                f'CraftaxEncoder requires {self.OBS_SIZE} observations, '
                f'got {obs_size}')

        # Keep the same single [hidden, 8268] Parameter as DefaultEncoder.
        # Muon treats each Parameter as one logical matrix, so splitting this
        # into categorical tables would change its orthogonalized update even
        # though the forward function remained mathematically identical.
        # Keep both DefaultEncoder's parameter geometry and its checkpoint
        # keys (``encoder.weight``/``encoder.bias``). Compact observations are
        # projected through views of this canonical Linear below.
        self.encoder = nn.Linear(self.FULL_OBS_SIZE, hidden_size)

        self.register_buffer(
            'block_offsets',
            1 + torch.arange(self.NUM_CELLS, dtype=torch.int64)
            * self.NUM_BLOCK_TYPES,
            persistent=False,
        )
        self.register_buffer(
            'item_offsets',
            1 + torch.arange(self.NUM_CELLS, dtype=torch.int64)
            * self.NUM_ITEM_TYPES,
            persistent=False,
        )
        self.register_buffer(
            'mob_offsets',
            1 + torch.arange(self.NUM_CELLS, dtype=torch.int64) * 256,
            persistent=False,
        )
        self.register_buffer(
            'mob_mask_bits',
            (
                torch.arange(256, dtype=torch.int64).unsqueeze(1)
                >> torch.arange(self.NUM_MOB_TYPES, dtype=torch.int64)
            ).bitwise_and(1).float(),
            persistent=False,
        )

    @property
    def weight(self):
        return self.encoder.weight

    @property
    def bias(self):
        return self.encoder.bias

    @property
    def _cell_weight(self):
        return self.weight[:, :self.NUM_CELLS * self.FULL_CELL_SIZE].t().reshape(
            self.NUM_CELLS, self.FULL_CELL_SIZE, -1)

    @property
    def block_weight(self):
        return self._cell_weight[:, :self.NUM_BLOCK_TYPES]

    @property
    def item_weight(self):
        start = self.NUM_BLOCK_TYPES
        return self._cell_weight[:, start:start + self.NUM_ITEM_TYPES]

    @property
    def mob_weight(self):
        start = self.NUM_BLOCK_TYPES + self.NUM_ITEM_TYPES
        stop = start + self.NUM_MOB_CLASSES * self.NUM_MOB_TYPES
        return self._cell_weight[:, start:stop].reshape(
            self.NUM_CELLS, self.NUM_MOB_CLASSES, self.NUM_MOB_TYPES, -1)

    @property
    def visibility_weight(self):
        return self._cell_weight[:, -1]

    @property
    def tail_weight(self):
        return self.weight[:, -self.TAIL_SIZE:]

    def reset_parameters(self):
        self.encoder.reset_parameters()

    @classmethod
    def migrate_compact_linear_weight(cls, weight):
        """Lift a legacy 843-input Linear weight into symbolic feature space.

        The compact transport stores categorical IDs as numbers and each mob
        class as an 8-bit presence mask. A Linear layer over those values is
        exactly representable over the canonical one-hot observation: category
        ``k`` receives ``k * weight`` and mob bit ``k`` receives
        ``2**k * weight``. This conversion therefore preserves the old policy
        on every valid packed observation; it is not an approximate resize.
        """
        if weight.ndim != 2 or weight.shape[1] != cls.OBS_SIZE:
            raise ValueError(
                f'expected a [hidden, {cls.OBS_SIZE}] compact Craftax '
                f'weight, got {tuple(weight.shape)}')

        hidden_size = weight.shape[0]
        compact_cells = weight[:, :cls.PACKED_MAP_SIZE].reshape(
            hidden_size, cls.NUM_CELLS, cls.PACKED_CHANNELS)
        full_cells = weight.new_zeros(
            hidden_size, cls.NUM_CELLS, cls.FULL_CELL_SIZE)

        block_values = torch.arange(
            cls.NUM_BLOCK_TYPES, dtype=weight.dtype, device=weight.device)
        full_cells[:, :, :cls.NUM_BLOCK_TYPES] = (
            compact_cells[:, :, 0].unsqueeze(-1) * block_values)

        item_start = cls.NUM_BLOCK_TYPES
        item_values = torch.arange(
            1, cls.NUM_ITEM_TYPES + 1,
            dtype=weight.dtype, device=weight.device)
        full_cells[:, :, item_start:item_start + cls.NUM_ITEM_TYPES] = (
            compact_cells[:, :, 1].unsqueeze(-1) * item_values)

        mob_start = item_start + cls.NUM_ITEM_TYPES
        mob_values = 2 ** torch.arange(
            cls.NUM_MOB_TYPES, dtype=weight.dtype, device=weight.device)
        for mob_class in range(cls.NUM_MOB_CLASSES):
            start = mob_start + mob_class * cls.NUM_MOB_TYPES
            full_cells[:, :, start:start + cls.NUM_MOB_TYPES] = (
                compact_cells[:, :, 3 + mob_class].unsqueeze(-1)
                * mob_values)

        full_cells[:, :, -1] = compact_cells[:, :, 2]
        return torch.cat(
            (full_cells.flatten(1), weight[:, cls.PACKED_MAP_SIZE:]), dim=1)

    @staticmethod
    def _embedding_sum(indices, weight):
        padding = weight.new_zeros(1, weight.shape[-1])
        table = torch.cat((padding, weight.reshape(-1, weight.shape[-1])), dim=0)
        return F.embedding_bag(
            indices,
            table,
            mode='sum',
            padding_idx=0,
        )

    @classmethod
    def expand_observation(cls, observations):
        """Materialize the canonical 8,268-vector for parity/debug tests."""
        observations = observations.reshape(-1, cls.OBS_SIZE)
        cells = observations[:, :cls.PACKED_MAP_SIZE].reshape(
            -1, cls.NUM_CELLS, cls.PACKED_CHANNELS)
        visible = cells[:, :, 2:3] > 0.5

        block = F.one_hot(
            cells[:, :, 0].long().clamp(0, cls.NUM_BLOCK_TYPES - 1),
            cls.NUM_BLOCK_TYPES,
        ).to(observations.dtype) * visible

        item_value = cells[:, :, 1].long()
        item = F.one_hot(
            (item_value - 1).clamp(0, cls.NUM_ITEM_TYPES - 1),
            cls.NUM_ITEM_TYPES,
        ).to(observations.dtype)
        item = item * (item_value > 0).unsqueeze(-1) * visible

        masks = cells[:, :, 3:].long().clamp(0, 255)
        shifts = torch.arange(
            cls.NUM_MOB_TYPES, dtype=torch.int64, device=observations.device)
        mobs = ((masks.unsqueeze(-1) >> shifts) & 1).to(observations.dtype)
        mobs = mobs.reshape(
            -1, cls.NUM_CELLS, cls.NUM_MOB_CLASSES * cls.NUM_MOB_TYPES
        ) * visible

        full_cells = torch.cat(
            (block, item, mobs, visible.to(observations.dtype)), dim=-1)
        tail = observations[:, cls.PACKED_MAP_SIZE:].to(full_cells.dtype)
        return torch.cat((full_cells.flatten(1), tail), dim=1)

    def forward(self, observations):
        observations = observations.reshape(-1, self.OBS_SIZE)
        cells = observations[:, :self.PACKED_MAP_SIZE].reshape(
            -1, self.NUM_CELLS, self.PACKED_CHANNELS)
        visible = cells[:, :, 2] > 0.5

        block_ids = cells[:, :, 0].long().clamp(0, self.NUM_BLOCK_TYPES - 1)
        block_indices = self.block_offsets + block_ids
        block_indices = torch.where(visible, block_indices, 0)
        hidden = self._embedding_sum(block_indices, self.block_weight)

        item_values = cells[:, :, 1].long().clamp(0, self.NUM_ITEM_TYPES)
        item_indices = self.item_offsets + (item_values - 1).clamp_min(0)
        item_indices = torch.where(visible & (item_values > 0), item_indices, 0)
        hidden = hidden + self._embedding_sum(item_indices, self.item_weight)

        # Convert each 8-bit presence mask to the additive embedding that the
        # corresponding eight one-hot channels would produce. One class at a
        # time keeps index storage bounded for large on-device batches.
        mask_bits = self.mob_mask_bits.to(dtype=self.mob_weight.dtype)
        mob_combinations = torch.einsum(
            'kt,pcth->pckh', mask_bits, self.mob_weight)
        mob_masks = cells[:, :, 3:].long().clamp(0, 255)
        for mob_class in range(self.NUM_MOB_CLASSES):
            values = mob_masks[:, :, mob_class]
            indices = self.mob_offsets + values
            indices = torch.where(visible & (values > 0), indices, 0)
            hidden = hidden + self._embedding_sum(
                indices, mob_combinations[:, mob_class])

        hidden = hidden + visible.to(self.visibility_weight.dtype).matmul(
            self.visibility_weight)
        tail = observations[:, self.PACKED_MAP_SIZE:].to(self.tail_weight.dtype)
        hidden = hidden + F.linear(tail, self.tail_weight, self.bias)
        return hidden


class ImpulseWarsEncoder(nn.Module):
    """Encoder for the fixed two-drone Impulse Wars static-vector ABI.

    The 1,000-byte observation contains a packed 11x11 map, categorical
    entity fields, two bytes of alignment padding, and 201 native float32
    values. Treating the whole buffer as byte magnitudes destroys the float
    features, so this mirrors the original environment-specific encoder while
    using the corrected 196-byte continuous-data offset.
    """

    OBS_SIZE = 1_000
    MAP_SIZE = 121
    DISCRETE_SIZE = 194
    CONTINUOUS_OFFSET = 196
    CONTINUOUS_SIZE = 201
    NUM_DRONES = 2

    def __init__(self, obs_size, hidden_size=512, cnn_channels=64,
            weapon_type_embedding_dims=2):
        super().__init__()
        if int(obs_size) != self.OBS_SIZE:
            raise ValueError(
                f'ImpulseWarsEncoder requires {self.OBS_SIZE} observation '
                f'bytes, got {obs_size}')

        self.register_buffer(
            'unpack_mask',
            torch.tensor([0x60, 0x10, 0x08, 0x07], dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer(
            'unpack_shift',
            torch.tensor([5, 4, 3, 0], dtype=torch.uint8),
            persistent=False,
        )

        factors = [3] * 4 + [4] * 4 + [self.NUM_DRONES + 1] * 30
        offsets = [0]
        for factor in factors[:-1]:
            offsets.append(offsets[-1] + factor)
        self.multihot_size = sum(factors)
        self.register_buffer(
            'discrete_offsets',
            torch.tensor(offsets, dtype=torch.int64).view(1, -1),
            persistent=False,
        )

        # Zero is the empty slot; weapon IDs 1..10 are real weapon types.
        self.weapon_type_embedding = nn.Embedding(
            11, weapon_type_embedding_dims)
        self.map_cnn = nn.Sequential(
            nn.Conv2d(8, cnn_channels, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
        )

        feature_size = (
            cnn_channels
            + self.multihot_size
            + 35 * weapon_type_embedding_dims
            + self.CONTINUOUS_SIZE
        )
        self.encoder = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
        )

    def _unpack_map(self, observations):
        batch_size = observations.shape[0]
        packed = observations[:, :self.MAP_SIZE].unsqueeze(-1)
        unpacked = (packed & self.unpack_mask) >> self.unpack_shift
        unpacked = unpacked.permute(0, 2, 1).reshape(
            batch_size, 4, 11, 11)

        wall_types = F.one_hot(
            unpacked[:, 0].long(), 4).permute(0, 3, 1, 2).float()
        floating = unpacked[:, 1:2].float()
        pickups = unpacked[:, 2:3].float()
        drones = F.one_hot(
            unpacked[:, 3].long(), self.NUM_DRONES
        ).permute(0, 3, 1, 2).float()
        return torch.cat((wall_types, floating, pickups, drones), dim=1)

    def forward(self, observations):
        observations = observations.reshape(-1, self.OBS_SIZE).contiguous()
        batch_size = observations.shape[0]

        map_features = self.map_cnn(self._unpack_map(observations))

        discrete = observations[:, 121:159].long() + self.discrete_offsets
        multihot = torch.zeros(
            batch_size, self.multihot_size,
            dtype=torch.float32, device=observations.device)
        multihot.scatter_(1, discrete, 1.0)

        weapon_ids = observations[:, 159:self.DISCRETE_SIZE].long()
        weapon_features = self.weapon_type_embedding(weapon_ids).flatten(1)

        # Reinterpret the aligned byte buffer in place. Viewing the entire
        # row first preserves its 1,000-byte stride and avoids a slice copy.
        continuous = observations.view(torch.float32).reshape(
            batch_size, self.OBS_SIZE // 4)[:, self.CONTINUOUS_OFFSET // 4:]
        if continuous.shape[1] != self.CONTINUOUS_SIZE:
            raise RuntimeError('invalid Impulse Wars continuous observation')

        return self.encoder(torch.cat(
            (map_features, multihot, weapon_features, continuous), dim=1))


def _strip_parallel_prefix(key, metadata=False):
    while key.startswith('module.') or (metadata and key == 'module'):
        key = '' if key == 'module' else key[len('module.'):]
    return key


def _normalized_state_dict(state_dict):
    """Copy a checkpoint state dict and remove only leading DDP wrappers."""
    normalized = OrderedDict()
    for key, value in state_dict.items():
        normalized[_strip_parallel_prefix(key)] = value
    if hasattr(state_dict, '_metadata'):
        normalized._metadata = OrderedDict(
            (_strip_parallel_prefix(key, metadata=True), value)
            for key, value in state_dict._metadata.items()
        )
    return normalized


def _unwrap_parallel_policy(policy):
    wrappers = (nn.DataParallel, nn.parallel.DistributedDataParallel)
    while isinstance(policy, wrappers):
        policy = policy.module
    return policy


def _migrate_craftax_state_dict(state_dict):
    """Return ``(state_dict, migration)`` for known Craftax checkpoints."""
    state_dict = _normalized_state_dict(state_dict)
    canonical_weight = 'encoder.encoder.weight'
    canonical_bias = 'encoder.encoder.bias'
    migration = None

    # A short-lived version of CraftaxEncoder used a top-level canonical
    # Parameter. Accept those checkpoints by restoring the legacy key names.
    if 'encoder.weight' in state_dict and canonical_weight not in state_dict:
        state_dict[canonical_weight] = state_dict.pop('encoder.weight')
        if 'encoder.bias' in state_dict:
            state_dict[canonical_bias] = state_dict.pop('encoder.bias')
        migration = 'canonical-key rename'

    weight = state_dict.get(canonical_weight)
    if weight is not None and weight.ndim == 2 \
            and weight.shape[1] == CraftaxEncoder.OBS_SIZE:
        state_dict[canonical_weight] = (
            CraftaxEncoder.migrate_compact_linear_weight(weight))
        compact_migration = '843-to-8268 exact linear migration'
        if migration is not None:
            compact_migration = f'{migration} and {compact_migration}'
        return state_dict, compact_migration

    # Also accept checkpoints produced by the initial table-Parameter version
    # of this port. These already live in symbolic feature space and can be
    # concatenated without changing their forward function.
    split_keys = (
        'encoder.block_weight',
        'encoder.item_weight',
        'encoder.mob_weight',
        'encoder.visibility_weight',
        'encoder.tail_weight',
        'encoder.bias',
    )
    if all(key in state_dict for key in split_keys):
        block = state_dict['encoder.block_weight']
        item = state_dict['encoder.item_weight']
        mob = state_dict['encoder.mob_weight']
        visibility = state_dict['encoder.visibility_weight']
        tail = state_dict['encoder.tail_weight']
        bias = state_dict['encoder.bias']
        hidden_size = bias.numel()
        expected_shapes = {
            'block': (
                CraftaxEncoder.NUM_CELLS,
                CraftaxEncoder.NUM_BLOCK_TYPES,
                hidden_size,
            ),
            'item': (
                CraftaxEncoder.NUM_CELLS,
                CraftaxEncoder.NUM_ITEM_TYPES,
                hidden_size,
            ),
            'mob': (
                CraftaxEncoder.NUM_CELLS,
                CraftaxEncoder.NUM_MOB_CLASSES,
                CraftaxEncoder.NUM_MOB_TYPES,
                hidden_size,
            ),
            'visibility': (CraftaxEncoder.NUM_CELLS, hidden_size),
        }
        actual_shapes = {
            'block': tuple(block.shape),
            'item': tuple(item.shape),
            'mob': tuple(mob.shape),
            'visibility': tuple(visibility.shape),
        }
        if actual_shapes != expected_shapes:
            raise RuntimeError(
                'Unsupported split Craftax checkpoint parameter shapes: '
                f'{actual_shapes}; expected {expected_shapes}')
        if tuple(tail.shape) == (hidden_size, CraftaxEncoder.TAIL_SIZE):
            tail_weight = tail
        elif tuple(tail.shape) == (CraftaxEncoder.TAIL_SIZE, hidden_size):
            tail_weight = tail.t()
        else:
            raise RuntimeError(
                'Unsupported split Craftax tail_weight shape: '
                f'{tuple(tail.shape)}')
        cell_weight = torch.cat((
            block,
            item,
            mob.reshape(
                CraftaxEncoder.NUM_CELLS,
                CraftaxEncoder.NUM_MOB_CLASSES * CraftaxEncoder.NUM_MOB_TYPES,
                hidden_size,
            ),
            visibility.unsqueeze(1),
        ), dim=1)
        for key in split_keys:
            del state_dict[key]
        state_dict[canonical_weight] = torch.cat(
            (cell_weight.reshape(-1, hidden_size).t(), tail_weight), dim=1)
        state_dict[canonical_bias] = bias
        return state_dict, 'split-table exact migration'

    return state_dict, migration


def load_compatible_state_dict(policy, state_dict, checkpoint_path=None,
        warn=True):
    """Strictly load a policy while handling known encoder transitions.

    Craftax representations are migrated only where the old and new functions
    are mathematically equivalent. The old Impulse Wars raw-byte Linear cannot
    be converted to the structured CNN/categorical encoder, so a construction-
    time load keeps a :class:`DefaultEncoder` and emits a visible warning. A
    policy already wrapped by DDP must select that fallback before wrapping,
    because replacing Parameters behind an existing reducer is unsafe. Every
    other key and shape is still checked by ``strict=True``.
    """
    label = str(checkpoint_path) if checkpoint_path is not None else '<memory>'
    migration = None
    wrapped_policy = policy
    policy = _unwrap_parallel_policy(policy)

    if isinstance(policy.encoder, CraftaxEncoder):
        state_dict, migration = _migrate_craftax_state_dict(state_dict)
    else:
        state_dict = _normalized_state_dict(state_dict)

    if isinstance(policy.encoder, ImpulseWarsEncoder):
        legacy_weight = state_dict.get('encoder.encoder.weight')
        legacy_bias = state_dict.get('encoder.encoder.bias')
        if legacy_weight is not None and legacy_bias is not None \
                and legacy_weight.ndim == 2 \
                and legacy_weight.shape[1] == ImpulseWarsEncoder.OBS_SIZE:
            if wrapped_policy is not policy:
                raise RuntimeError(
                    f'Checkpoint {label} requires the Impulse Wars legacy '
                    'encoder fallback, but the policy is already wrapped for '
                    'data-parallel training. Set load_model_path/load_id so '
                    'the checkpoint is loaded before DDP construction, or set '
                    'torch.encoder=DefaultEncoder to select the legacy '
                    'architecture explicitly.')
            target_parameter = next(policy.encoder.parameters())
            legacy_encoder = DefaultEncoder(
                ImpulseWarsEncoder.OBS_SIZE,
                hidden_size=legacy_weight.shape[0],
            ).to(device=target_parameter.device, dtype=target_parameter.dtype)
            policy.encoder = legacy_encoder
            migration = 'Impulse Wars legacy raw-byte encoder fallback'
            if warn:
                warnings.warn(
                    f'Checkpoint {label} predates ImpulseWarsEncoder. Its '
                    'raw-byte DefaultEncoder was retained exactly because an '
                    'arbitrary byte-linear policy cannot be converted to the '
                    'new structured CNN/categorical encoder. Start a new run '
                    'without this checkpoint to use ImpulseWarsEncoder.',
                    UserWarning,
                    stacklevel=2,
                )

    if migration is not None and warn \
            and not migration.startswith('Impulse Wars'):
        warnings.warn(
            f'Checkpoint {label}: applied {migration}.',
            UserWarning,
            stacklevel=2,
        )

    try:
        return policy.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        encoder_name = type(policy.encoder).__name__
        raise RuntimeError(
            f'Checkpoint {label} is not compatible with {encoder_name} and '
            'the configured policy dimensions. PufferLib uses strict loading '
            'so a partial policy is never run. Match torch.encoder and '
            'policy.hidden_size/num_layers to the checkpoint, or start a new '
            'run for a changed architecture.'
        ) from exc

class MinimalEntityEncoder(nn.Module):
    def __init__(self, obs_size, hidden_size=128):
        super().__init__()
        self.self_obs_size = 2
        self.point_obs_size = 4
        self.num_points = 16
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            nn.Linear(self.self_obs_size + self.point_obs_size, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_size),
        )

    def forward(self, observations):
        self_obs = observations[:, :self.self_obs_size].unsqueeze(1).expand(
            observations.shape[0], self.num_points, self.self_obs_size)
        point_obs = observations[:, self.self_obs_size:].reshape(
            observations.shape[0], self.num_points, self.point_obs_size)
        cat = torch.cat([self_obs, point_obs], dim=-1)
        return self.encoder(cat).max(dim=1)[0]

class DefaultDecoder(nn.Module):
    def __init__(self, nvec, hidden_size=128):
        super().__init__()
        self.nvec = tuple(nvec)
        self.is_continuous = sum(nvec) == len(nvec)

        if self.is_continuous:
            num_atns = len(nvec)
            self.decoder_mean = nn.Linear(hidden_size, num_atns)
            self.decoder_logstd = nn.Parameter(torch.zeros(1, num_atns))
        else:
            self.decoder = nn.Linear(hidden_size, int(np.sum(nvec)))

        self.value_function = nn.Linear(hidden_size, 1)

    def forward(self, hidden):
        if self.is_continuous:
            mean = self.decoder_mean(hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            logits = torch.distributions.Normal(mean, torch.exp(logstd))
        else:
            logits = self.decoder(hidden)
            if len(self.nvec) > 1:
                logits = logits.split(self.nvec, dim=1)

        values = self.value_function(hidden)
        return logits, values

class MLP(nn.Module):
    def __init__(self, hidden_size, num_layers=1, **kwargs):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers += [nn.Linear(hidden_size, hidden_size), nn.GELU()]
        self.net = nn.Sequential(*layers)

    def initial_state(self, batch_size, device):
        return ()

    def forward_eval(self, h, state):
        return self.net(h), state

    def forward_train(self, h):
        return self.net(h)

class MinGRU(nn.Module):
    # https://arxiv.org/abs/2410.01201v1
    def __init__(self, hidden_size, num_layers=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, 3 * hidden_size, bias=False) for _ in range(num_layers)
        ])

    def _g(self, x):
        return torch.where(x >= 0, x + 0.5, x.sigmoid())

    def _log_g(self, x):
        return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

    def _highway(self, x, out, proj):
        g = proj.sigmoid()
        return g * out + (1.0 - g) * x

    def _heinsen_scan(self, log_coeffs, log_values):
        a_star = log_coeffs.cumsum(dim=1)
        return (a_star + (log_values - a_star).logcumsumexp(dim=1)).exp()

    def initial_state(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),)

    def forward_eval(self, h, state):
        state = state[0]
        assert state.shape[1] == h.shape[0]
        h = h.unsqueeze(1)
        state_out = []
        for i in range(self.num_layers):
            hidden, gate, proj = self.layers[i](h).chunk(3, dim=-1)
            out = torch.lerp(state[i:i+1].transpose(0, 1), self._g(hidden), gate.sigmoid())
            h = self._highway(h, out, proj)
            state_out.append(out[:, -1:])
        return h.squeeze(1), (torch.stack(state_out, 0).squeeze(2),)

    def forward_train(self, h):
        T = h.shape[1]
        for i in range(self.num_layers):
            hidden, gate, proj = self.layers[i](h).chunk(3, dim=-1)
            log_coeffs = -F.softplus(gate)
            log_values = -F.softplus(-gate) + self._log_g(hidden)
            out = self._heinsen_scan(log_coeffs, log_values)[:, -T:]
            h = self._highway(h, out, proj)
        return h

class LSTM(nn.Module):
    def __init__(self, hidden_size, num_layers=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        self.cell = nn.ModuleList([torch.nn.LSTMCell(hidden_size, hidden_size) for _ in range(num_layers)])

        for i in range(num_layers):
            cell = self.cell[i]
            w_ih = getattr(self.lstm, f'weight_ih_l{i}')
            w_hh = getattr(self.lstm, f'weight_hh_l{i}')
            b_ih = getattr(self.lstm, f'bias_ih_l{i}')
            b_hh = getattr(self.lstm, f'bias_hh_l{i}')
            nn.init.orthogonal_(w_ih, 1.0)
            nn.init.orthogonal_(w_hh, 1.0)
            b_ih.data.zero_()
            b_hh.data.zero_()
            cell.weight_ih = w_ih
            cell.weight_hh = w_hh
            cell.bias_ih = b_ih
            cell.bias_hh = b_hh

    def initial_state(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h, c

    def forward_eval(self, h, state):
        assert state[0].shape[1] == state[1].shape[1] == h.shape[0]
        lstm_h, lstm_c = state
        for i in range(self.num_layers):
            h, c = self.cell[i](h, (lstm_h[i], lstm_c[i]))
            lstm_h[i] = h
            lstm_c[i] = c
        return h, (lstm_h, lstm_c)

    def forward_train(self, h):
        # h: [B, T, H]
        h = h.transpose(0, 1)
        h, _ = self.lstm(h)
        return h.transpose(0, 1)

class GRU(nn.Module):
    def __init__(self, hidden_size, num_layers=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers)
        self.cell = nn.ModuleList([torch.nn.GRUCell(hidden_size, hidden_size) for _ in range(num_layers)])
        self.norm = torch.nn.RMSNorm(hidden_size)

        for i in range(num_layers):
            cell = self.cell[i]
            w_ih = getattr(self.gru, f'weight_ih_l{i}')
            w_hh = getattr(self.gru, f'weight_hh_l{i}')
            b_ih = getattr(self.gru, f'bias_ih_l{i}')
            b_hh = getattr(self.gru, f'bias_hh_l{i}')
            nn.init.orthogonal_(w_ih, 1.0)
            nn.init.orthogonal_(w_hh, 1.0)
            b_ih.data.zero_()
            b_hh.data.zero_()
            cell.weight_ih = w_ih
            cell.weight_hh = w_hh
            cell.bias_ih = b_ih
            cell.bias_hh = b_hh

    def initial_state(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h,)

    def forward_eval(self, h, state):
        assert state[0].shape[1] == h.shape[0]
        state = state[0]
        for i in range(self.num_layers):
            h_in = h
            h = self.cell[i](h, state[i])
            state[i] = h
            h = h + h_in
            h = self.norm(h)
        return h, (state,)

    def forward_train(self, h):
        # h: [B, T, H]
        h = h.transpose(0, 1)
        h_in = h
        h, _ = self.gru(h)
        h = h + h_in
        h = self.norm(h)
        return h.transpose(0, 1)

class NatureEncoder(nn.Module):
    '''NatureCNN encoder (Mnih et al. 2015). Returns [batch, hidden_size].'''
    def __init__(self, env, hidden_size=512, framestack=1, flat_size=64*7*7,
            channels_last=False, downsample=1, **kwargs):
        super().__init__()
        self.channels_last = channels_last
        self.downsample = downsample
        self.network = nn.Sequential(
            nn.Conv2d(framestack, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(flat_size, hidden_size),
            nn.ReLU(),
        )

    def forward(self, observations):
        if self.channels_last:
            observations = observations.permute(0, 3, 1, 2)
        if self.downsample > 1:
            observations = observations[:, :, ::self.downsample, ::self.downsample]
        return self.network(observations.float() / 255.0)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        inputs = x
        x = F.relu(x)
        x = self.conv0(x)
        x = F.relu(x)
        x = self.conv1(x)
        return x + inputs

class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(input_shape[0], out_channels, 3, padding=1)
        self.res_block0 = ResidualBlock(out_channels)
        self.res_block1 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)

class ImpalaEncoder(nn.Module):
    '''IMPALA ResNet encoder (Espeholt et al. 2018). Returns [batch, hidden_size].'''
    def __init__(self, env, hidden_size=256, cnn_width=16, **kwargs):
        super().__init__()
        h, w, c = env.single_observation_space.shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [cnn_width, 2*cnn_width, 2*cnn_width]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(shape[0] * shape[1] * shape[2], hidden_size),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)

    def forward(self, observations):
        return self.network(observations.permute(0, 3, 1, 2).float() / 255.0)
