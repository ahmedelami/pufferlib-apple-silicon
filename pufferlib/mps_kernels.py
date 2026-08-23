"""Small fused Metal kernels for launch-heavy portable trainer operations."""

from functools import lru_cache
import importlib

import torch


_ADVANTAGE_LIBRARY = None

_ADVANTAGE_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;

kernel void puff_advantage(
        device float* dispatch,
        device const float* values,
        device const float* rewards,
        device const float* terminals,
        device const float* importance,
        device float* advantages,
        constant uint& horizon,
        constant float& gamma,
        constant float& gae_lambda,
        constant float& rho_clip,
        constant float& c_clip,
        uint row [[thread_position_in_grid]]) {
    const uint offset = row * horizon;
    float last = 0.0f;
    for (int t = int(horizon) - 2; t >= 0; --t) {
        const uint current = offset + uint(t);
        const uint next = current + 1;
        const float next_nonterminal = 1.0f - terminals[next];
        const float imp = importance[current];
        const float rho = min(imp, rho_clip);
        const float c = min(imp, c_clip);
        const float delta = rho * (
            rewards[next]
            + gamma * values[next] * next_nonterminal
            - values[current]);
        last = delta + gamma * gae_lambda * c * last * next_nonterminal;
        advantages[current] = last;
    }
    // The first tensor determines the one-dimensional dispatch width.
    dispatch[row] = last;
}
'''


_CATEGORICAL_SOURCE = r'''
#include <metal_stdlib>
#include <c10/metal/random.h>

using namespace metal;

constant constexpr float kNanReplacement = 1.0e-8f;
constant constexpr float kFloatEpsilon =
    ::metal::numeric_limits<float>::epsilon();

inline float sanitized_probability(float value) {
    // Mirrors torch.nan_to_num(probs, 1e-8, 1e-8, 1e-8).
    return ::metal::isfinite(value) ? value : kNanReplacement;
}

inline bool candidate_wins(float candidate, float incumbent) {
    // Match torch.argmax: NaN wins over a non-NaN and the first tie wins.
    const bool candidate_nan = ::metal::isnan(candidate);
    const bool incumbent_nan = ::metal::isnan(incumbent);
    return (!incumbent_nan && candidate_nan) ||
        (!candidate_nan && !incumbent_nan && candidate > incumbent);
}

kernel void puff_sample_philox_exp_race(
        device int* actions,
        device float* sampled_logprobs,
        device const float* probs,
        device const float* log_probs,
        constant long& seed,
        constant long& base_offset,
        constant uint& rows,
        constant uint& categories,
        uint row [[thread_position_in_grid]]) {
    if (row >= rows || categories == 0) {
        return;
    }

    const ulong base =
        static_cast<ulong>(row) * static_cast<ulong>(categories);
    uint best_category = 0;
    float best_score = 0.0f;
    bool have_best = false;
    ulong cached_round = ~0ul;
    uint4 raw = uint4(0);

    for (uint category = 0; category < categories; ++category) {
        const ulong flat_index = static_cast<ulong>(base + category);
        const ulong round = flat_index >> 2;
        // Widths not divisible by four can share a pure Philox round across
        // rows. Recomputing it is safe; reservation still follows the flat
        // four-values-per-round sequence used by MPS exponential_.
        if (round != cached_round) {
            raw = c10::metal::philox4::rand(
                seed, base_offset + long(round));
            cached_round = round;
        }

        const uint lane = static_cast<uint>(flat_index & 3ul);
        const float uniform = ::metal::min(
            c10::metal::detail::uint32_to_uniform_float(raw[lane]),
            1.0f - kFloatEpsilon);
        const float exponential =
            -::metal::precise::log(1.0f - uniform);
        const float score =
            sanitized_probability(probs[base + category]) / exponential;
        if (!have_best || candidate_wins(score, best_score)) {
            best_score = score;
            best_category = category;
            have_best = true;
        }
    }

    actions[row] = static_cast<int>(best_category);
    sampled_logprobs[row] = log_probs[base + best_category];
}
'''


def _advantage_library():
    global _ADVANTAGE_LIBRARY
    if _ADVANTAGE_LIBRARY is None:
        _ADVANTAGE_LIBRARY = torch.mps.compile_shader(_ADVANTAGE_SOURCE)
    return _ADVANTAGE_LIBRARY


@lru_cache(maxsize=1)
def _categorical_library():
    # DynamicMetalLib retains the compiled library and per-kernel pipeline
    # cache for the process lifetime.
    return torch.mps.compile_shader(_CATEGORICAL_SOURCE)


def philox_rounds(num_values):
    """Counter rounds consumed by MPS exponential_ for flat float32 output."""
    num_values = int(num_values)
    if num_values <= 0:
        raise ValueError('Philox reservation requires a positive value count')
    return (num_values + 3) // 4


class MPSCategoricalSampler:
    """Internal persistent sampler for the validated MPS rollout path.

    Returned buffers are owned by this object and overwritten by its next
    ``sample`` call. The caller must copy them before reuse.
    """

    def __init__(self, rows, categories):
        self.rows = int(rows)
        self.categories = int(categories)
        if self.rows <= 0 or self.categories <= 0:
            raise ValueError('sampler rows and categories must be positive')
        if self.rows > 0xFFFF_FFFF or self.categories > 0xFFFF_FFFF \
                or self.rows * self.categories > 0xFFFF_FFFF_FFFF_FFFF:
            raise ValueError('sampler shape exceeds Metal index limits')
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS is unavailable')
        torch_version = str(torch.__version__).split('+', 1)[0]
        torch_git = getattr(torch.version, 'git_version', None)
        if torch_version != '2.13.0' \
                or torch_git != 'cf30153c4c131c8164ee7798e5022d810682e2cb':
            raise RuntimeError(
                'fused MPS sampling requires the validated PyTorch 2.13.0 '
                'build at cf30153c4c131c8164ee7798e5022d810682e2cb')

        self._bridge = importlib.import_module('pufferlib._mps_rng')
        reserve = getattr(self._bridge, 'reserve_default_mps_philox', None)
        if not callable(reserve):
            raise RuntimeError('the atomic MPS RNG reservation bridge is invalid')
        self._reserve = reserve
        self._library = _categorical_library()
        self.actions = torch.empty(
            self.rows, dtype=torch.int32, device='mps')
        self.sampled_logprobs = torch.empty(
            self.rows, dtype=torch.float32, device='mps')
        self.rounds = philox_rounds(self.rows * self.categories)
        self._preflight()

    def _preflight(self):
        """Materialize the pipeline without consuming global RNG state."""
        rng_before = torch.mps.get_rng_state().clone()
        probs = torch.full(
            (self.rows, self.categories),
            1.0 / self.categories,
            dtype=torch.float32,
            device='mps')
        log_probs = torch.zeros_like(probs)
        # Fixed explicit counter state validates the exact dispatch without
        # reserving from or otherwise mutating PyTorch's default generator.
        self._library.puff_sample_philox_exp_race(
            self.actions,
            self.sampled_logprobs,
            probs,
            log_probs,
            0,
            0,
            self.rows,
            self.categories,
        )
        torch.mps.synchronize()
        if not torch.equal(torch.mps.get_rng_state(), rng_before):
            raise RuntimeError('fused sampler preflight mutated MPS RNG state')

    def sample(self, probs, log_probs):
        expected_shape = (self.rows, self.categories)
        tensors = (probs, log_probs)
        if any(
                tensor.device.type != 'mps'
                or tensor.dtype != torch.float32
                or not tensor.is_contiguous()
                for tensor in tensors):
            raise RuntimeError(
                'fused categorical inputs must be contiguous MPS float32')
        if tuple(probs.shape) != expected_shape \
                or tuple(log_probs.shape) != expected_shape:
            raise RuntimeError(
                f'fused categorical inputs must have shape {expected_shape}')
        if probs.requires_grad or log_probs.requires_grad:
            raise RuntimeError('fused rollout sampling does not support autograd')

        seed, base_offset = self._reserve(self.rounds)
        self._library.puff_sample_philox_exp_race(
            self.actions,
            self.sampled_logprobs,
            probs,
            log_probs,
            seed,
            base_offset,
            self.rows,
            self.categories,
        )
        return self.actions, self.sampled_logprobs


def puff_advantage(values, rewards, terminals, importance, advantages,
        gamma, gae_lambda, rho_clip, c_clip, dispatch=None):
    """Compute Puffer/V-trace advantages with one Metal thread per agent."""
    tensors = (values, rewards, terminals, importance, advantages)
    if any(t.device.type != 'mps' or t.dtype != torch.float32
            or not t.is_contiguous() for t in tensors):
        raise ValueError('Metal advantage kernel requires contiguous MPS float32 tensors')
    if values.shape != rewards.shape or values.shape != terminals.shape \
            or values.shape != importance.shape or values.shape != advantages.shape:
        raise ValueError('Metal advantage tensors must have identical shapes')
    num_steps, horizon = values.shape
    if dispatch is None:
        dispatch = torch.empty(num_steps, dtype=torch.float32, device='mps')
    if (dispatch.device.type != 'mps' or dispatch.dtype != torch.float32
            or not dispatch.is_contiguous() or dispatch.numel() != num_steps):
        raise ValueError('Metal advantage dispatch buffer must be contiguous MPS float32')

    _advantage_library().puff_advantage(
        dispatch, values, rewards, terminals, importance, advantages,
        horizon, gamma, gae_lambda, rho_clip, c_clip)
    return advantages
