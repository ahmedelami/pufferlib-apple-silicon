"""Exact-shape Metal MinGRU training scan for the validated M5 Pro path.

The public trainer enables this module only behind its fail-closed runtime
guards. Import and fake-tensor tracing are CPU-safe: the Metal library is
compiled lazily on the first real MPS dispatch. Evaluation intentionally keeps
``MinGRU.forward_eval`` unchanged.
"""

from functools import lru_cache

import torch


BATCH_SEGMENTS = 1_024
HORIZON = 64
HIDDEN_SIZE = 64
COMBINED_SIZE = 3 * HIDDEN_SIZE
THREADS = BATCH_SEGMENTS * HIDDEN_SIZE


_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;

constant constexpr uint kBatch = 1024;
constant constexpr uint kHorizon = 64;
constant constexpr uint kHidden = 64;
constant constexpr uint kCombined = 192;

inline float stable_sigmoid(float value) {
    const float z = metal::precise::exp(-metal::abs(value));
    return value >= 0.0f ? 1.0f / (1.0f + z) : z / (1.0f + z);
}

inline float candidate_value(float hidden) {
    return hidden >= 0.0f ? hidden + 0.5f : stable_sigmoid(hidden);
}

// Match the branch-stable lerp used by PufferLib's native CUDA backend.
inline float stable_lerp(float begin, float end, float weight) {
    const float difference = end - begin;
    return metal::abs(weight) < 0.5f
        ? begin + weight * difference
        : end - difference * (1.0f - weight);
}

kernel void puff_mingru_train_scan_forward_f32(
        device float* output,
        device float* scan_state,
        device const float* combined,
        device const float* input,
        uint lane [[thread_position_in_grid]]) {
    if (lane >= kBatch * kHidden) {
        return;
    }

    const uint batch = lane / kHidden;
    const uint hidden_index = lane - batch * kHidden;
    float state = 0.0f;
    for (uint timestep = 0; timestep < kHorizon; ++timestep) {
        const uint input_index =
            (batch * kHorizon + timestep) * kHidden + hidden_index;
        const uint combined_base =
            (batch * kHorizon + timestep) * kCombined;
        const float hidden = combined[combined_base + hidden_index];
        const float gate = combined[
            combined_base + kHidden + hidden_index];
        const float projection = combined[
            combined_base + 2 * kHidden + hidden_index];
        const float candidate = candidate_value(hidden);
        const float update = stable_sigmoid(gate);
        const float highway = stable_sigmoid(projection);
        const float input_value = input[input_index];

        state = stable_lerp(state, candidate, update);
        scan_state[input_index] = state;
        output[input_index] =
            highway * state + (1.0f - highway) * input_value;
    }
}

kernel void puff_mingru_train_scan_backward_f32(
        device float* grad_combined,
        device float* grad_input,
        device const float* combined,
        device const float* input,
        device const float* scan_state,
        device const float* grad_output,
        uint lane [[thread_position_in_grid]]) {
    if (lane >= kBatch * kHidden) {
        return;
    }

    const uint batch = lane / kHidden;
    const uint hidden_index = lane - batch * kHidden;
    float carry = 0.0f;
    for (int timestep = int(kHorizon) - 1; timestep >= 0; --timestep) {
        const uint input_index =
            (batch * kHorizon + uint(timestep)) * kHidden + hidden_index;
        const uint combined_base =
            (batch * kHorizon + uint(timestep)) * kCombined;
        const float hidden = combined[combined_base + hidden_index];
        const float gate = combined[
            combined_base + kHidden + hidden_index];
        const float projection = combined[
            combined_base + 2 * kHidden + hidden_index];
        const float input_value = input[input_index];
        const float state = scan_state[input_index];
        const float previous_state = timestep == 0
            ? 0.0f
            : scan_state[input_index - kHidden];
        const float output_gradient = grad_output[input_index];

        const float candidate = candidate_value(hidden);
        const float update = stable_sigmoid(gate);
        const float highway = stable_sigmoid(projection);
        const float state_gradient = carry + output_gradient * highway;
        const float candidate_derivative = hidden >= 0.0f
            ? 1.0f
            : candidate * (1.0f - candidate);

        grad_combined[combined_base + hidden_index] =
            state_gradient * update * candidate_derivative;
        grad_combined[combined_base + kHidden + hidden_index] =
            state_gradient * (candidate - previous_state)
                * update * (1.0f - update);
        grad_combined[combined_base + 2 * kHidden + hidden_index] =
            output_gradient * (state - input_value)
                * highway * (1.0f - highway);
        grad_input[input_index] = output_gradient * (1.0f - highway);
        carry = state_gradient * (1.0f - update);
    }
}
'''


@lru_cache(maxsize=1)
def _library():
    return torch.mps.compile_shader(_SOURCE)


def _validate_forward(combined, input):
    if tuple(combined.shape) != (
            BATCH_SEGMENTS, HORIZON, COMBINED_SIZE):
        raise RuntimeError(
            'Metal MinGRU combined tensor must have shape [1024,64,192]')
    if tuple(input.shape) != (BATCH_SEGMENTS, HORIZON, HIDDEN_SIZE):
        raise RuntimeError(
            'Metal MinGRU input tensor must have shape [1024,64,64]')
    if combined.device.type != 'mps' or input.device.type != 'mps':
        raise RuntimeError('Metal MinGRU inputs must be on MPS')
    if combined.dtype != torch.float32 or input.dtype != torch.float32:
        raise RuntimeError('Metal MinGRU is validated only for float32')
    if not combined.is_contiguous() or not input.is_contiguous():
        raise RuntimeError('Metal MinGRU inputs must be contiguous')


def _validate_backward(combined, input, scan_state, grad_output):
    _validate_forward(combined, input)
    expected_shape = (BATCH_SEGMENTS, HORIZON, HIDDEN_SIZE)
    if (tuple(scan_state.shape) != expected_shape
            or scan_state.device.type != 'mps'
            or scan_state.dtype != torch.float32
            or not scan_state.is_contiguous()):
        raise RuntimeError(
            'Metal MinGRU scan state must be contiguous MPS float32 '
            'with shape [1024,64,64]')
    if (tuple(grad_output.shape) != expected_shape
            or grad_output.device.type != 'mps'
            or grad_output.dtype != torch.float32
            or not grad_output.is_contiguous()):
        raise RuntimeError(
            'Metal MinGRU output gradient must be contiguous MPS float32 '
            'with shape [1024,64,64]')


@torch.library.custom_op(
    'pufferlib::mingru_train_scan_forward',
    mutates_args=(),
    device_types='mps',
)
def _scan_forward(combined: torch.Tensor, input: torch.Tensor) \
        -> tuple[torch.Tensor, torch.Tensor]:
    _validate_forward(combined, input)
    output = torch.empty_like(input)
    scan_state = torch.empty_like(input)
    _library().puff_mingru_train_scan_forward_f32(
        output,
        scan_state,
        combined,
        input,
        threads=[THREADS, 1, 1],
        group_size=[256, 1, 1],
    )
    return output, scan_state


@_scan_forward.register_fake
def _(combined, input):
    return torch.empty_like(input), torch.empty_like(input)


@torch.library.custom_op(
    'pufferlib::mingru_train_scan_backward',
    mutates_args=(),
    device_types='mps',
)
def _scan_backward(
        combined: torch.Tensor,
        input: torch.Tensor,
        scan_state: torch.Tensor,
        grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_backward(combined, input, scan_state, grad_output)
    grad_combined = torch.empty_like(combined)
    grad_input = torch.empty_like(input)
    _library().puff_mingru_train_scan_backward_f32(
        grad_combined,
        grad_input,
        combined,
        input,
        scan_state,
        grad_output,
        threads=[THREADS, 1, 1],
        group_size=[256, 1, 1],
    )
    return grad_combined, grad_input


@_scan_backward.register_fake
def _(combined, input, scan_state, grad_output):
    del scan_state, grad_output
    return torch.empty_like(combined), torch.empty_like(input)


def _setup_context(ctx, inputs, output):
    combined, input = inputs
    _, scan_state = output
    ctx.mark_non_differentiable(scan_state)
    ctx.save_for_backward(combined, input, scan_state)


def _backward(ctx, grad_output, _grad_scan_state):
    combined, input, scan_state = ctx.saved_tensors
    return _scan_backward(
        combined, input, scan_state, grad_output.contiguous())


_scan_forward.register_autograd(_backward, setup_context=_setup_context)


def mingru_train_scan(combined, input):
    """Return one highway-connected fixed-shape MinGRU sequence."""
    output, _scan_state = _scan_forward(combined, input)
    return output


def forward_train(network, hidden):
    """Exact 2x64 training forward used by the guarded instance override."""
    if tuple(hidden.shape) != (BATCH_SEGMENTS, HORIZON, HIDDEN_SIZE):
        raise RuntimeError(
            'Metal MinGRU requires training input shape [1024,64,64]')
    if (network.hidden_size != HIDDEN_SIZE
            or network.num_layers != 2
            or len(network.layers) != 2):
        raise RuntimeError('Metal MinGRU requires exactly two width-64 layers')
    for layer in network.layers:
        combined = layer(hidden)
        if not combined.is_contiguous() or not hidden.is_contiguous():
            raise RuntimeError('Metal MinGRU operands must remain contiguous')
        hidden = mingru_train_scan(combined, hidden)
    return hidden


__all__ = ['forward_train', 'mingru_train_scan']
