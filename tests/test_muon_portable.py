"""Portable regression coverage for the local Muon optimizer."""

from copy import deepcopy

import pytest


torch = pytest.importorskip("torch")

from pufferlib.muon import NS_COEFS, Muon


def _legacy_zeropower(gradient, eps=1e-7):
    original = gradient.clone()
    value = original
    if original.size(-2) > original.size(-1):
        value = value.mT
    value = value / torch.clamp(
        original.norm(dim=(-2, -1)), min=eps
    )
    for a, b, c in NS_COEFS:
        square = value @ value.mT
        polynomial = c * square
        polynomial.diagonal(dim1=-2, dim2=-1).add_(b)
        polynomial = polynomial @ square
        polynomial.diagonal(dim1=-2, dim2=-1).add_(a)
        value = polynomial @ value
    if original.size(-2) > original.size(-1):
        value = value.mT
    return value.to(original.dtype)


def _legacy_step(parameters, buffers, gradients, lr, momentum, weight_decay):
    with torch.no_grad():
        for parameter, buffer, source_gradient in zip(
            parameters, buffers, gradients
        ):
            gradient = source_gradient.clone()
            buffer.mul_(momentum)
            buffer.add_(gradient)
            gradient.add_(buffer * momentum)
            if gradient.ndim >= 2:
                gradient = gradient.view(gradient.shape[0], -1)
                gradient = _legacy_zeropower(gradient)
                gradient *= max(
                    1, gradient.size(-2) / gradient.size(-1)
                ) ** 0.5
            parameter.mul_(1 - lr * weight_decay)
            parameter.sub_(lr * gradient.view(parameter.shape))


@pytest.fixture(params=("cpu", "mps"))
def device(request):
    if request.param == "mps" and not (
        torch.backends.mps.is_built() and torch.backends.mps.is_available()
    ):
        pytest.skip("MPS is not available in this PyTorch runtime")
    return torch.device(request.param)


@pytest.mark.parametrize("weight_decay", (0.0, 0.07))
def test_muon_matches_preoptimization_updates(device, weight_decay):
    generator = torch.Generator().manual_seed(4491)
    shapes = ((11, 7), (5,), (4, 3, 2), (3, 13))
    initial = [
        torch.randn(shape, generator=generator, dtype=torch.float32)
        for shape in shapes
    ]
    reference = [value.to(device).clone() for value in initial]
    candidate = [
        torch.nn.Parameter(value.to(device).clone()) for value in initial
    ]
    buffers = [torch.zeros_like(value) for value in reference]
    optimizer = Muon(
        candidate,
        lr=0.0031,
        momentum=0.87,
        weight_decay=weight_decay,
    )

    for step in range(20):
        gradients = [
            torch.sin(
                torch.arange(value.numel(), dtype=torch.float32).reshape(value.shape)
                * (0.013 + step * 0.001)
                + step
            ).to(device)
            for value in initial
        ]
        _legacy_step(
            reference, buffers, gradients,
            lr=0.0031, momentum=0.87, weight_decay=weight_decay,
        )
        for parameter, gradient in zip(candidate, gradients):
            parameter.grad = gradient.clone()
        optimizer.step()
        optimizer.zero_grad()

    for actual, expected in zip(candidate, reference):
        torch.testing.assert_close(
            actual, expected, rtol=3e-6, atol=3e-6
        )
    for parameter, expected_buffer in zip(candidate, buffers):
        torch.testing.assert_close(
            optimizer.state[parameter]["momentum_buffer"],
            expected_buffer,
            rtol=2e-6,
            atol=2e-6,
        )


def test_muon_state_dict_continuation(device):
    parameter = torch.nn.Parameter(
        torch.linspace(-1, 1, 63, device=device).reshape(9, 7)
    )
    optimizer = Muon([parameter], lr=0.002, momentum=0.91)
    parameter.grad = torch.cos(parameter.detach() * 3)
    optimizer.step()

    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed = Muon([resumed_parameter], lr=0.002, momentum=0.91)
    resumed.load_state_dict(deepcopy(optimizer.state_dict()))

    for step in range(10):
        gradient = torch.sin(
            torch.arange(63, device=device, dtype=torch.float32).reshape(9, 7)
            * 0.01
            + step
        )
        parameter.grad = gradient.clone()
        resumed_parameter.grad = gradient.clone()
        optimizer.step()
        resumed.step()
        optimizer.zero_grad()
        resumed.zero_grad()

    torch.testing.assert_close(
        resumed_parameter, parameter, rtol=0, atol=0
    )
    torch.testing.assert_close(
        resumed.state[resumed_parameter]["momentum_buffer"],
        optimizer.state[parameter]["momentum_buffer"],
        rtol=0,
        atol=0,
    )
