"""Parity and device coverage for the optimized Torch policy sampler."""

import pytest


torch = pytest.importorskip("torch")
from torch.distributions.utils import logits_to_probs

try:
    from pufferlib import _C
except Exception as exc:  # pragma: no cover - source-only installations
    pytest.skip(f"pufferlib._C is not built: {exc}", allow_module_level=True)

if getattr(_C, "precision_bytes", None) != 4:
    pytest.skip(
        "the Torch backend requires a float32 pufferlib._C build",
        allow_module_level=True,
    )

from pufferlib.torch_pufferl import sample_logits


def _reference_entropy(log_probs):
    min_real = torch.finfo(log_probs.dtype).min
    safe_log_probs = torch.clamp(log_probs, min=min_real)
    return -(safe_log_probs * logits_to_probs(log_probs)).sum(-1)


def _reference_sample_logits(logits, action=None):
    """The pre-optimization implementation."""
    is_discrete = isinstance(logits, torch.Tensor)
    if isinstance(logits, torch.distributions.Normal):
        batch = logits.loc.shape[0]
        if action is None:
            action = logits.sample().view(batch, -1)
        action = action.view(batch, -1)
        log_probs = logits.log_prob(action).sum(1)
        entropy = logits.entropy().view(batch, -1).sum(1)
        return action, log_probs, entropy
    if is_discrete:
        logits = logits.unsqueeze(0)
    else:
        logits = torch.nn.utils.rnn.pad_sequence(
            [head.transpose(0, 1) for head in logits],
            batch_first=False,
            padding_value=-torch.inf,
        ).permute(1, 2, 0)

    normalized = logits - logits.logsumexp(dim=-1, keepdim=True)
    probabilities = logits_to_probs(logits)
    if action is None:
        probabilities = torch.nan_to_num(
            probabilities, 1e-8, 1e-8, 1e-8
        )
        action = torch.multinomial(
            probabilities.reshape(-1, probabilities.shape[-1]),
            1,
            replacement=True,
        ).int()
        action = action.reshape(probabilities.shape[:-1])
    else:
        batch = logits[0].shape[0]
        action = action.view(batch, -1).T
    selected = normalized.gather(-1, action.long().unsqueeze(-1)).squeeze(-1)
    entropy = _reference_entropy(normalized).sum(0)
    if is_discrete:
        return action.T, selected.squeeze(0), entropy.squeeze(0)
    return action.T, selected.sum(0), entropy


@pytest.fixture(params=("cpu", "mps"))
def device(request):
    if request.param == "mps" and not (
        torch.backends.mps.is_built() and torch.backends.mps.is_available()
    ):
        pytest.skip("MPS is not available in this PyTorch runtime")
    return torch.device(request.param)


@pytest.mark.parametrize("kind", ("discrete", "multidiscrete", "normal"))
def test_supplied_action_matches_reference_and_gradients(device, kind):
    batch = 257
    generator = torch.Generator().manual_seed(20260822)
    if kind == "discrete":
        raw = [torch.randn(batch, 7, generator=generator)]
        actions = torch.randint(0, 7, (batch, 1), generator=generator)
    elif kind == "multidiscrete":
        raw = [
            torch.randn(batch, 3, generator=generator),
            torch.randn(batch, 5, generator=generator),
            torch.randn(batch, 2, generator=generator),
        ]
        actions = torch.stack(
            [
                torch.randint(0, 3, (batch,), generator=generator),
                torch.randint(0, 5, (batch,), generator=generator),
                torch.randint(0, 2, (batch,), generator=generator),
            ],
            dim=1,
        )
    else:
        raw = [
            torch.randn(batch, 4, generator=generator),
            torch.rand(batch, 4, generator=generator) + 0.2,
        ]
        actions = torch.randn(batch, 4, generator=generator)

    reference_raw = [
        value.detach().clone().to(device).requires_grad_() for value in raw
    ]
    candidate_raw = [
        value.detach().clone().to(device).requires_grad_() for value in raw
    ]
    actions = actions.to(device)

    if kind == "discrete":
        reference_logits = reference_raw[0]
        candidate_logits = candidate_raw[0]
    elif kind == "multidiscrete":
        reference_logits = tuple(reference_raw)
        candidate_logits = tuple(candidate_raw)
    else:
        reference_logits = torch.distributions.Normal(
            reference_raw[0], reference_raw[1]
        )
        candidate_logits = torch.distributions.Normal(
            candidate_raw[0], candidate_raw[1]
        )

    reference_action, reference_logprob, reference_entropy = (
        _reference_sample_logits(reference_logits, actions)
    )
    candidate_action, candidate_logprob, candidate_entropy = sample_logits(
        candidate_logits, action=actions
    )

    torch.testing.assert_close(candidate_action, reference_action, rtol=0, atol=0)
    torch.testing.assert_close(
        candidate_logprob, reference_logprob, rtol=2e-6, atol=2e-6
    )
    torch.testing.assert_close(
        candidate_entropy, reference_entropy, rtol=2e-6, atol=2e-6
    )

    reference_loss = reference_logprob.square().mean() + reference_entropy.mean()
    candidate_loss = candidate_logprob.square().mean() + candidate_entropy.mean()
    reference_loss.backward()
    candidate_loss.backward()
    for candidate, reference in zip(candidate_raw, reference_raw):
        torch.testing.assert_close(
            candidate.grad, reference.grad, rtol=2e-5, atol=2e-6
        )


@pytest.mark.parametrize("kind", ("discrete", "multidiscrete", "normal"))
def test_no_entropy_path_preserves_supplied_action_logprob(device, kind):
    batch = 113
    if kind == "discrete":
        logits = torch.linspace(-3, 3, batch * 5, device=device).reshape(batch, 5)
        actions = (torch.arange(batch, device=device) % 5).reshape(batch, 1)
    elif kind == "multidiscrete":
        logits = (
            torch.randn(batch, 3, device=device),
            torch.randn(batch, 6, device=device),
        )
        actions = torch.stack(
            (
                torch.arange(batch, device=device) % 3,
                torch.arange(batch, device=device) % 6,
            ),
            dim=1,
        )
    else:
        logits = torch.distributions.Normal(
            torch.randn(batch, 2, device=device),
            torch.rand(batch, 2, device=device) + 0.1,
        )
        actions = torch.randn(batch, 2, device=device)

    full_action, full_logprob, full_entropy = sample_logits(
        logits, action=actions, compute_entropy=True
    )
    fast_action, fast_logprob, fast_entropy = sample_logits(
        logits, action=actions, compute_entropy=False
    )

    torch.testing.assert_close(fast_action, full_action, rtol=0, atol=0)
    torch.testing.assert_close(fast_logprob, full_logprob, rtol=0, atol=0)
    assert full_entropy is not None
    assert fast_entropy is None


@pytest.mark.parametrize(
    ("logits", "expected_action_shape"),
    [
        (lambda device: torch.randn(127, 7, device=device), (127, 1)),
        (
            lambda device: (
                torch.randn(127, 3, device=device),
                torch.randn(127, 5, device=device),
            ),
            (127, 2),
        ),
        (
            lambda device: torch.distributions.Normal(
                torch.randn(127, 4, device=device),
                torch.rand(127, 4, device=device) + 0.1,
            ),
            (127, 4),
        ),
    ],
)
def test_rollout_sampling_without_entropy(device, logits, expected_action_shape):
    action, logprob, entropy = sample_logits(
        logits(device), compute_entropy=False
    )

    assert action.shape == expected_action_shape
    assert logprob.shape == (expected_action_shape[0],)
    assert torch.isfinite(logprob).all()
    assert entropy is None


@pytest.mark.parametrize("kind", ("discrete", "multidiscrete", "normal"))
def test_seeded_rollout_sampling_matches_preoptimization_path(device, kind):
    batch = 1_024
    generator = torch.Generator().manual_seed(7351)
    if kind == "discrete":
        logits = torch.randn(batch, 7, generator=generator).to(device)
    elif kind == "multidiscrete":
        logits = (
            torch.randn(batch, 3, generator=generator).to(device),
            torch.randn(batch, 8, generator=generator).to(device),
            torch.randn(batch, 5, generator=generator).to(device),
        )
    else:
        logits = torch.distributions.Normal(
            torch.randn(batch, 4, generator=generator).to(device),
            (torch.rand(batch, 4, generator=generator) + 0.2).to(device),
        )

    torch.manual_seed(918273)
    if device.type == "mps":
        torch.mps.manual_seed(918273)
    reference_action, reference_logprob, _ = _reference_sample_logits(logits)

    torch.manual_seed(918273)
    if device.type == "mps":
        torch.mps.manual_seed(918273)
    candidate_action, candidate_logprob, entropy = sample_logits(
        logits, compute_entropy=False
    )

    torch.testing.assert_close(candidate_action, reference_action, rtol=0, atol=0)
    torch.testing.assert_close(
        candidate_logprob, reference_logprob, rtol=2e-6, atol=2e-6
    )
    assert entropy is None


def test_breakout_seeded_action_trajectory_is_bitwise_reproducible(device):
    """Protect the exact boundary case that exp(log_softmax) changes."""
    logits = torch.randn(
        4_096, 3, generator=torch.Generator().manual_seed(42)
    ).to(device)

    torch.manual_seed(1028)
    if device.type == "mps":
        torch.mps.manual_seed(1028)
    reference_action, reference_logprob, _ = _reference_sample_logits(logits)

    torch.manual_seed(1028)
    if device.type == "mps":
        torch.mps.manual_seed(1028)
    candidate_action, candidate_logprob, entropy = sample_logits(
        logits, compute_entropy=False
    )

    torch.testing.assert_close(candidate_action, reference_action, rtol=0, atol=0)
    torch.testing.assert_close(
        candidate_logprob, reference_logprob, rtol=0, atol=0
    )
    assert entropy is None
