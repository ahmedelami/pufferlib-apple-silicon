"""Standalone diagnostic comparing PufferLib Muon with heavyball ForeachMuon."""

import warnings

import pytest
import torch
import torch.nn as nn

from pufferlib.muon import Muon as TorchMuon


__test__ = False
pytestmark = pytest.mark.optional

warnings.filterwarnings(action="ignore", category=UserWarning, module=r"heavyball.*")
heavyball = pytest.importorskip(
    "heavyball", reason="Muon comparison requires the optional heavyball package"
)
ForeachMuon = getattr(heavyball, "ForeachMuon", None)
if ForeachMuon is None:
    pytest.skip(
        "Muon comparison requires the legacy heavyball.ForeachMuon API",
        allow_module_level=True,
    )


CONFIG = {
    "learning_rate": 1e-3,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_eps": 1e-8,
}


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(10, 20, bias=True)
        self.act = nn.ReLU()
        self.l2 = nn.Linear(20, 1, bias=True)

    def forward(self, x):
        return self.l2(self.act(self.l1(x)))


def compare_muon_implementations(n_epochs=5):
    """Run the original print-oriented comparison without pytest collecting it."""
    heavyball.utils.compile_mode = "default"
    torch.manual_seed(42)
    torch.set_num_threads(1)

    model1 = Net()
    model2 = Net()
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        p2.data.copy_(p1.data)

    x = torch.randn(16, 10)
    y = torch.randn(16, 1)
    heavy_optimizer = ForeachMuon(
        model1.parameters(),
        lr=CONFIG["learning_rate"],
        betas=(CONFIG["adam_beta1"], CONFIG["adam_beta2"]),
        eps=CONFIG["adam_eps"],
        heavyball_momentum=True,
        compile_step=False,
    )
    torch_optimizer = TorchMuon(
        model2.parameters(),
        lr=CONFIG["learning_rate"],
        momentum=CONFIG["adam_beta1"],
        eps=CONFIG["adam_eps"],
        weight_decay=0.0,
    )
    loss_fn = nn.MSELoss()

    print(f"{'Epoch':<6} {'AllClose':<10} {'Max Abs Diff':<15}")
    print("-" * 35)
    for epoch in range(n_epochs):
        heavy_optimizer.zero_grad()
        torch_optimizer.zero_grad()
        loss1 = loss_fn(model1(x), y)
        loss2 = loss_fn(model2(x), y)
        loss1.backward()
        loss2.backward()
        heavy_optimizer.step()
        torch_optimizer.step()

        all_close = True
        max_diff = 0.0
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            diff = (p1.data - p2.data).abs()
            max_diff = max(max_diff, diff.max().item())
            if not torch.allclose(p1.data, p2.data, atol=1e-6, rtol=1e-5):
                all_close = False

        print(f"{epoch + 1:<6} {str(all_close):<10} {max_diff:<15.3e}")
        if not all_close and max_diff > 1e-4:
            print("❗ Significant divergence detected.")
            break

    print("\\n✅ Test complete.")


if __name__ == "__main__":
    compare_muon_implementations()
