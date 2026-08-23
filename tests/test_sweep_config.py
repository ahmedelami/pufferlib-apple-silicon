import configparser
from pathlib import Path
import sys

import pytest

from pufferlib import pufferl
from pufferlib.sweep import Hyperparameters


ROOT = Path(__file__).resolve().parents[1]


def _native_config_env_names():
    env_names = set()
    for binding in (ROOT / "ocean").glob("*/binding.c"):
        config_name = binding.parent.name
        if config_name == "squared_continuous":
            config_name = "squared"
        env_path = ROOT / "config" / f"{config_name}.ini"
        if not env_path.exists():
            continue
        parser = configparser.ConfigParser()
        parser.read(env_path)
        env_names.update(parser["base"]["env_name"].split())
    return sorted(env_names)


@pytest.mark.parametrize("env_name", ["default", *_native_config_env_names()])
def test_shipped_sweep_config_parses(monkeypatch, env_name):
    monkeypatch.setattr(sys, "argv", ["puffer"])
    args = pufferl.load_config(env_name)
    hypers = Hyperparameters(args["sweep"], verbose=False)

    assert hypers.num > 0
    assert not any(name.startswith("match_") for name in hypers.flat_spaces)


def test_impulse_wars_sweep_only_varies_effective_parameters(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["puffer"])
    args = pufferl.load_config("impulse_wars")
    hypers = Hyperparameters(args["sweep"], verbose=False)

    assert "env/reward_enemy_kill" in hypers.flat_spaces
    assert "env/reward_kill" not in hypers.flat_spaces
    assert "env/num_envs" not in hypers.flat_spaces
    assert "train/batch_size" not in hypers.flat_spaces
    assert args["sweep"]["max_suggestion_cost"] == 900


def test_mps_is_one_automatic_sweep_device(monkeypatch):
    monkeypatch.setattr(pufferl.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(pufferl.torch.backends.mps, "is_available", lambda: True)

    assert pufferl._sweep_device_count({"gpus": 0}) == 1
    assert pufferl._sweep_device_count({"gpus": 3}) == 3


def test_negative_sweep_device_count_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        pufferl._sweep_device_count({"gpus": -1})
