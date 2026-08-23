"""CPU-only tests for portable Torch distributed launch routing.

CUDA, multiprocessing, and process-group operations are mocked so these tests
exercise launch semantics on developer and CI machines without accelerators.
"""

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from pufferlib import pufferl as launch


def _args(*, gpus=3, total_timesteps=240):
    return {
        "env_name": "routing_test",
        "slowly": True,
        "train": {
            "gpus": gpus,
            "horizon": 2,
            "minibatch_size": 8,
            "total_timesteps": total_timesteps,
        },
        "vec": {"total_agents": 4},
    }


def _portable_backend():
    try:
        from pufferlib import _C
    except Exception as exc:
        pytest.skip(f"pufferlib._C is not built: {exc}")
    if getattr(_C, "precision_bytes", None) != 4:
        pytest.skip("the portable Torch trainer requires a float32 native build")

    from pufferlib import torch_pufferl

    return torch_pufferl


def test_torchrun_context_requires_and_parses_all_coordinates():
    assert launch._torchrun_context({}) is None
    assert launch._torchrun_context({
        "RANK": "2", "WORLD_SIZE": "8",
    }) is None
    assert launch._torchrun_context({
        "RANK": "2", "WORLD_SIZE": "8", "LOCAL_RANK": "3",
    }) == (2, 8, 3)

    with pytest.raises(ValueError):
        launch._torchrun_context({
            "RANK": "not-an-integer",
            "WORLD_SIZE": "8",
            "LOCAL_RANK": "3",
        })


def test_internal_torch_workers_share_one_rendezvous_and_skip_native_nccl(
        monkeypatch):
    calls = []
    rendezvous_calls = []

    def forbidden_nccl_id():
        raise AssertionError("portable Torch launch requested a native NCCL id")

    monkeypatch.setattr(launch, "_C", SimpleNamespace(
        create_pufferl=object(),
        get_nccl_id=forbidden_nccl_id,
    ))

    def reserve_rendezvous():
        rendezvous_calls.append(True)
        return "tcp://127.0.0.1:43210"

    monkeypatch.setattr(launch, "_local_torch_rendezvous", reserve_rendezvous)

    def local_train(env_name, worker_args, **kwargs):
        calls.append(("local", env_name, worker_args, kwargs))

    monkeypatch.setattr(launch, "_train", local_train)

    class FakeProcess:
        def __init__(self, target, args, kwargs):
            self.target = target
            self.args = args
            self.kwargs = kwargs
            self.exitcode = None

        def start(self):
            env_name, worker_args = self.args
            calls.append(("spawn", env_name, worker_args, self.kwargs))
            self.exitcode = 0

        def is_alive(self):
            return self.exitcode is None

        def terminate(self):
            self.exitcode = -15

        def join(self):
            pass

    class FakeContext:
        @staticmethod
        def Process(*, target, args, kwargs):
            assert target is local_train
            return FakeProcess(target, args, kwargs)

    def get_context(method):
        assert method == "spawn"
        return FakeContext()

    monkeypatch.setattr(launch.mp, "get_context", get_context)

    launch.train("routing_test", args=_args())

    assert rendezvous_calls == [True]
    assert len(calls) == 3
    workers = {worker_args["rank"]: (kind, worker_args, kwargs)
        for kind, _, worker_args, kwargs in calls}
    assert set(workers) == {0, 1, 2}
    assert all(worker[0] == "spawn" for worker in workers.values())
    assert workers[0][2] == {"verbose": True, "result_queue": None}
    assert workers[1][2] == workers[2][2] == {
        "verbose": False, "result_queue": None}
    assert {workers[rank][1]["gpu_id"] for rank in workers} == {0, 1, 2}

    for _, worker_args, _ in workers.values():
        assert worker_args["world_size"] == 3
        assert worker_args["train"]["total_timesteps"] == 80
        assert worker_args["torch_dist_init_method"] == \
            "tcp://127.0.0.1:43210"
        assert worker_args["nccl_id"] == b""


def test_external_torchrun_invokes_exactly_one_local_worker_without_spawning(
        monkeypatch):
    for name, value in {
        "RANK": "2",
        "WORLD_SIZE": "4",
        "LOCAL_RANK": "1",
    }.items():
        monkeypatch.setenv(name, value)

    def forbidden_nccl_id():
        raise AssertionError("torchrun requested a native NCCL id")

    monkeypatch.setattr(launch, "_C", SimpleNamespace(
        create_pufferl=object(),
        get_nccl_id=forbidden_nccl_id,
    ))

    calls = []

    def local_train(env_name, worker_args, **kwargs):
        calls.append((env_name, worker_args, kwargs))

    monkeypatch.setattr(launch, "_train", local_train)

    def forbidden_spawn(*args, **kwargs):
        raise AssertionError("external torchrun attempted nested multiprocessing")

    monkeypatch.setattr(launch.mp, "get_context", forbidden_spawn)

    args = _args(gpus=99, total_timesteps=400)
    launch.train("routing_test", args=args)

    assert len(calls) == 1
    env_name, worker_args, kwargs = calls[0]
    assert env_name == "routing_test"
    assert kwargs == {"verbose": False}
    assert worker_args["rank"] == 2
    assert worker_args["world_size"] == 4
    assert worker_args["gpu_id"] == 1
    assert worker_args["train"]["gpus"] == 4
    assert worker_args["train"]["total_timesteps"] == 100
    assert worker_args["nccl_id"] == b""
    assert "torch_dist_init_method" not in worker_args


def test_create_pufferl_selects_index_before_vec_and_wraps_ddp_after_trainer(
        monkeypatch):
    backend = _portable_backend()
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)

    events = []

    class FakeVec:
        gpu = True
        total_agents = 64

        def close(self):
            events.append(("vec_close",))

    vec = FakeVec()

    def create_vec(args, native_gpu):
        events.append(("create_vec", args["gpu_id"], native_gpu))
        return vec

    monkeypatch.setattr(backend, "_C", SimpleNamespace(
        gpu=1,
        create_vec=create_vec,
    ))
    monkeypatch.setattr(
        backend,
        "resolve_device",
        lambda requested, native_cuda: torch.device("cuda"),
    )

    def resolve_rollout_device(requested, device, **kwargs):
        events.append(("resolve_rollout", device, kwargs))
        return device

    monkeypatch.setattr(backend, "resolve_rollout_device", resolve_rollout_device)
    monkeypatch.setattr(
        torch.cuda,
        "set_device",
        lambda index: events.append(("set_device", index)),
    )

    class FakePolicy:
        hidden_size = 37

        @staticmethod
        def forward_eval(*args, **kwargs):
            return None

        @staticmethod
        def initial_state(*args, **kwargs):
            return None

    learner = FakePolicy()

    def load_policy(args, loaded_vec, device):
        events.append(("load_policy", loaded_vec, device))
        return learner

    monkeypatch.setattr(backend, "load_policy", load_policy)

    def trainer_init(self, args, loaded_vec, policy, *, device, rollout_device):
        events.append(("trainer_init", loaded_vec, policy, device, rollout_device))
        self.policy = policy
        self.rollout_policy = policy
        self._owns_process_group = False

    monkeypatch.setattr(backend.PuffeRL, "__init__", trainer_init)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    def init_process_group(**kwargs):
        events.append(("init_process_group", kwargs))

    monkeypatch.setattr(
        torch.distributed, "init_process_group", init_process_group)

    class FakeDDP:
        def __init__(self, module, device_ids, output_device):
            events.append(("ddp", module, device_ids, output_device))
            self.module = module

    monkeypatch.setattr(
        torch.nn.parallel, "DistributedDataParallel", FakeDDP)

    args = {
        "rank": 2,
        "world_size": 4,
        "gpu_id": 5,
        "vec": {},
        "torch": {"device": "cuda", "rollout_device": "auto"},
        "torch_dist_init_method": "tcp://127.0.0.1:43210",
    }

    trainer = backend.PuffeRL.create_pufferl(args)

    names = [event[0] for event in events]
    assert names.index("set_device") < names.index("create_vec")
    assert names.index("trainer_init") < names.index("init_process_group")
    assert names.index("init_process_group") < names.index("ddp")
    assert ("set_device", 5) in events
    assert ("create_vec", 5, 1) in events
    assert args["vec"]["num_buffers"] == 1
    assert args["rank"] == 2
    assert args["world_size"] == 4
    assert args["gpu_id"] == 5

    load_event = next(event for event in events if event[0] == "load_policy")
    assert load_event[1] is vec
    assert load_event[2] == torch.device("cuda", 5)
    trainer_event = next(event for event in events if event[0] == "trainer_init")
    assert trainer_event[3:] == (
        torch.device("cuda", 5), torch.device("cuda", 5))
    init_event = next(
        event for event in events if event[0] == "init_process_group")
    assert init_event[1] == {
        "backend": "nccl",
        "rank": 2,
        "world_size": 4,
        "init_method": "tcp://127.0.0.1:43210",
    }
    ddp_event = next(event for event in events if event[0] == "ddp")
    assert ddp_event[1] is learner
    assert ddp_event[2:] == ([5], 5)
    assert isinstance(trainer.policy, FakeDDP)
    assert trainer.rollout_policy is trainer.policy
    assert trainer.policy.hidden_size == learner.hidden_size
    assert trainer._owns_process_group is True


def test_supervisor_terminates_peer_ranks_and_propagates_failure(monkeypatch):
    processes = []

    class FakeProcess:
        def __init__(self, target, args, kwargs):
            self.args = args
            self.exitcode = None
            processes.append(self)

        @property
        def rank(self):
            return self.args[1]["rank"]

        def start(self):
            if self.rank == 1:
                self.exitcode = 7

        def is_alive(self):
            return self.exitcode is None

        def terminate(self):
            self.exitcode = -15

        def join(self):
            pass

    class FakeContext:
        @staticmethod
        def Process(*, target, args, kwargs):
            return FakeProcess(target, args, kwargs)

    monkeypatch.setattr(launch.mp, "get_context", lambda method: FakeContext())
    monkeypatch.setattr(launch.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="exit codes"):
        launch._supervise_train_group(
            "routing_test", _args(), [0, 1, 2], {}, False)

    assert {process.rank: process.exitcode for process in processes} == {
        0: -15,
        1: 7,
        2: -15,
    }


def test_distributed_stop_decision_is_broadcast_or_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def broadcast(flag, src):
        calls.append((flag.device.type, src))
        flag.fill_(1)

    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    trainer = SimpleNamespace(device=torch.device("cpu"))
    assert launch._sync_stop_decision(trainer, False, 2) is True
    assert calls == [("cpu", 0)]

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    assert launch._sync_stop_decision(trainer, True, 2) is False
    assert launch._sync_stop_decision(trainer, True, 1) is True


def test_direct_setup_failure_is_not_silently_swallowed(monkeypatch, tmp_path):
    class BrokenBackend:
        @staticmethod
        def create_pufferl(_args):
            raise RuntimeError("simulated setup failure")

    monkeypatch.setattr(launch, "_resolve_backend", lambda _args: BrokenBackend)
    args = {
        "rank": 0,
        "world_size": 1,
        "gpu_id": 0,
        "wandb": False,
        "sweep": {"metric": "score"},
        "train": {"total_timesteps": 1},
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "log_dir": str(tmp_path / "logs"),
        "env_name": "routing_test",
    }
    with pytest.raises(RuntimeError, match="simulated setup failure"):
        launch._train("routing_test", dict(args))

    queued = []
    result_queue = SimpleNamespace(put=queued.append)
    launch._train("routing_test", dict(args), result_queue=result_queue)
    assert queued == [(0, [], [], [])]


def test_train_worker_closes_trainer_when_rollout_raises(monkeypatch):
    trainer = SimpleNamespace(global_step=0)
    closed = []

    class BrokenBackend:
        @staticmethod
        def create_pufferl(_args):
            return trainer

        @staticmethod
        def rollouts(_trainer):
            raise ValueError("simulated rollout failure")

        @staticmethod
        def train(_trainer):
            raise AssertionError("train should not be reached")

        @staticmethod
        def close(value):
            closed.append(value)

    monkeypatch.setattr(launch, "_resolve_backend", lambda _args: BrokenBackend)
    with pytest.raises(ValueError, match="simulated rollout failure"):
        launch._train_worker({"train": {"total_timesteps": 1}})
    assert closed == [trainer]


@pytest.mark.parametrize("amp_dtype", ["float16", "fp16"])
def test_fp16_fails_closed_before_device_or_vec_work(amp_dtype):
    backend = _portable_backend()
    args = {
        "train": {},
        "torch": {"amp_dtype": amp_dtype},
    }

    with pytest.raises(ValueError, match="unsupported without gradient scaling"):
        backend.PuffeRL(
            args,
            vec=SimpleNamespace(gpu=True),
            policy=None,
            device=torch.device("cuda", 0),
            rollout_device=torch.device("cuda", 0),
        )
