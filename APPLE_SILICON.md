# Apple Silicon backend

PufferLib 4.0 now has an Apple Silicon execution path designed around the
machine's unified-memory CPU/GPU architecture:

- Ocean simulators compile natively for ARM64 with Apple CPU tuning and
  OpenMP. The build no longer emits x86 AVX flags on ARM.
- Policy training runs on the Apple GPU through PyTorch MPS.
- Rollout placement is chosen by batch size when the environment has a measured
  threshold. For Breakout with DefaultEncoder + MinGRU and horizon 64, the M5
  Pro crossover is 4,096 agents: direct MPS rollouts at or above the threshold,
  or a CPU actor plus one bulk horizon transfer below it. Uncalibrated
  environment/model pairs conservatively use the hybrid path.
- Puffer/V-trace advantage calculation uses one fused Metal shader dispatch
  instead of a Python launch per timestep.
- On the measured PyTorch 2.13 runtime, direct MPS uses the private
  `torch.mps._host_alias_storage` interop primitive to let the CPU environment
  read actions and write observations through the same shared Metal buffers.
  Every host access is explicitly synchronized on both sides. If the private
  API is absent, rejects an allocation, or is disabled with
  `--torch.mps-host-alias off`, the trainer automatically uses ordinary
  synchronized staged copies.
- Breakout's exact validated FP32 shape can compile its rollout and learner
  policy methods with MPS Inductor. This is deliberately a machine-specific
  promotion, not a broad `torch.compile` claim. The complete identity guard is
  PyTorch 2.13.0 at git revision
  `cf30153c4c131c8164ee7798e5022d810682e2cb`, Darwin ARM64, Mac17,8, Apple M5
  Pro with 20 GPU cores and 24 GiB unified memory, and macOS 27.0 build
  26A5378j. The workload guard pins Breakout, 4,096 agents, 118 observations,
  one categorical action head of size 3, horizon 64, minibatch 65,536, the
  2x64 MinGRU policy with 32,452 parameters, direct FP32 MPS, a CPU vector
  backend, one distributed rank, active host aliasing, and
  `PYTORCH_ENABLE_MPS_FALLBACK=0`.
- The compiler also pins the validated policy classes and their unmodified
  class-level `forward` and `forward_eval` methods. It rejects instance method
  overrides, disabled or error-suppressing Dynamo state, incompatible layout
  forcing (`TORCHINDUCTOR_FORCE_LAYOUT_OPT=1`), a compiler-bisector backend
  override, and any wrapper that is not a real Dynamo wrapper around the exact
  original method. Each compile is full-graph and static-shape with per-call
  `layout_optimization=False`. Before installing either compiled callable, it
  runs a synchronized finite forward/evaluation/backward preflight; both
  wrappers must pass together. `auto` remains eager on any identity, setup,
  wrapper, or preflight mismatch, while explicit `inductor` fails closed.
  Full guard, compilation, sampler, and preflight startup time is retained in
  the run provenance and counted by the learning holdout.
- The promoted compiled rollout uses a persistent fused categorical sampler.
  Its Metal kernel implements the same Philox exponential-race algorithm as
  `torch.multinomial`, while a small version-gated C++ bridge reserves the
  default MPS generator's Philox range atomically under the generator mutex.
  Startup performs an exact explicitly seeded dispatch preflight without
  changing the process RNG state. `auto` falls back to `torch.multinomial` if
  bridge construction or sampler preflight fails; explicit `inductor` fails
  closed. The fused sampler is limited to the validated compiled Breakout
  rollout. Eager, other discrete or multidiscrete shapes, Normal policies, and
  supplied-action evaluation retain their existing sampling paths.
- Persistent rollout storage stays contiguous and time-major before exposing an
  agent-major strided view. Direct MPS therefore does not duplicate the full
  horizon, and hybrid mode avoids a redundant source-device contiguous copy;
  only selected minibatches are materialized agent-major.
- Policy math can use BF16 autocast, while PPO statistics, advantages, losses,
  and optimizer state stay in FP32. FP32 remains the default because it is the
  numerical-parity baseline. BF16 has not yet passed a time-to-score or
  convergence comparison and should be treated as an experimental throughput
  option.
- Craftax keeps its compact 843-value native transport, but the policy expands
  it algebraically to the exact 8,268-value symbolic one-hot projection. Its
  canonical Linear parameter and Muon update geometry match the pre-port
  8,268-input DefaultEncoder, so those checkpoints load strictly with unchanged
  keys and weights. A short-lived 843-input Linear checkpoint is lifted exactly
  into symbolic feature space at load time. Overlapping mob types are
  transported losslessly.
- Impulse Wars now uses an ABI-aware CNN/categorical/float encoder. Historical
  1,000-byte DefaultEncoder checkpoints cannot be mapped mathematically into
  that architecture. The strict loader detects them, emits a warning, and
  retains the old raw-byte encoder exactly; start a new run without an old
  checkpoint to use ImpulseWarsEncoder.
- Matsci uses an explicit native ballistic backend by default. It implements
  the same zero-force, 0.5-timestep integration and periodic 20-unit box as the
  old LAMMPS setup without carrying LAMMPS through every vector step. Merely
  installing LAMMPS no longer changes environment behavior.

The native Craftax vector ABI therefore reports `obs_size == 843`; it is an
explicit packed transport, not the public `Craftax-Symbolic-v1` 8,268-value
shape. External consumers that need the canonical JAX-compatible tensor can
call `CraftaxEncoder.expand_observation(observations)`. The trainer selects that
encoder automatically, so it never treats packed categorical IDs as ordinal
features.

## Setup

```bash
brew install llvm libomp
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python -e '.[test]'
./build.sh breakout --mps
```

`--mps` builds the native CPU simulator extension and, on the validated Torch
runtime, atomically builds the optional MPS RNG bridge used by the fused
sampler. If that private ABI is unavailable, the build warns and leaves the
portable eager sampler available. The trainer detects that the monolithic CUDA
backend is absent and selects the Torch/MPS backend automatically:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 puffer train breakout
```

The fallback variable must be present when Python starts. Without it, the
portable eager MPS backend still runs, but the guarded Breakout compiler stays
off because that runtime combination was not used for the acceptance evidence.

Matsci's LAMMPS reference backend is opt-in and fails closed if its headers or
library cannot be linked. `pkg-config lammps` or `pkg-config liblammps` is
preferred; otherwise set
`LAMMPS_INCLUDE_DIR` to the directory containing `lammps/library.h` and
`LAMMPS_LIB_DIR` to the directory containing `liblammps`:

```bash
./build.sh matsci --mps             # optimized native backend (default)
./build.sh matsci --mps --lammps    # explicit LAMMPS reference backend
```

Useful overrides are:

```bash
# Force the hybrid CPU-actor/MPS-learner path
puffer train breakout --torch.rollout-device cpu

# Force direct MPS rollout inference
puffer train breakout --torch.rollout-device mps

# Disable the version-dependent shared-buffer alias and use staged copies
puffer train breakout --torch.mps-host-alias off

# Disable the guarded Breakout policy compiler and use eager MPS
puffer train breakout --torch.compile-policy off

# Require the validated compiler shape or fail with every mismatched guard
PYTORCH_ENABLE_MPS_FALLBACK=0 puffer train breakout \
  --torch.compile-policy inductor

# Opt into BF16 policy autocast
puffer train breakout --torch.amp-dtype bfloat16

# Deliberate CPU-only reference
puffer train breakout --torch.device cpu --torch.rollout-device cpu
```

## Validation

Build and smoke every environment binding in an isolated subprocess, run the
Craftax trajectory oracle against its matching extension, then leave a
representative Breakout extension active for the shared suite:

```bash
PATH="$PWD/.venv/bin:$PATH" PYTORCH_ENABLE_MPS_FALLBACK=0 \
  .venv/bin/python tests/native_binding_smoke.py \
  --build-all --build-mode mps \
  --json-output work/native_smoke_matrix_final.json
./build.sh craftax --mps
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python tests/craftax_parity.py \
  --seeds 16 --seed-start 0 --steps 2000 --action-seed 0 \
  --policy uniform --num-threads 18
./build.sh breakout --mps
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest -q -rs
```

`scape` and `tactical` remain buildable as upstream interactive prototypes, but
are intentionally rejected by the training vector API. Their observations are
empty and their step functions ignore policy actions; Tactical's original
maintainer commit also records that it was incomplete and did not run through
the Python demo. Treating their constant trajectories as successful hardware
ports would be misleading.

The final M5 Pro validation built and imported all 61 native bindings. All 58
constructible environments passed reset, one complete zero-action step,
ABI/shape/dtype/finiteness/log checks, and clean close; Scape and Tactical were
the two expected fail-closed prototypes, and Drive was the sole fixture skip
because its generated map binary is not checked in. The final Breakout-backed
shared collection reported 214 passed, 25 explicitly skipped, zero failures,
and no collection errors. Extension-specific suites were also run immediately
after their matching native builds, including Craftax's 16-seed x 2,000-step
JAX trajectory parity test.

CUDA-only and optional research tests are explicitly reported as skipped when
their hardware, toolkit, or datasets are unavailable. They are not silently
treated as Apple/MPS tests.

MPS correctness coverage includes fused-advantage parity against the native CPU
kernel, allocator-churned matrix multiplication and host-alias coherence,
bitwise alias-versus-staged epoch equivalence, policy forward/backward and
gradient parity, direct MPS epochs, and hybrid epochs. The compiled policy gate
also covers allocator-churned fixed inputs, exact seeded actions and RNG state,
raw and clipped gradients, Muon parameter/momentum updates, a real compiled
epoch, and the isolated learning-quality holdout below. The fused sampler was
validated in 48 exact cases spanning the production 4,096-by-3 shape, varied
widths, exceptional logits, and seeds on both sides of signed 64-bit storage;
actions, gathered log probabilities, and final MPS RNG state matched
`torch.multinomial` bit-for-bit. Its persistent buffers showed flat current and
driver allocation across 1,024 dispatches. All reported validation and
benchmark commands set `PYTORCH_ENABLE_MPS_FALLBACK=0`; the private-alias and
sampler fallbacks above are separate from PyTorch's unsupported-operation CPU
fallback.

Craftax's parity contract is exact symbolic observations, rewards, terminals,
seeded resets, and policy/optimizer behavior. Its optimized state keeps the
light map quantized to bytes, so it is intentionally not a byte-for-byte copy
of JAX's float32 `EnvState`; reachable visibility behavior is covered by the
trajectory parity test. Any external native-state importer must also rebuild
the derived terrain and mob bitmaps, as the provided test fixture does.

## Benchmarking

The benchmark excludes CLI rendering, final evaluation epochs, checkpoints,
and experiment tracking. It warms up each backend, synchronizes device work,
checks losses and parameters for finite values, and reports median plus range.
Each JSON result includes every raw epoch sample, the effective Torch/train/
vector/environment configuration, alias/fallback/compiler state, hardware and
OS identity, extension precision, Git revision, and a source-patch SHA-256
fingerprint. Compiler and fused-sampler startup are excluded from the
steady-state table but included in the isolated learning holdout:

```bash
./build.sh breakout --mps
PYTORCH_ENABLE_MPS_FALLBACK=0 python benchmarks/apple_silicon.py \
  --env breakout \
  --agents 4096 \
  --horizon 64 \
  --minibatch-size 65536 \
  --threads 18 \
  --torch-threads 12 \
  --torch-interop-threads 1 \
  --warmup-epochs 3 \
  --epochs 11 \
  --compile-policy auto \
  --modes mps

# Reproduce the tuned CPU row from the 1/6/12/18 intra-op thread sweep
PYTORCH_ENABLE_MPS_FALLBACK=0 python benchmarks/apple_silicon.py \
  --env breakout \
  --agents 4096 \
  --horizon 64 \
  --minibatch-size 65536 \
  --threads 18 \
  --torch-threads 12 \
  --torch-interop-threads 18 \
  --warmup-epochs 3 \
  --epochs 11 \
  --modes cpu

# Reproduce the CPU-actor/MPS-learner row in a separate process
PYTORCH_ENABLE_MPS_FALLBACK=0 python benchmarks/apple_silicon.py \
  --env breakout \
  --agents 4096 \
  --horizon 64 \
  --minibatch-size 65536 \
  --threads 18 \
  --torch-threads 18 \
  --torch-interop-threads 18 \
  --warmup-epochs 3 \
  --epochs 11 \
  --modes hybrid

# Reproduce the BF16 row separately (hybrid BF16 is intentionally rejected)
PYTORCH_ENABLE_MPS_FALLBACK=0 python benchmarks/apple_silicon.py \
  --env breakout \
  --agents 4096 \
  --horizon 64 \
  --minibatch-size 65536 \
  --threads 18 \
  --torch-threads 18 \
  --torch-interop-threads 18 \
  --warmup-epochs 3 \
  --epochs 11 \
  --amp-dtype bfloat16 \
  --modes mps
```

## Measured M5 Pro result

On this 20-GPU-core/18-CPU-core M5 Pro, all rows use 4,096 Breakout agents,
horizon 64, minibatch 65,536, 18 native environment threads, and three warmup
epochs followed by 11 measured epochs. Each row reports its measured Torch
intra-op thread count; the compiled row uses one inter-op thread and the older
reference rows used 18:

| Mode | Precision | Torch threads | Epochs | Median agent steps/s | Measured range |
|---|---:|---:|---:|---:|---:|
| CPU rollout + CPU learner | FP32 | 12 | 11 | 456,786 | 431,849–462,090 |
| CPU rollout + MPS learner | FP32 | 18 | 11 | 846,656 | 825,547–858,530 |
| MPS rollout + MPS learner (eager) | FP32 | 18 | 11 | 1,232,524 | 1,209,814–1,247,203 |
| MPS rollout + MPS learner (compiled + fused sampler, primary) | FP32 | 12 | 11 | **2,094,181** | **1,937,189–2,129,181** |
| MPS rollout + MPS learner (compiled + fused sampler, repeat) | FP32 | 12 | 11 | **2,091,980** | **1,957,241–2,182,506** |
| MPS rollout + MPS learner | BF16 autocast | 18 | 11 | 1,253,506 | 1,182,212–1,287,415 |

An immediate independent repeat of the eager direct FP32 row measured 1,230,262
steps/s, 0.18% below its table median. The two independent production
compiled-plus-fused medians give a representative result of approximately
2.093M steps/s: 4.58x the tuned 456,786-step/s CPU baseline, 2.47x the
846,656-step/s hybrid path, and 1.70x the earlier 1,232,524-step/s eager MPS
row for this workload. BF16 remains experimental because it has not passed a
time-to-score comparison and is deliberately excluded from the compiler guard.
These are local end-to-end training-loop measurements, not a CUDA comparison.

The sampler itself was promoted through a controlled alternating ABBA test
against `torch.multinomial`. The full epoch digest was exact across rollout
tensors and state, policy and optimizer state, losses, environment logs, and
MPS, CPU, NumPy, and Python RNG state. The baseline measured 2,097,824 steps/s
and the fused sampler measured 2,193,237 steps/s, a 4.55% increase. Median
rollout time fell from 45.372 ms to 40.019 ms and total epoch time from 124.960
ms to 119.524 ms; learner time was unchanged. Those paired numbers isolate the
sampler effect. The two production rows above, rather than the higher A/B
sample, are the canonical absolute-throughput evidence.

Two post-validation sanity repeats run under substantial concurrent Screen
Sharing/WindowServer load measured 1,158,809 and 1,165,024 steps/s. They are
retained as loaded-system evidence but excluded from the matched table above;
benchmark comparisons should control foreground and background GPU/CPU use.

### Learning quality and cold compiler/sampler startup

Throughput was promoted only after two paired learning gates. First, five fixed
seeds compared eager FP32 MPS with the deterministic CPU reference for 32
epochs (8,388,608 steps) each. MPS/CPU median tail score was 1.012, normalized
learning-curve AUC was 0.995, both 90% bootstrap lower guards exceeded 0.87,
both sustained score thresholds were reached by 5/5 runs at the same median
environment step, and MPS completed the runs 2.79x faster.

The compiler-plus-fused-sampler candidate then used ten unseen fixed seeds
(`73,79,83,89,97,101,103,107,109,113`) against eager MPS. Every replicate ran
in a fresh process, order alternated eager/compiled, and each compiled process
received a unique empty Inductor cache. Compiler and sampler startup were
included; discovery seeds were not pooled or selectively rerun. The final
report also verifies the requested and effective compiler and sampler modes,
the real Dynamo wrappers, the synchronized graph preflight, active host
aliasing, fallback-disabled runtime, and alternating order. Its top-level
metadata records the exact host/runtime identity; the runner itself creates a
fresh temporary Inductor cache for each compiled child. All predeclared checks
passed:

| Compiled / eager metric | Result | Required |
|---|---:|---:|
| Median tail score | 0.9689097 | >= 0.90 |
| Tail-score 90% bootstrap lower guard | 0.9375442 | >= 0.75 |
| Median learning-curve AUC | 1.0104565 | >= 0.90 |
| AUC 90% bootstrap lower guard | 0.9876438 | >= 0.75 |
| Median full-run wall speedup, cold startup included | 1.2468183x | >= 1.0x |
| Sustained score-2 step ratio | 1.0000000 | <= 1.10 |
| Sustained score-4 step ratio | 1.0000000 | <= 1.10 |

Both thresholds were reached in 10/10 eager and 10/10 compiled runs, and every
loss remained finite. The eager-time/compiled-time ratios at the early
thresholds were 0.6612152 for score 2 and 0.7885239 for score 4: both paths
needed the same environment steps, but cold compiler and sampler startup made
the candidate slower to those early milestones. Across the complete 32-epoch
run, eager-time/compiled-time was 1.2468183, so the candidate was already
faster overall. Seed 107 had a 0.633 compiled/eager tail-score ratio, so the
conclusion rests on the preregistered paired distribution and strong bootstrap
guards rather than pretending every individual trajectory is identical.
Reproduce the isolated protocol with:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python \
  benchmarks/compile_policy_holdout.py
```

PyTorch MPS does not provide a deterministic `scatter_reduce` backward for this
model. The MPS arms therefore use fixed RNG seeds plus paired multi-seed
statistics rather than falsely claiming bitwise deterministic training.

### Craftax rollout sizing and memory

Craftax's horizon storage is large enough that layout and batch sizing matter:
16,384 agents x 128 steps x 843 float32 values is 6.586 GiB. The old
agent-major conversion duplicated that allocation. The direct-MPS path now
keeps the time-major storage and gathers only a 105.375 MiB minibatch.

At 4,096 agents, a controlled layout probe measured no full-horizon allocation
for the strided view versus an extra 1.646 GiB for the old contiguous
transpose. Transform plus all 16 minibatch gathers took 87.5 ms median versus
141.5 ms. In the end-to-end Craftax trainer, the new path measured 206,695
steps/s versus 200,717 and reduced median train time from 801.0 to 736.8 ms.

The M5 Pro sizing sweep used FP32 direct MPS, exact seeded resets, horizon 128,
minibatch 32,768, and 18 native threads. It selected 8,192 agents as the
hardware default:

| Agents | Median agent steps/s | Measured range | Driver allocation after run |
|---:|---:|---:|---:|
| 4,096 | 206,695 | 195,215–214,076 | 2.71 GiB |
| 8,192 | **249,989** | **245,319–252,133** | 4.39 GiB |
| 12,288 | 219,123 | 206,395–250,002 | 7.06 GiB |
| 16,384 | 192,361 | 189,234–226,433 | 8.68 GiB |

The 16,384 row used three measured epochs after one warmup because of its much
longer runtime; the other rows used five measured epochs after two warmups.
After all sampler, shared-buffer coherence, and optimizer changes were frozen,
the selected 8,192 row was remeasured and updated in the table; the other batch
sizes and all driver-allocation values are from the original sizing sweep.
Driver allocation includes MPS allocator caches after trainer teardown, so it
is an observed high-water proxy rather than an instantaneous peak. The Mac's
Metal recommended maximum was 17.76 GiB. The 8,192 default retains substantial
headroom for the native Craftax state, the OS, and other applications while
also giving the best sustained result in this sweep.

Raw throughput from an NVIDIA GPU is not an apples-to-apples acceptance target
for a Mac that cannot run CUDA. Compare CUDA and MPS only with the same commit,
environment, model, precision, agent count, horizon, minibatch, replay ratio,
seed, warmup policy, and timing boundaries. Also check convergence or
time-to-score: steps per second alone does not establish training equivalence.
No NVIDIA device was available for this port, so CUDA parity is deliberately
left unclaimed until that controlled run is performed on a named CUDA host.
The harness supports that portable-path run directly: build the float32 CUDA
extension with `./build.sh breakout --float`, then use the same command above
with `--modes cuda`. The repository's monolithic native-CUDA backend should be
reported separately because it has different rollout/storage machinery. The
portable launch path now resolves indexed CUDA devices before native vector
allocation and supports both internal multi-GPU spawning and external
`torchrun`; mocked routing/DDP tests pass here, but a real two-GPU epoch remains
part of the named-NVIDIA acceptance run.
