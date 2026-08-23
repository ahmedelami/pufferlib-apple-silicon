![figure](https://pufferai.github.io/source/resource/header.png)

[![Discord](https://dcbadge.vercel.app/api/server/spT4huaGYV?style=plastic)](https://discord.gg/spT4huaGYV)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40jsuarez)](https://twitter.com/jsuarez)

PufferLib is a fast and sane reinforcement learning library that can train tiny, super-human models in seconds. The included learning algorithm, hyperparameter tuning, and simulation methods are the product of our own research. All our tools are free and open source. Need a high performance environment for your application? We build them professionally and offer training + extended support. Contact jsuarez🐡puffer🐡ai.

All of our documentation is hosted at [puffer.ai](https://puffer.ai "PufferLib Documentation"). @jsuarez5341 on [Discord](https://discord.gg/puffer) for support. Post there before opening issues. We're always looking for new contributors!

## Apple Silicon (Metal/MPS)

The 4.0 trainer can use the Apple GPU through PyTorch MPS while Ocean
environments run in the native ARM64 vector backend. On an Apple Silicon Mac:

```bash
brew install llvm libomp
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python -e '.[test]'
./build.sh breakout --mps
PYTORCH_ENABLE_MPS_FALLBACK=0 puffer train breakout
```

Device selection is automatic. Calibrated environment/model pairs can keep
large rollout batches on MPS; uncalibrated pairs use a CPU actor and transfer
each completed horizon to MPS once. Use `--torch.rollout-device cpu` or
`--torch.rollout-device mps` to override it. Float32 is the correctness
default. BF16 is an explicit experimental opt-in: its execution-contract and
semantic gates pass, but its preregistered 10-seed promotion gate narrowly
failed, so it is not selected automatically.

Run the documented 4,096-agent direct-MPS workload with:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python benchmarks/apple_silicon.py \
  --env breakout --agents 4096 --horizon 64 --minibatch-size 65536 \
  --threads 18 --torch-threads 12 --torch-interop-threads 1 \
  --warmup-epochs 3 --epochs 11 --compile-policy auto \
  --compile-ppo off --mingru-train-scan off --modes mps
```

The quality-safe Breakout default is the guarded compiled FP32 policy with its
fused Metal/Philox rollout sampler. On the target 20-GPU-core M5 Pro, two
production runs measured 2,094,181 and 2,091,980 agent steps/s. The
representative ~2.093M result is 4.58x the tuned CPU path, 2.47x the
CPU-rollout/MPS-learner path, and 1.70x eager MPS on the documented
4,096-agent workload. A provenance-checked 10-seed learning holdout passed its
predeclared quality and full-run wall-time gates against eager MPS.

A faster experimental stack adds a compiled supplied-action PPO graph and a
training-only Metal MinGRU scan. It reached a 2.207M-step/s median over the
complete 93,847,552-step workload versus 1.719M for its quality-safe baseline;
the median paired speed ratio was 1.2904x. It failed the preregistered
learning-quality gate: tail score was 0.9866x its baseline, but its bootstrap
lower guard was only 0.6509; learning AUC was 0.6751x (0.5351 lower guard), and
score-4 took 1.0769x as many steps. All runs remained finite and passed identity
and no-recompile checks. Breakout therefore keeps `torch.compile_ppo=off` and
`torch.mingru_train_scan=off`; nonmatching or failed explicit requests fail
closed, and `auto` falls back to the promoted compiled-policy path.

There is online CUDA context, but not a controlled parity result. PufferLib's
[official documentation](https://puffer.ai/docs.html) reports a 20M-step/s
native headline and a 3–5 second Breakout run on an RTX 5090. That headline is
about 11.6x the 93,847,552-step MPS baseline or 9.6x the short
~2.093M MPS result. The official
[experiments release](https://github.com/PufferAI/PufferLib/releases/tag/experiments)
contains raw Breakout runs, but it does not identify each run's GPU, commit,
precision, or equivalent timing boundary. Native CUDA and Torch/MPS also use
different machinery, so CUDA is clearly faster in the published results while
a same-commit Torch-CUDA-versus-MPS ratio remains unproven.

See [APPLE_SILICON.md](APPLE_SILICON.md) for architecture, validation, and
benchmark guidance.

## Star to puff up the project!

<a href="https://star-history.dera.page/#pufferai/pufferlib&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://star-history.dera.page/svg?repos=pufferai/pufferlib&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://star-history.dera.page/svg?repos=pufferai/pufferlib&type=Date" />
   <img alt="Star History Chart" src="https://star-history.dera.page/svg?repos=pufferai/pufferlib&type=Date" />
 </picture>
</a>
