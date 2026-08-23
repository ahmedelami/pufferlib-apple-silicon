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
default; BF16 is opt-in for direct-MPS rollouts.

Run the documented 4,096-agent direct-MPS workload with:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python benchmarks/apple_silicon.py \
  --env breakout --agents 4096 --horizon 64 --minibatch-size 65536 \
  --threads 18 --torch-threads 12 --torch-interop-threads 1 \
  --warmup-epochs 3 --epochs 11 --compile-policy auto --modes mps
```

On the target 20-GPU-core M5 Pro, two production runs of the guarded compiled
FP32 path with its fused Metal/Philox rollout sampler measured 2,094,181 and
2,091,980 agent steps/s. The representative ~2.093M result is 4.58x the tuned
CPU path, 2.47x the CPU-rollout/MPS-learner path, and 1.70x eager MPS on the
documented 4,096-agent Breakout workload. A provenance-checked 10-seed learning
holdout passed the predeclared tail-score, learning-AUC, steps-to-score,
finiteness, and full-run wall-time gates against eager MPS. Compiler and sampler
startup were included: steps-to-score matched, median early wall-clock
milestones were slower because of cold startup, and the complete 32-epoch runs
were 1.2468x faster. The full shared suite reports 214 passed, 25 explicitly skipped, and no
failures. Other hardware, shapes, and policies stay eager. This is a same-Mac
comparison; CUDA parity remains unclaimed until the harness's `--modes cuda`
path is run on a named NVIDIA host.

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
