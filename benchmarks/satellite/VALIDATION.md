# Orbital serving validation

Validation updated on 2026-11-10 for `codex/orbital-serving-engine`.
Upstream baseline: `8f2ba935a5cb82f5fc231eea6b687a87443e88a6`.
Implementation commit: `125221177` (following the foundation commit `3816b836d`).
GPU tests and evaluation script: `5c5d820264784d88c719fc4a6780157c21ea6c54`.
See [architecture, usage, and limits](../../docs/design/orbital_serving.md).

## Result summary

| Validation | Result |
| --- | --- |
| macOS component suite | 67 passed |
| Linux component suite | 67 passed, 12.96 s |
| Single-GPU real-model serving | 3 passed, 57.39 s |
| Two-GPU collectives and independent-engine handoffs | 5 passed, 183.74 s |
| Upstream scheduler/output regressions | 301 passed; one baseline-reproduced hardware limitation |
| GSM8K, first 64 questions, 5-shot | Stock and orbital both 18/64 (28.125%) |
| Real NCCL collective calibration | Nine algorithm/size cases completed; exact SUM checks passed |
| Applicable pre-commit checks | Passed, including types and Buildkite test selection |

The upstream regression suite is **not fully green** on this host. One test
constructs a three-rank pipeline configuration on a machine with two GPUs. It
fails in unchanged `ParallelConfig` validation on both this branch and the exact
upstream baseline; details and reproduction commands are below. No tests were
skipped, rerun automatically, marked xfail, or given wider numerical tolerances.

## Local component validation

Environment: macOS arm64, Python 3.12, PyTorch 2.15.0, CPU/Gloo. Tests import
vLLM from this checkout; CUDA extensions are not built on this host. Importing
the checkout emits a missing generated `vllm._version` metadata warning.

```bash
GLOO_SOCKET_IFNAME=lo0 .venv/bin/python -m pytest \
  --confcutdir=tests/satellite tests/satellite -q
PRE_COMMIT_HOME=/private/tmp/orbital-precommit .venv/bin/pre-commit run
git diff --check
```

Result: **67 passed**, one generated-version metadata warning, in 15.67 seconds.
All applicable pre-commit checks passed, including mypy, formatting, lint, and
Buildkite test selection. `git diff --check` passed. No checks were bypassed.

The local suite covers:

- Deterministic policy decisions and property-based time, token, and energy bounds.
- Actual scheduler and KV allocator integration, pausing, cancellation, deadline
  errors, preservation of frozen blocks, refreshed forecasts, and startup fencing.
- SQLite reopen, competing ownership transactions, lost replies, cancellation,
  stale-owner output, and atomic final-token/completion commits.
- Real request capture and replay with original history and generation limits.
- Actual tensor KV remapping for FP32/FP16/BF16, incomplete-tail zeroing, corrupt
  images, incompatible rank/model/layout metadata, and transfer size bounds.
- Coordinator handoffs using real schedulers and state adapters, with injected
  external RPC reply loss, authority reopen, abort, and destination restart.
- Real 2-, 3-, and 4-process Gloo collectives, a noncontiguous-rank subgroup,
  rank agreement, rejected updates, input preservation, and scratch-cost choices.

These are component and integration tests. Scheduler tests provide explicit
`ModelRunnerOutput` values at the scheduler I/O boundary; they do not execute a
GPU model. Tensor transfer tests use real CPU tensors. No passing CPU test is
reported as a passing CUDA serving test.

## Local calibration smoke run

```bash
GLOO_SOCKET_IFNAME=lo0 PYTHONPATH=. .venv/bin/python -m torch.distributed.run \
  --master-addr=127.0.0.1 --master-port=29617 --nproc-per-node=2 \
  benchmarks/satellite/calibrate_collectives.py \
  --backend gloo --dtype float32 --sizes 256 4096 65536 \
  --repeats 5 --warmup 2 --output /tmp/orbital-collective-calibration.json
```

Completed all nine algorithm/size cases and checked the resulting SUM tensors.
[Recorded measurements](local_gloo_smoke.json) are a local loopback smoke run,
not satellite network measurements or evidence of GPU acceleration. Five
repetitions are insufficient for performance conclusions. Do not deploy this
CPU-derived plan as a GPU calibration. The initial `--standalone` invocation
could not resolve this Mac's hostname; the explicit loopback rendezvous succeeded.

## Linux A100 environment

The GPU runs used an isolated checkout and virtual environment on Ubuntu 24.04.3
x86_64, with two NVIDIA A100 80GB PCIe GPUs. The interconnect is PHB (PCIe host
bridge), not NVLink. The machine had no other GPU workloads during the serving,
accuracy, or collective measurements. CPU component and upstream scheduler tests
ran alongside portions of serving and accuracy validation, so observed timings
may include CPU contention. Collective calibration ran after all other tests and
accuracy runs completed.

| Component | Version |
| --- | --- |
| Python | 3.12.3 |
| NVIDIA kernel driver | 570.195.03 |
| User-space forward compatibility package | cuda-compat-13-0, 580.196.07-1ubuntu1 |
| PyTorch / CUDA runtime | 2.14.0+cu130 / 13.0 |
| NCCL runtime | 2.30.7 |
| Transformers | 5.22.0 |
| FlashInfer / Triton | 0.7.3 / 3.8.0 |
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Model revision | 7ae557604adf67be50417f59c2c2f167def9a775 |

The Python source was installed editable. Compiled extensions came from the
wheel for the **exact upstream commit**, whose C++/CUDA source is unchanged by
this fork. This baseline publishes CUDA 13 artifacts. The initial automatic
backend selection attempted an unavailable CUDA 12.9 index and failed; explicitly
selecting `cu130` resolved installation. The shallow editable checkout reports
`0.1.dev1+gf30925955.precompiled` as its package version, not a released vLLM tag.

Driver 570 needs the official user-space compatibility package for this CUDA 13
runtime. The package was extracted as an ordinary user and enabled only through
`LD_LIBRARY_PATH` for these commands. The system driver and linker configuration
were not changed. See NVIDIA's
[forward compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)
and the [vLLM GPU installation instructions](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/).

The package came from NVIDIA's Ubuntu 24.04 CUDA repository:
`cuda-compat-13-0_580.196.07-1ubuntu1_amd64.deb`, with verified SHA-256
`41fea2138cad65ffdda951c720499a0ac6d9d1f4429844cd63f174a00df9f4a0`.
The system CUDA 12.8 compiler compiled the FlashInfer sampling JIT during the
first run. A real CUDA matrix multiplication and extension import passed before
serving tests started.

```bash
uv venv --python 3.12
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=8f2ba935a5cb82f5fc231eea6b687a87443e88a6 \
VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
uv pip install -e . --torch-backend=cu130
uv pip install -r requirements/test/cuda.in --torch-backend=cu130
uv pip install hypothesis
```

The tested runtime source was commit `f309259552cdfaa3e899eaa600a4c0dd2e8703ab`.
The test setup fixes, new cross-GPU tests, and evaluation script were copied to
the host before execution; their Git blob hashes matched the files subsequently
committed in `5c5d82026`. No runtime implementation fix was required by these GPU
runs. Full versions, source provenance, dataset checksums, and JUnit summaries
are in [the machine-readable record](linux_a100_validation.json).

## Real-model CUDA validation

All commands below ran with the task's compatibility-library directory in
`LD_LIBRARY_PATH`, the virtual environment on `PATH`, and `VLLM_NO_USAGE_STATS=1`.

```bash
.venv/bin/python -m pytest tests/satellite_gpu/test_serving.py \
  -x -vv --timeout=300 --junitxml=/tmp/orbital-serving.xml
.venv/bin/python -m pytest \
  tests/satellite_gpu/test_tensor_parallel.py \
  tests/satellite_gpu/test_handoff.py \
  -x -vv --timeout=300 --junitxml=/tmp/orbital-two-gpu.xml
GLOO_SOCKET_IFNAME=lo .venv/bin/python -m pytest \
  --confcutdir=tests/satellite tests/satellite \
  -q --timeout=180 --junitxml=/tmp/orbital-components.xml
```

The eight CUDA tests exercise actual pretrained-model inference:

- Predictive scheduling compared with stock vLLM's top log probabilities.
- Checkpoint/replay and real GPU KV restoration, with exact continuation tokens
  and the original generation limit under `VLLM_BATCH_INVARIANT=1`.
- TP=2 native, direct, and all-gather collectives compared with stock numerical
  output; per-rank counters confirm that each selected algorithm executed.
- Two independent EngineCore processes on separate physical GPUs, handing off
  by recomputation or KV transfer. The journal receives the exact 64-token
  baseline output, commits completion, and retires the source.

The handoff tests operate through EngineCore and the coordinator's token journal;
they do not migrate an existing OpenAI HTTP connection or frontend session.
The fault-injection scenarios (lost RPC replies, reopen, cancellation, ownership
races, and destination restart) run in the separate component suite.

Two test setup issues were exposed before the passing serving run: the runner
fixture disables chunked prefill by default, and `LLM.generate()` changes its
supplied sampling parameters to final-only output. The tests now enable chunked
prefill explicitly and clone baseline sampling parameters, preserving cumulative
outputs for checkpointing. No runtime behavior or assertion threshold was changed
to accommodate these failures. The final runs emitted dependency deprecation
warnings, including Python's warning about fork from a multithreaded process.

## Upstream regression results

```bash
.venv/bin/python -m pytest \
  tests/v1/core/test_scheduler.py \
  tests/v1/core/test_priority_preemption_bug.py \
  tests/v1/core/test_output.py \
  -q --timeout=120 --maxfail=3 --junitxml=/tmp/orbital-upstream.xml
```

Result: **301 passed, one failed** in 175.78 seconds. The failure is
`test_diffusion_read_deferral_keeps_a_longer_pp_wait`, which sets
`pipeline_parallel_size=3`. `ParallelConfig` rejects world size three when only
two GPUs are available, before it constructs a scheduler.

In a detached worktree at upstream commit `8f2ba935a`, using the same dependencies
and compiled extensions, the following command reproduced the same failure:

```bash
PYTHONPATH=. /home/ubuntu/orbital-validation-20261110/.venv/bin/python -m pytest \
  tests/v1/core/test_scheduler.py::test_diffusion_read_deferral_keeps_a_longer_pp_wait \
  -q --timeout=120 --junitxml=/tmp/orbital-upstream-baseline.xml
```

Both runs report: `World size (3) is larger than the number of available GPUs (2)`.
This case remains unvalidated on hardware meeting its three-GPU requirement.

## Accuracy comparison

```bash
PYTHONPATH=. .venv/bin/python benchmarks/satellite/evaluate_gsm8k.py \
  --mode stock --output /tmp/orbital-gsm8k-stock.json
PYTHONPATH=. .venv/bin/python benchmarks/satellite/evaluate_gsm8k.py \
  --mode orbital --output /tmp/orbital-gsm8k-orbital.json
```

Both use the upstream GSM8K scorer, the first 64 test questions in dataset order,
five training examples, greedy completion, a 256-token cap, TP=2, eager execution,
chunked prefill, no prefix caching, and 20% GPU memory allocation. Orbital uses a
permissive synthetic forecast and a deliberately forced direct-exchange plan.
This compares accuracy; the forced plan is not a measured deployment choice.

| Metric | Stock | Orbital, forced direct |
| --- | --- | --- |
| Correct answers | 18/64 | 18/64 |
| Accuracy | 28.125% | 28.125% |
| Invalid answer rate | 0% | 0% |
| Generated tokens | 7,135 | 7,135 |
| Generation elapsed time | 28.71 s | 48.53 s |
| Output tokens/second | 248.51 | 147.01 |

There was no aggregate accuracy drop on this subset. Matching aggregate counts
does not establish identical answers to every question or accuracy on the full
dataset. Each orbital rank recorded 53,900 direct-exchange calls, including
warmup. Raw results: [stock](linux_a100_gsm8k_stock.json) and
[orbital](linux_a100_gsm8k_orbital.json).

The forced direct path was approximately **69% slower** in this run. This is a
single accuracy run, not a controlled throughput benchmark, and combines
scheduler and communication overhead. It does not establish which component
caused that overhead. It demonstrates why a deployment must calibrate its plan
instead of assuming that a custom collective beats NCCL on a local PCIe link.

## NCCL calibration

```bash
PYTHONPATH=. .venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=2 benchmarks/satellite/calibrate_collectives.py \
  --backend nccl --dtype float16 --sizes 256 4096 65536 \
  --repeats 20 --warmup 5 --output /tmp/orbital-nccl-calibration.json
```

All nine algorithm/size combinations completed, with exact SUM verification on
both ranks. Each case uses five warmup and 20 measured iterations, synchronized
timing, and the maximum duration across ranks. Native NCCL had the lowest median
at all three sizes in this sample; large-message differences were small and
variable. [Raw medians, p95 values, and the candidate plan](linux_a100_nccl.json)
are specific to this two-GPU PCIe execution group. These are not satellite-link
measurements and do not prove an orbital performance improvement.

## Remaining deployment validation

Measured spacecraft power/thermal budgets, orbital/contact traces, delayed or
disconnected links, hardware fault recovery, larger models, additional numerical
formats, and long-running load tests still need deployment-specific validation.
Buildkite selects all new tests, but this report records manual runs, not hosted
Buildkite results. Passing these bounded tests is not a guarantee of bug-free or
flight-ready operation.

Further deployment work includes the fleet router, token gateway integration,
authority availability, hardware telemetry and protection, DTN bulk transfer,
and hardware-specific calibration. Dynamic TP resizing, attention replication
changes, multimodal checkpointing, and seamless OpenAI stream migration are not
implemented by this branch.
