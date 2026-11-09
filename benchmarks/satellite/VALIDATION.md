# Orbital serving validation

Validation recorded on 2026-11-09 for `codex/orbital-serving-engine`.
Upstream baseline: `8f2ba935a5cb82f5fc231eea6b687a87443e88a6`.
See [architecture, usage, and limits](../../docs/design/orbital_serving.md).

## Executed locally

Environment: macOS arm64, Python 3.12, PyTorch 2.15.0, CPU/Gloo. Tests import
vLLM from this checkout; CUDA extensions are not built on this host. Importing
the checkout emits a missing generated `vllm._version` metadata warning.

```bash
GLOO_SOCKET_IFNAME=lo0 .venv/bin/python -m pytest \
  --confcutdir=tests/satellite tests/satellite -q
PRE_COMMIT_HOME=/private/tmp/orbital-precommit .venv/bin/pre-commit run
git diff --check
```

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

## Calibration smoke run

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

## Pending release gates

No Linux CUDA host or SSH connection was available during this validation.
The following tests are implemented and selected by Buildkite, but have **not
been executed**:

```bash
.venv/bin/python -m pytest tests/satellite_gpu/test_serving.py -v
.venv/bin/python -m pytest tests/satellite_gpu/test_tensor_parallel.py -v
```

They cover real pretrained Qwen inference, stock-vs-orbital numerical comparisons,
checkpoint/replay and GPU KV continuation, and two-GPU communication dispatch.
Run these, relevant upstream scheduler/engine tests, model evaluations, and
measured resource/network trace experiments before claiming GPU correctness,
performance improvement, or production readiness. Buildkite wiring in a fork
does not itself mean hosted Buildkite jobs have run.

Further deployment work includes the fleet router, token gateway integration,
authority availability, hardware telemetry and protection, DTN bulk transfer,
and hardware-specific calibration. Dynamic TP resizing, attention replication
changes, multimodal checkpointing, and seamless OpenAI stream migration are not
implemented by this branch.
