# Direct collective investigation on two A100 GPUs

Measured on 2026-11-10. Original runtime: `441afdfd1`.
Optimized runtime and measurement harness:
`680c291c03fa15c6a3d3b13cb6261c652a8f3df3`.
The [environment and model revision](VALIDATION.md#linux-a100-environment)
are unchanged: two A100 80GB PCIe GPUs, PyTorch 2.14.0+cu130, NCCL 2.30.7,
and Qwen/Qwen2.5-0.5B-Instruct at revision
`7ae557604adf67be50417f59c2c2f167def9a775`. No CUDA/C++ code or dependencies changed.

## Outcome

The optimization reduced generation time by **27.5%** in the repeated
before/after experiment: **39.39 to 28.55 seconds** for the same 64-question
batch. A follow-up across three independently launched engines per configuration
measured **28.40 seconds orbital versus 27.38 seconds stock**, or **3.7% residual
overhead**. This more conservative comparison is the final stock comparison.

The first experiment used three timed passes in one warmed engine per mode:

| Configuration | Median generation time | Range across three runs |
| --- | --- | --- |
| Stock, before | 28.43 s | 28.25–28.61 s |
| Scheduler only, original | 30.02 s | 28.79–30.30 s |
| Direct communication only, original | 37.77 s | 37.62–37.81 s |
| Full orbital, original direct | 39.39 s | 39.06–39.78 s |
| Stock, fresh comparison | 28.60 s | 28.46–29.10 s |
| Full orbital, optimized direct | 28.55 s | 28.19–28.71 s |
| Direct communication only, optimized | 33.27 s | 33.19–33.27 s |

Because the communication-only result differed from the combined engine, a
second experiment alternated fresh engines: stock, orbital, orbital, stock,
stock, orbital. Each engine performed one full-workload warmup and one timed
pass. The paired order reverses for the middle pair to reduce ordering effects.

| Configuration | Independent-engine times | Median |
| --- | --- | --- |
| Stock | 27.378, 27.772, 27.300 s | 27.378 s |
| Optimized orbital | 27.986, 28.397, 28.860 s | 28.397 s |

The first experiment's apparent 0.2% win does not establish parity or a speedup
across fresh launches. Both GPUs' sampled SM clocks stayed at 1,410 MHz during
all six timed windows. This checks for observed GPU clock changes, but does not
control every source of host variation or isolate scheduling/batching effects.
See [source hashes, test results, and telemetry summaries](optimization/validation.json).

The earlier [single-run result](VALIDATION.md#accuracy-comparison), 48.53 s
versus 28.71 s, gave the reported 69% overhead. With full-workload warmup,
three repeats, and no concurrent validation workload, the unchanged engine's
overhead was 38.5%. The old run cannot isolate how much of the difference came
from warmup, host contention, or other variability. The controlled comparison
above is the basis for the optimization claim.

The original scheduler-only overhead was 5.6%; direct communication alone
added 32.8% relative to the initial stock median. These ablations identify the
larger cost without assuming their effects must add linearly.

## Causes and changes

The original direct implementation already batched its peer sends and receives.
Its overhead was in dispatch and local tensor handling, not an accidentally
serial sequence of ungrouped NCCL operations:

1. It copied its own contribution into a two-rank gather buffer, converted the
   first contribution to FP32, cloned that temporary, added the other rank,
   then converted the result back. The trace records four `aten::copy_` calls,
   including the two conversions and the clone's copy.
2. It used c10d `P2POp`, `batch_isend_irecv`, and Work waits. Stock's matched
   native path uses vLLM's thinner PyNCCL interface. The original orbital
   `native` option also unnecessarily cloned and used c10d.

For two ranks, the optimized implementation receives only the peer's input and
executes one binary addition. FP16/BF16 retain the original single-rounding
semantics, without materializing FP32 buffers. Inputs stay unchanged, results
own separate storage, and groups larger than two retain rank-ordered accumulation.
The larger-group path also avoids a redundant clone after dtype conversion.

CUDA engines now reuse their existing PyNCCL communicator for native, direct,
and all-gather operations. Direct still performs grouped NCCL send/receive, with
the same logical payload. Communication and reduction use the caller's stream,
following [NCCL's stream semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html)
and [grouped peer-exchange contract](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html).
The c10d path remains available when PyNCCL is unavailable on the whole group.
Transport selection is agreed across ranks; disagreement or a disabled agreed
transport fails explicitly instead of switching a single rank's transport.

Both the original and optimized full-model runs recorded **213,248 direct
calls and 18,735,856,384 input/send bytes per rank**, including warmup and all
four dataset passes. These are payload counters, not measurements of NCCL's
protocol overhead on the physical link. The optimization did not skip requests
or disguise native all-reduce as direct exchange.

## Profiling evidence

For one 14,336-byte BF16 direct reduction, on both ranks:

| Operator | Original c10d | Optimized c10d | Optimized PyNCCL |
| --- | --- | --- | --- |
| `aten::copy_` | 4 | 0 | 0 |
| `aten::to` | 2 | 0 | 0 |
| `aten::clone` | 1 | 0 | 0 |
| Local addition | 1 | 1 | 1 |
| NCCL kernel | SendRecv | SendRecv | SendRecv |

These are nested operator counts, not additive kernel counts or timings.
[Recorded counts](optimization/operator_counts.json) establish the executed path.
Profiler setup perturbs latency; only unprofiled iterations contribute to the
following timings. Each case uses 20 warmup and 100 measured iterations, a
synchronization boundary, and the maximum duration across the two ranks.

| BF16 message size | Original direct | Optimized, c10d | Optimized, PyNCCL |
| --- | --- | --- | --- |
| 1,792 bytes, one token | 527.7 µs | 219.1 µs | 111.1 µs |
| 14,336 bytes, eight tokens | 526.9 µs | 217.5 µs | 117.8 µs |
| 229,376 bytes, 128 tokens | 750.2 µs | 591.4 µs | 525.6 µs |

The two smaller cases improve by approximately 79% and 78%; the larger case
improves by 30%. Keeping c10d while changing the local reduction isolates a large
part of the gain; reusing PyNCCL reduces dispatch overhead further.

Native PyNCCL still wins at the smaller sizes (60.8 and 90.5 µs), while all-gather
plus local addition wins at the tested larger size (317.3 µs). A calibrated plan
should choose among these paths for its actual workload. There is no universal
direct-collective speedup across sizes or networks.

Raw measurements, including p95 and candidate costs:
[original](optimization/collectives-before.json),
[optimized c10d](optimization/collectives-after-torch.json), and
[optimized PyNCCL](optimization/collectives-after-pynccl.json).

## Accuracy and correctness

The evaluation uses the first 64 GSM8K test questions, five-shot completion,
greedy decoding, a 256-token cap, TP=2, eager execution, chunked prefill,
no prefix caching, and custom all-reduce disabled. Each mode starts a fresh
engine, warms it with the full dataset once, then measures three passes.
Initialization, warmup, and profiling are excluded from generation timings.

All measured full-orbital and stock passes retained **18/64 correct answers**,
zero invalid answers, and **7,135 generated tokens**. These are repeated
measurements of the same subset, not 192 independent questions. Aggregate
accuracy does not establish per-question equality; the separate real-model
tests check numerical output and exact recovery tokens.

- **68 local component tests passed** in 12.79 s, including real Gloo groups
  of one through four ranks and a noncontiguous-rank subgroup.
- **9 CUDA tests passed** in 294.63 s: eight real-model serving, TP, and handoff
  tests, plus the new CUDA transport/numerical contract test.
- **5 Linux Gloo tests passed** in 23.90 s, covering one through four ranks
  under the GPU host's PyTorch version.
- New numerical checks cover FP16, BF16, FP32, and FP64, cancellation,
  subnormal values, overflow, odd lengths, nonzero storage offsets, input
  preservation, and output ownership. CUDA checks exercise both c10d and
  PyNCCL, reject transport disagreement, and reject use after communicator
  destruction. Numerical tolerances remain zero for these contract checks.
- Applicable pre-commit checks passed, including types, formatting, and
  Buildkite test selection. The new CUDA test extends an already selected file;
  the two-GPU step also tracks the shared numerical fixture as a source dependency.

The enlarged numerical fixtures require a 16 MiB test scratch allowance for
the largest FP64 case. They assert the requested algorithm actually ran; the
separate small-budget test still verifies native fallback. Production scratch
limits and numerical tolerances were not relaxed.

## Reproduction

Use the recorded runtime commits with the measurement scripts from `680c291c0`.
The original-runtime ablations used the updated scripts before runtime changes
were installed. Keep the same dependencies, model revision, and GPU topology.
On the supplied host:

```bash
source /home/ubuntu/orbital-validation-runtime/activate.sh
cd /home/ubuntu/orbital-validation-20261110
PYTHONPATH=. .venv/bin/python benchmarks/satellite/evaluate_gsm8k.py \
  --mode stock --warmup-runs 1 --repeats 3 --output /tmp/stock.json
PYTHONPATH=. .venv/bin/python benchmarks/satellite/evaluate_gsm8k.py \
  --mode orbital --warmup-runs 1 --repeats 3 --output /tmp/orbital.json
```

Use `--mode scheduler` or `--mode communication` for isolated ablations.
The original run order was stock, scheduler, communication, orbital; the
optimized order was orbital, stock, communication. All runs were sequential.
Individual model runs are preserved in the [evidence directory](optimization/).
For the independent-engine follow-up, use `--repeats 1 --warmup-runs 1` in six
fresh processes in the alternating order above. A separate `nvidia-smi` process
sampled GPU clocks, utilization, power, and temperature once per second.

```bash
PYTHONPATH=. .venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=2 benchmarks/satellite/calibrate_collectives.py \
  --backend nccl --transport pynccl --dtype bfloat16 \
  --sizes 896 7168 114688 --warmup 20 --repeats 100 \
  --profile-dir /tmp/orbital-profiles --output /tmp/orbital-costs.json
.venv/bin/python -m pytest \
  tests/satellite_gpu/test_tensor_parallel.py \
  tests/satellite_gpu/test_serving.py tests/satellite_gpu/test_handoff.py \
  -x -vv --timeout=300
```

Use `--transport torch` for the c10d ablation. Chrome traces and full logs remain
in `/home/ubuntu/orbital-optimization-20261110` on the A100 host. The SSH session
carrying calibration was reset after the reports were written; a fresh
connection verified the completed artifacts and that the calibration workers
had exited. This did not interrupt the subsequent model comparisons.

This is a small-model comparison under permissive synthetic budgets. It does
not establish performance for production CUDA graphs, larger models, other TP
degrees, spacecraft energy limits, or intersatellite links. The remaining 3.7%
observed overhead includes the full orbital configuration; these measurements
do not assign an exact fraction of it to scheduling versus communication.
