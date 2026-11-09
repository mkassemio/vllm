# Orbital serving engine

Orbital serving is an experimental, opt-in extension for inference under
changing compute budgets and communication conditions. Ordinary vLLM is the
reference implementation. The feature is under development; this page records
the architecture and the validation boundary rather than a flight-readiness
claim.

## Architecture

The fleet planner supplies placement and versioned resource forecasts. Each
engine contains a predictive scheduler, a durable request-state manager, and a
communication-aware execution policy. Earth applications enter through a
gateway; onboard applications can submit locally without ground connectivity.

The predictive policy runs before KV allocation. The existing allocator owns
physical blocks. Generated tokens, computed tokens, and delivered tokens are
separate positions. A request handoff freezes its source, prepares a destination,
and commits a new ownership generation before destination execution starts.

Every distributed rank must execute the same collective sequence under the same
plan version. A failed collective fails the execution group; individual ranks
must not independently switch algorithms after a failure.

## Contracts

- Forecast time is Unix seconds. Local time progression uses a monotonic anchor.
- Power is watts; energy is joules. Limits describe the inference payload's
  available allocation, after spacecraft reserves.
- Windows are ordered, nonoverlapping, and half-open. Missing, expired, invalid,
  or regressing forecasts prevent admission.
- A forecast's energy is the remaining allocation at its issue time. The engine
  accounts for work reserved since that issue; repeated reads do not replenish it.
- Cost profiles are calibrated estimates with a safety factor. They do not prove
  worst-case execution time or replace spacecraft hardware protection.
- Requests retain their model revision, token history, generation limits, and
  supported sampling state across recovery.
- At-least-once transport uses idempotent handoff messages and ordered output
  records. It does not promise exactly-once network delivery.

## Implementation milestones

1. Fork, immutable upstream baseline, contracts, and CPU policy tests.
2. Scheduler integration before allocation and accounting changes.
3. Durable replay snapshots, ownership, cancellation, and output fencing.
4. Compatible KV export/import with transactional handoff.
5. Calibrated collective selection and coordinated execution changes.
6. Combined fault scenarios through real serving entry points.
7. Numerical evaluation, performance attribution, and reproducible documentation.

Initial support targets synchronous dense decoder inference and compatible
replicas. Additional sampling, cache, and parallel layouts need explicit
validation. Unsupported combinations must be rejected rather than silently
treated as supported.

## Evaluation

Compare stock vLLM, stock plus the same fleet router, scheduler, scheduler plus
state manager, and all three components. Match hardware, model revision, input
traffic, network traces, and resource limits. Include stale or inaccurate
forecasts, shared-link contention, and handoff overhead. Report deadline goodput,
first-token latency, token gaps, recovery delay, transferred bytes, and wasted
computation. Report modeled resource use separately from measured hardware use.

Run pure policy tests without initializing the GPU stack:

```bash
.venv/bin/python -m pytest --confcutdir=tests/satellite tests/satellite -q
```

CUDA execution and multi-GPU validation require a Linux CUDA host. Passing pure
policy tests is not evidence that GPU serving or KV transfer is correct.
