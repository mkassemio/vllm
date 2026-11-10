# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproducible stock/orbital model evaluation using the upstream GSM8K scorer.

Run each mode in a fresh process with the same hardware and arguments. The
orbital forecast is deliberately permissive; this evaluates model accuracy,
not energy savings or performance under satellite resource constraints.
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k_offline
from vllm import LLM, SamplingParams
from vllm.satellite.contracts import Forecast, OperatingWindow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("stock", "orbital"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-questions", type=int, default=64)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument(
        "--collective", choices=("native", "direct", "allgather"), default="direct"
    )
    args = parser.parse_args()
    if args.num_questions < 1 or args.tensor_parallel_size < 1:
        parser.error("Question count and TP degree must be positive")
    revision = "7ae557604adf67be50417f59c2c2f167def9a775"
    with tempfile.TemporaryDirectory(prefix="orbital-eval-") as directory:
        path = Path(directory) / "forecast.json"
        extra: dict = {}
        options: dict = {}
        if args.mode == "orbital":
            extra["orbital"] = {
                "forecast_path": str(path),
                "profile": {
                    "fixed_step_s": 0.001,
                    "prefill_token_s": 0.001,
                    "decode_token_s": 0.001,
                    "power_w": 100,
                },
            }
            costs = [{"name": "native", "latency_s": 1, "seconds_per_byte": 0}]
            if args.collective != "native":
                costs.append(
                    {"name": args.collective, "latency_s": 0, "seconds_per_byte": 0}
                )
            extra["orbital_communication"] = {
                "version": 1,
                "world_size": args.tensor_parallel_size,
                "max_scratch_bytes": 268435456,
                "costs": costs,
            }
            options["scheduler_cls"] = "vllm.satellite.scheduler.SatelliteScheduler"
        llm = LLM(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            revision=revision,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=4096,
            max_num_batched_tokens=128,
            max_num_seqs=8,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            enforce_eager=True,
            async_scheduling=False,
            gpu_memory_utilization=0.2,
            disable_custom_all_reduce=True,
            seed=42,
            additional_config=extra,
            **options,
        )
        if args.mode == "orbital":
            now = time.time()
            forecast = Forecast(
                1,
                now,
                now + 3600,
                (OperatingWindow(now, now + 3600, 1000, 1_000_000, 10, 128, 8),),
            )
            path.write_text(json.dumps(forecast.to_dict()))
        llm.generate(
            ["Warm up this model."], SamplingParams(temperature=0, max_tokens=8)
        )
        result = evaluate_gsm8k_offline(
            llm, num_questions=args.num_questions, num_shots=5, max_tokens=256
        )
        report: dict = {
            "mode": args.mode,
            "model_revision": revision,
            "tensor_parallel_size": args.tensor_parallel_size,
            "collective": args.collective if args.mode == "orbital" else "stock",
            "result": result,
            "scope": (
                "Accuracy regression with permissive synthetic budgets; "
                "not a satellite performance claim."
            ),
        }
        if args.mode == "orbital":
            report["communication"] = llm.collective_rpc(
                "orbital_communication_statistics"
            )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        llm.llm_engine.engine_core.shutdown()
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
