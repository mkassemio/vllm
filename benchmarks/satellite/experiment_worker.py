# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real GPU actor for controlled EngineCore experiments, not an HTTP server."""

import json
import subprocess
import sys
import time
import traceback
from multiprocessing.connection import Client
from pathlib import Path


def main():
    connection = Client(sys.argv[1], family="AF_UNIX")
    llm = None
    try:
        from vllm import LLM, SamplingParams
        from vllm.v1.engine import EngineCoreRequest

        config = json.loads(Path(sys.argv[2]).read_text())
        llm = LLM(**config)
        core = llm.llm_engine.engine_core
        tokenizer = llm.get_tokenizer()
        engine = core.engine_core

        def step():
            before = {
                name: (r, r.num_computed_tokens, r.num_preemptions)
                for name, r in engine.scheduler.requests.items()
            }
            start = time.monotonic()
            output = core.get_output()
            end = time.monotonic()
            scheduled = []
            preemptions = []
            for name, (request, computed, previous_preemptions) in before.items():
                reset = request.num_preemptions > previous_preemptions
                if reset:
                    preemptions.append(name)
                origin = 0 if reset else computed
                count = request.num_computed_tokens - origin
                if count > 0:
                    scheduled.append(
                        {
                            "request_id": name,
                            "tokens": count,
                            "computed_start": origin,
                            "kind": "prefill"
                            if origin < request.num_prompt_tokens or count > 1
                            else "decode",
                        }
                    )
            policy = getattr(engine.scheduler, "orbital_plan", None)
            return {
                "start_mono": start,
                "end_mono": end,
                "elapsed_s": end - start,
                "model_executed": core.last_step_model_executed,
                "scheduled": scheduled,
                "preempted": preemptions,
                "policy": {
                    "estimated_s": policy.estimated_s,
                    "token_limits": policy.token_limits,
                }
                if policy is not None
                else None,
                "outputs": [
                    {
                        "id": o.request_id,
                        "tokens": o.new_token_ids,
                        "finish_reason": str(o.finish_reason)
                        if o.finish_reason is not None
                        else None,
                        "error": str(o.error) if o.error else None,
                    }
                    for o in output.outputs
                ],
            }

        import torch

        import vllm

        connection.send(
            {
                "result": {
                    "source_commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True
                    ).strip(),
                    "vllm_file": vllm.__file__,
                    "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(),
                    "pid": __import__("os").getpid(),
                }
            }
        )
        while True:
            command, payload = connection.recv()
            try:
                if command == "shutdown":
                    break
                if command == "prompt":
                    seed = tokenizer.encode(
                        "Satellite computers process observations and deliver useful "
                        "results. Explain the following engineering considerations. ",
                        add_special_tokens=False,
                    )
                    length = payload["length"]
                    result = (seed * (length // len(seed) + 1))[:length]
                elif command == "add":
                    params = SamplingParams(
                        temperature=0,
                        max_tokens=payload["output_tokens"],
                        ignore_eos=True,
                        extra_args=payload.get("extra_args"),
                    )
                    core.add_request(
                        EngineCoreRequest(
                            request_id=payload["id"],
                            prompt_token_ids=payload["prompt"],
                            mm_features=None,
                            sampling_params=params,
                            pooling_params=None,
                            arrival_time=payload.get("arrival_time", time.time()),
                            lora_request=None,
                            cache_salt=None,
                            data_parallel_rank=None,
                            priority=payload.get("priority", 0),
                            external_req_id=payload["id"],
                        )
                    )
                    result = None
                elif command == "step":
                    result = step()
                elif command == "abort":
                    core.abort_requests(payload["ids"])
                    result = None
                elif command == "control":
                    result = core.orbital_control(
                        payload["command"], payload["payload"]
                    )
                elif command == "status":
                    result = {"live": sorted(engine.scheduler.requests)}
                    if hasattr(engine.scheduler, "orbital_status"):
                        result["orbital"] = engine.scheduler.orbital_status()
                else:
                    raise ValueError(f"Unknown command: {command}")
                connection.send({"result": result})
            except Exception:
                connection.send({"error": traceback.format_exc()})
    except BaseException:
        connection.send({"error": traceback.format_exc()})
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
        connection.close()


if __name__ == "__main__":
    main()
