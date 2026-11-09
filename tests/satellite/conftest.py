# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real scheduler/allocator integration; no GPU model is substituted or mocked."""

import json
import time

import pytest
from transformers import Qwen2Config, Qwen2ForCausalLM

from tests.v1.core.utils import create_scheduler
from vllm.satellite.scheduler import SatelliteScheduler


@pytest.fixture
def scheduler_factory(tmp_path):
    model_path = tmp_path / "model"
    model_config = Qwen2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        architectures=["Qwen2ForCausalLM"],
    )
    Qwen2ForCausalLM(model_config).save_pretrained(model_path)
    now = time.time()
    forecast_path = tmp_path / "forecast.json"
    forecast_path.write_text(
        json.dumps(
            {
                "version": 1,
                "issued_s": now - 1,
                "expires_s": now + 3600,
                "windows": [
                    {
                        "start_s": now - 1,
                        "end_s": now + 3600,
                        "max_power_w": 100,
                        "energy_j": 10000,
                        "max_step_s": 1,
                        "max_tokens": 3,
                        "max_requests": 2,
                    }
                ],
            }
        )
    )

    def factory(**kwargs):
        scheduler = create_scheduler(
            model=str(model_path),
            max_model_len=256,
            max_num_batched_tokens=32,
            skip_tokenizer_init=True,
            scheduler_cls=SatelliteScheduler,
            device="cpu",
            additional_config={
                "orbital": {
                    "forecast_path": str(forecast_path),
                    "profile": {
                        "fixed_step_s": 0.001,
                        "prefill_token_s": 0.001,
                        "decode_token_s": 0.01,
                        "power_w": 50,
                    },
                }
            },
            **kwargs,
        )
        forecast = json.loads(forecast_path.read_text())
        forecast["version"] += 1
        forecast["issued_s"] = time.time()
        forecast_path.write_text(json.dumps(forecast))
        return scheduler

    return factory


@pytest.fixture
def scheduler(scheduler_factory):
    return scheduler_factory()
