import asyncio

import pytest

from model_providers import (
    AIModel,
    NVIDIA_DEMO_MODELS,
    assign_healthy_text_models,
    check_nvidia_demo_models,
)


def test_missing_provider_defaults_to_openai():
    model = AIModel.from_config(
        {"name": "legacy", "model_name": "gpt-test", "api_key": "test"},
        openai_client_factory=lambda **kwargs: object(),
    )

    assert model.provider == "openai"


def test_nvidia_response_normalizes_reasoning_and_usage():
    class Response:
        content = "answer"
        additional_kwargs = {"reasoning_content": "reasoning"}
        usage_metadata = {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}

    class Client:
        def invoke(self, messages, **kwargs):
            return Response()

    model = AIModel.from_config(
        {
            "name": "nvidia-test",
            "model_name": "vendor/model",
            "api_key": "test",
            "provider": "nvidia",
        },
        nvidia_client_factory=lambda **kwargs: Client(),
    )

    content, usage = asyncio.run(model.call_ai_model("system", "user"))

    assert content == "answer"
    assert usage == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
        "reasoning_content": "reasoning",
    }


def test_health_checks_isolate_failures_and_check_every_model():
    visited = []

    async def checker(config):
        visited.append(config["model_name"])
        if config["model_name"] == "bad/model":
            raise RuntimeError("temporarily unavailable")
        return "ok"

    candidates = [
        {"name": "good", "model_name": "good/model"},
        {"name": "bad", "model_name": "bad/model"},
    ]
    results = asyncio.run(check_nvidia_demo_models("key", candidates, checker=checker))

    assert set(visited) == {"good/model", "bad/model"}
    assert [result["healthy"] for result in results] == [True, False]
    assert "temporarily unavailable" in results[1]["error"]


def test_assignments_only_use_healthy_models():
    assignments = assign_healthy_text_models(
        ["healthy-a", "healthy-b"],
        ["Writer", "Reviewer 1", "Reviewer 2", "Reviewer 3", "Editor"],
        chooser=lambda models: models[0],
    )

    assert set(assignments.values()) == {"healthy-a"}


def test_demo_catalog_contains_requested_models():
    assert {config["model_name"] for config in NVIDIA_DEMO_MODELS} == {
        "google/diffusiongemma-26b-a4b-it",
        "openai/gpt-oss-120b",
        "thinkingmachines/inkling",
        "poolside/laguna-xs-2.1",
        "z-ai/glm-5.2",
        "moonshotai/kimi-k2.6",
        "stepfun-ai/step-3.7-flash",
    }

