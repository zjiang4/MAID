from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, Iterable, Mapping, Sequence

from openai import AsyncOpenAI, OpenAI

try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
except ImportError:  # Keep legacy OpenAI installations usable.
    ChatNVIDIA = None


DEFAULT_NVIDIA_DEMO_API_KEY = "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"


NVIDIA_DEMO_MODELS = [
    {
        "name": "NVIDIA DiffusionGemma 26B",
        "model_name": "google/diffusiongemma-26b-a4b-it",
        "temperature": 1,
        "top_p": 0.95,
        "max_completion_tokens": 4096,
        "invoke_kwargs": {"chat_template_kwargs": {"enable_thinking": True}},
    },
    {
        "name": "NVIDIA GPT OSS 120B",
        "model_name": "openai/gpt-oss-120b",
        "temperature": 1,
        "top_p": 1,
        "max_tokens": 4096,
    },
    {
        "name": "NVIDIA Inkling",
        "model_name": "thinkingmachines/inkling",
        "temperature": 1,
        "top_p": 0.95,
        "max_tokens": 8192,
    },
    {
        "name": "NVIDIA Laguna XS 2.1",
        "model_name": "poolside/laguna-xs-2.1",
        "temperature": 1,
        "top_p": 0.95,
        "max_tokens": 8192,
    },
    {
        "name": "NVIDIA GLM 5.2",
        "model_name": "z-ai/glm-5.2",
        "temperature": 1,
        "top_p": 1,
        "max_tokens": 16384,
        "seed": 42,
    },
    {
        "name": "NVIDIA Kimi K2.6",
        "model_name": "moonshotai/kimi-k2.6",
        "temperature": 1,
        "top_p": 1,
        "max_completion_tokens": 16384,
    },
    {
        "name": "NVIDIA Step 3.7 Flash",
        "model_name": "stepfun-ai/step-3.7-flash",
        "temperature": 1,
        "top_p": 0.95,
        "max_completion_tokens": 16384,
    },
]


def _usage_dict(response: Any) -> dict[str, Any]:
    metadata = getattr(response, "usage_metadata", None) or {}
    usage = {
        "prompt_tokens": metadata.get("input_tokens", 0),
        "completion_tokens": metadata.get("output_tokens", 0),
        "total_tokens": metadata.get("total_tokens", 0),
    }
    reasoning = (getattr(response, "additional_kwargs", None) or {}).get(
        "reasoning_content"
    )
    if reasoning:
        usage["reasoning_content"] = reasoning
    return usage


@dataclass
class AIModel:
    name: str
    model_name: str
    api_key: str
    base_url: str | None = None
    model_type: str = "Text Output"
    provider: str = "openai"
    client: Any = None
    async_client: Any = None
    invoke_kwargs: dict[str, Any] | None = None

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        openai_client_factory: Callable[..., Any] = OpenAI,
        openai_async_client_factory: Callable[..., Any] = AsyncOpenAI,
        nvidia_client_factory: Callable[..., Any] | None = None,
    ) -> "AIModel":
        name = str(config.get("name", "")).strip()
        model_name = str(config.get("model_name") or name).strip()
        api_key = str(config.get("api_key", ""))
        if not all((name, model_name, api_key)):
            raise ValueError("Display name, model ID, and API key are required.")

        provider = str(config.get("provider", "openai")).lower()
        base_url = str(config.get("base_url") or "").strip() or None
        model_type = str(config.get("model_type", "Text Output"))
        invoke_kwargs = dict(config.get("invoke_kwargs") or {})

        if provider == "nvidia":
            factory = nvidia_client_factory or ChatNVIDIA
            if factory is None:
                raise RuntimeError(
                    "NVIDIA support requires langchain-nvidia-ai-endpoints."
                )
            client_kwargs = {
                "model": model_name,
                "api_key": api_key,
                **dict(config.get("client_kwargs") or {}),
            }
            client = factory(**client_kwargs)
            return cls(
                name=name,
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                model_type=model_type,
                provider=provider,
                client=client,
                invoke_kwargs=invoke_kwargs,
            )

        if provider != "openai":
            raise ValueError(f"Unsupported provider: {provider}")
        client = openai_client_factory(api_key=api_key, base_url=base_url)
        async_client = openai_async_client_factory(api_key=api_key, base_url=base_url)
        return cls(
            name=name,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            model_type=model_type,
            provider=provider,
            client=client,
            async_client=async_client,
            invoke_kwargs=invoke_kwargs,
        )

    async def call_ai_model(
        self, system_prompt: str, user_prompt: str, json_mode: bool = False
    ) -> tuple[str, dict[str, Any]]:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            if self.provider == "nvidia":
                response = await asyncio.to_thread(
                    self.client.invoke, messages, **(self.invoke_kwargs or {})
                )
                return str(response.content or ""), _usage_dict(response)

            kwargs: dict[str, Any] = {"model": self.model_name, "messages": messages}
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = await self.async_client.chat.completions.create(**kwargs)
            usage = response.usage
            usage_dict = (
                {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
                if usage
                else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )
            return response.choices[0].message.content or "", usage_dict
        except Exception as exc:
            error_msg = f"Error calling model '{self.name}': {exc}"
            print(error_msg)
            return f"Error: {error_msg}", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

    async def call_ai_model_stream(
        self, system_prompt: str, user_prompt: str
    ) -> AsyncGenerator[str, None]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            if self.provider == "nvidia":
                chunks = await asyncio.to_thread(
                    lambda: list(self.client.stream(messages, **(self.invoke_kwargs or {})))
                )
                for chunk in chunks:
                    if getattr(chunk, "content", None):
                        yield str(chunk.content)
                return

            stream = await self.async_client.chat.completions.create(
                model=self.model_name, messages=messages, stream=True
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as exc:
            yield f"Error: {exc}"


async def _default_health_checker(config: Mapping[str, Any]) -> str:
    model = AIModel.from_config(config)
    content, _ = await model.call_ai_model(
        "You are a health-check assistant.", "Reply with exactly: OK"
    )
    if not content or content.startswith("Error:"):
        raise RuntimeError(content or "Empty response")
    return content


async def check_nvidia_demo_models(
    api_key: str,
    candidates: Sequence[Mapping[str, Any]] = NVIDIA_DEMO_MODELS,
    *,
    checker: Callable[[Mapping[str, Any]], Any] = _default_health_checker,
    timeout: float = 45.0,
    on_result: Callable[[int, int, Mapping[str, Any]], Any] | None = None,
) -> list[dict[str, Any]]:
    async def check(candidate: Mapping[str, Any]) -> dict[str, Any]:
        client_keys = {
            key: candidate[key]
            for key in (
                "temperature",
                "top_p",
                "max_tokens",
                "max_completion_tokens",
                "seed",
            )
            if key in candidate
        }
        config = {
            "name": candidate["name"],
            "model_name": candidate["model_name"],
            "api_key": api_key,
            "provider": "nvidia",
            "model_type": "Text Output",
            "client_kwargs": client_keys,
            "invoke_kwargs": dict(candidate.get("invoke_kwargs") or {}),
        }
        try:
            response = await asyncio.wait_for(checker(config), timeout=timeout)
            return {"config": config, "healthy": True, "response": response, "error": ""}
        except Exception as exc:
            return {"config": config, "healthy": False, "response": "", "error": str(exc)}

    async def indexed_check(index: int, candidate: Mapping[str, Any]):
        return index, await check(candidate)

    results: list[dict[str, Any] | None] = [None] * len(candidates)
    completed = 0
    tasks = [
        asyncio.create_task(indexed_check(index, candidate))
        for index, candidate in enumerate(candidates)
    ]
    for task in asyncio.as_completed(tasks):
        index, result = await task
        results[index] = result
        completed += 1
        if on_result:
            callback_result = on_result(completed, len(candidates), result)
            if asyncio.iscoroutine(callback_result):
                await callback_result
    return [result for result in results if result is not None]


def assign_healthy_text_models(
    healthy_names: Sequence[str],
    roles: Iterable[str],
    *,
    chooser: Callable[[Sequence[str]], str] = random.choice,
) -> dict[str, str]:
    if not healthy_names:
        return {}
    return {role: chooser(healthy_names) for role in roles}
