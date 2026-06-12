"""Wrapper asíncrono sobre el SDK de Anthropic con rate limiting."""

import asyncio
import json
import logging
from pathlib import Path

from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import SimulationConfig

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.client = AsyncAnthropic(api_key=config.anthropic_api_key)
        self._semaphore = asyncio.Semaphore(config.max_concurrent_calls)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._log_path = Path(config.data_path("logs"))
        self._log_path.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self._log_path / "llm_calls.jsonl", "a", encoding="utf-8")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
    async def chat(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 300,
    ) -> str:
        async with self._semaphore:
            response = await self.client.messages.create(
                model=self.config.llm_model,
                max_tokens=max_tokens,
                temperature=self.config.llm_temperature,
                system=system,
                messages=messages,
            )
            text = response.content[0].text
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens

            self._log_file.write(json.dumps({
                "system": system[:200],
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "response": text[:500],
            }, ensure_ascii=False) + "\n")
            self._log_file.flush()

            return text

    @property
    def estimated_cost_usd(self) -> float:
        # Precios Sonnet 4 aprox
        return (self.total_input_tokens * 3 + self.total_output_tokens * 15) / 1_000_000

    def close(self):
        self._log_file.close()
