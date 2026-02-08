"""Text inference worker — bridges between AI Power Grid and an Ollama/OpenAI backend."""

import asyncio
import logging
import time
from typing import List

import httpx

from .api_client import APIClient
from .config import Settings

logger = logging.getLogger(__name__)

# AIPG context injected when prompt mentions the project
AIPG_CONTEXT = """AI Power Grid (AIPG) is a distributed network for AI workloads with native cryptocurrency incentives. Key points:

- Platform: Distributed AI compute network built on AI Horde with workflow engine
- Tokenomics: 150M max supply
- Network: P2P port 8865, RPC port 9788, PoW/PoUW consensus
- Links: aipowergrid.io, explorer.aipowergrid.io, pool.aipowergrid.io
- Social: @AIPowerGrid (Twitter), t.me/AIPowerGrid (Telegram)
- Meet founder: https://calendly.com/half-aipowergrid/30min"""

AIPG_TERMS = ["aipg", "ai power grid", "aipowergrid"]


class TextWorker:
    def __init__(self):
        self.api = APIClient()
        self.backend = httpx.AsyncClient(timeout=120)
        self.model_name: str = Settings.MODEL_NAME
        self.grid_model_name: str = Settings.GRID_MODEL_NAME or self._build_grid_model_name()

    def _build_grid_model_name(self) -> str:
        """Build the grid-advertised model name with domain prefix."""
        if Settings.BACKEND_TYPE == "ollama":
            return f"gridbridge/{self.model_name}"
        # For openai-compatible backends, try to derive domain
        url = Settings.OPENAI_URL.lower()
        if "openai.com" in url:
            return f"openai/{self.model_name}"
        return f"gridbridge/{self.model_name}"

    def _get_completions_url(self) -> str:
        """Get the chat completions endpoint URL."""
        if Settings.BACKEND_TYPE == "ollama":
            return f"{Settings.OLLAMA_URL}/v1/chat/completions"
        return f"{Settings.OPENAI_URL}/chat/completions"

    def _get_auth_headers(self) -> dict:
        """Get authorization headers for the backend."""
        headers = {"Content-Type": "application/json"}
        if Settings.BACKEND_TYPE == "ollama":
            # Ollama doesn't require auth by default
            return headers
        if Settings.OPENAI_API_KEY:
            headers["Authorization"] = f"Bearer {Settings.OPENAI_API_KEY}"
        return headers

    def _transform_payload(self, payload: dict) -> dict:
        """Transform a Grid/Horde text payload into OpenAI chat format."""
        prompt = payload.get("prompt", "")
        max_tokens = int(payload.get("max_length", 80))
        temperature = float(payload.get("temperature", 0.8))
        top_p = float(payload.get("top_p", 0.9))

        # Check for AIPG mentions and inject context
        has_aipg = any(term in prompt.lower() for term in AIPG_TERMS)
        if has_aipg:
            system_prompt = "You are a helpful assistant with expertise in AI Power Grid (AIPG). Provide concise, accurate information about the platform."
            prompt = f"{AIPG_CONTEXT}\n\nUser Query: {prompt}"
        else:
            system_prompt = "You are a helpful assistant."

        openai_payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        if "stop_sequence" in payload:
            openai_payload["stop"] = payload["stop_sequence"]
        if "frequency_penalty" in payload:
            openai_payload["frequency_penalty"] = float(payload["frequency_penalty"])
        if "presence_penalty" in payload:
            openai_payload["presence_penalty"] = float(payload["presence_penalty"])

        return openai_payload

    async def process_once(self) -> bool:
        """Pop one job, run inference, submit result. Returns True if a job was processed."""
        job = await self.api.pop_job([self.grid_model_name])
        if not job:
            return False

        job_id = job["id"]
        payload = job.get("payload", {})
        logger.info(f"Received job {job_id[:8]} — {payload.get('max_length', '?')} tokens")

        # Transform and send to backend
        openai_payload = self._transform_payload(payload)
        url = self._get_completions_url()
        headers = self._get_auth_headers()

        text = ""
        retries = 0
        while retries < 5:
            try:
                resp = await self.backend.post(url, json=openai_payload, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices and "message" in choices[0]:
                        text = choices[0]["message"].get("content", "")
                    break
                elif resp.status_code == 429:
                    logger.warning("Rate limited, waiting 5s...")
                    await asyncio.sleep(5)
                    retries += 1
                elif resp.status_code >= 500:
                    logger.warning(f"Backend error {resp.status_code}, retrying...")
                    await asyncio.sleep(3)
                    retries += 1
                else:
                    logger.error(f"Backend error {resp.status_code}: {resp.text}")
                    break
            except httpx.ConnectError:
                logger.error("Backend unreachable, retrying in 3s...")
                await asyncio.sleep(3)
                retries += 1
            except httpx.ReadTimeout:
                logger.error("Backend timeout, retrying...")
                retries += 1

        # Submit result
        submit_payload = {
            "id": job_id,
            "generation": text,
            "seed": 0,
        }
        try:
            await self.api.submit_result(submit_payload)
            logger.info(f"Completed job {job_id[:8]}")
        except Exception as e:
            logger.error(f"Failed to submit job {job_id[:8]}: {e}")

        return True

    async def run(self):
        """Main worker loop."""
        logger.info(f"Worker starting — model: {self.grid_model_name}")
        logger.info(f"Backend: {Settings.BACKEND_TYPE} @ {self._get_completions_url()}")

        while True:
            try:
                processed = await self.process_once()
                if not processed:
                    await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                await asyncio.sleep(5)

    async def cleanup(self):
        await self.backend.aclose()
        await self.api.close()
