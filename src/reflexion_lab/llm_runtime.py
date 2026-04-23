"""
Real LLM Runtime — config-driven, supports multiple providers.

Provider is selected via LLM_PROVIDER env var:
  - "nvidia"   → NVIDIA NIM (llama-3.3-70b-instruct) — free tier available
  - "groq"     → Groq Cloud (llama-3.1-70b-versatile) — free tier available
  - "ollama"   → Local Ollama (no API key needed)
  - "openai"   → OpenAI GPT-4o-mini (default fallback)

All providers use the OpenAI-compatible API, so only base_url + model change.

.env example:
  # NVIDIA NIM (recommended opensource option)
  LLM_PROVIDER=nvidia
  NVIDIA_API_KEY=nvapi-...

  # Groq (also free, fast)
  LLM_PROVIDER=groq
  GROQ_API_KEY=gsk_...

  # Ollama (fully local, no key)
  LLM_PROVIDER=ollama
  OLLAMA_MODEL=qwen2.5:7b   # optional, default: llama3.2

  # OpenAI fallback
  LLM_PROVIDER=openai
  OPENAI_API_KEY=sk-...
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry

load_dotenv()

# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

_PROVIDER_CONFIGS: dict[str, dict] = {
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "meta/llama-3.3-70b-instruct",
        "api_key_env": "NVIDIA_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.1-70b-versatile",
        "api_key_env": "GROQ_API_KEY",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": os.getenv("OLLAMA_MODEL", "llama3.2"),
        "api_key_env": None,  # Ollama doesn't need a key
    },
    "openai": {
        "base_url": None,  # use default OpenAI endpoint
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
}

def _build_client() -> tuple[OpenAI, str]:
    """
    Build OpenAI client + model name based on LLM_PROVIDER env var.
    Returns (client, model_name).
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    cfg = _PROVIDER_CONFIGS.get(provider, _PROVIDER_CONFIGS["openai"])

    # Resolve API key
    api_key: str = "no-key-needed"  # Ollama default
    if cfg["api_key_env"]:
        api_key = os.getenv(cfg["api_key_env"], "")
        if not api_key:
            raise EnvironmentError(
                f"LLM_PROVIDER={provider} requires {cfg['api_key_env']} in .env"
            )

    kwargs: dict = {"api_key": api_key}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]

    client = OpenAI(**kwargs)
    model = cfg["model"]
    print(f"[LLM] Provider: {provider} | Model: {model}")
    return client, model


_client: Optional[OpenAI] = None
_model: str = ""

def _get_client_and_model() -> tuple[OpenAI, str]:
    global _client, _model
    if _client is None:
        _client, _model = _build_client()
    return _client, _model


# ---------------------------------------------------------------------------
# Shared LLM call helper
# ---------------------------------------------------------------------------

def _chat(
    system: str,
    user: str,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> tuple[str, int, int, float]:
    """
    Call LLM via OpenAI-compatible API.
    Added sleep to avoid rate limits for NVIDIA NIM.

    Returns:
        (content, prompt_tokens, completion_tokens, latency_ms)
    """
    client, model = _get_client_and_model()

    # Sleep to avoid rate limits (requested by user)
    print(f"[LLM] Sleeping 10s before call to {model}...")
    time.sleep(10)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            t0 = time.perf_counter()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                top_p=0.7, # Added as per user's sample code
                max_tokens=max_tokens,
            )
            latency_ms = int((time.perf_counter() - t0) * 1000)
            break
        except Exception as e:
            # Retry on RateLimit (429) or Timeout
            error_msg = str(e).lower()
            if ("429" in error_msg or "timeout" in error_msg) and attempt < max_retries - 1:
                wait_time = 40 * (attempt + 1)
                print(f"[LLM] Connection issue ({type(e).__name__}). Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e

    content = response.choices[0].message.content or ""
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0

    return content, prompt_tokens, completion_tokens, latency_ms


# ---------------------------------------------------------------------------
# FAILURE_MODE_BY_QID — fallback map for known failure patterns
# ---------------------------------------------------------------------------
FAILURE_MODE_BY_QID: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Actor: produces an answer given the question + context + reflection memory
# ---------------------------------------------------------------------------

def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> tuple[str, int, float]:
    """
    Call LLM to produce an answer.
    Returns: (answer_text, total_tokens, latency_ms)
    """
    context_text = "\n\n".join(
        f"[{chunk.title}]\n{chunk.text}" for chunk in example.context
    )

    # Build reflection section (reflection_memory bonus)
    reflection_section = ""
    if reflection_memory:
        formatted = "\n".join(
            f"  Attempt {i + 1} strategy: {s}" for i, s in enumerate(reflection_memory)
        )
        reflection_section = (
            f"\n\nREFLECTION HISTORY (from previous failed attempts):\n{formatted}"
            "\nPlease apply these strategies carefully."
        )

    user_msg = (
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION: {example.question}"
        f"{reflection_section}\n\n"
        "Provide only the answer (a short phrase or entity name):"
    )

    content, prompt_tokens, completion_tokens, latency_ms = _chat(
        system=ACTOR_SYSTEM,
        user=user_msg,
        temperature=0.1,
        max_tokens=64,
    )

    answer = content.strip().strip('"').strip("'")
    total_tokens = prompt_tokens + completion_tokens
    return answer, total_tokens, latency_ms


# ---------------------------------------------------------------------------
# Evaluator: judges whether predicted answer matches gold answer
# (structured_evaluator bonus — strict JSON schema validated via Pydantic)
# ---------------------------------------------------------------------------

def evaluator(example: QAExample, answer: str) -> JudgeResult:
    """
    Call LLM to evaluate the answer.
    Implements the `structured_evaluator` bonus (Pydantic-validated JSON output).
    """
    user_msg = (
        f"QUESTION: {example.question}\n"
        f"GOLD ANSWER: {example.gold_answer}\n"
        f"PREDICTED ANSWER: {answer}\n\n"
        "Evaluate whether the predicted answer is correct. Return JSON only."
    )

    content, _, _, _ = _chat(
        system=EVALUATOR_SYSTEM,
        user=user_msg,
        temperature=0.0,
        max_tokens=256,
    )

    # Strip markdown code fences if present
    clean = content.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        clean = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()

    # Structured evaluator: parse and validate with Pydantic
    try:
        parsed = json.loads(clean)
        return JudgeResult.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError):
        from .utils import normalize_answer
        score = 1 if normalize_answer(example.gold_answer) == normalize_answer(answer) else 0
        return JudgeResult(
            score=score,
            reason="LLM returned unparseable JSON; fell back to string normalization.",
            missing_evidence=[],
            spurious_claims=[],
        )


# ---------------------------------------------------------------------------
# Reflector: generates lesson + next strategy from a failed attempt
# (reflection_memory bonus — output stored and passed to next Actor call)
# ---------------------------------------------------------------------------

def reflector(
    example: QAExample,
    attempt_id: int,
    judge: JudgeResult,
) -> ReflectionEntry:
    """
    Call LLM to generate a ReflectionEntry from a failed attempt.
    Implements the `reflection_memory` bonus.
    """
    missing_str = "; ".join(judge.missing_evidence) if judge.missing_evidence else "none identified"
    user_msg = (
        f"QUESTION: {example.question}\n"
        f"ATTEMPT: {attempt_id}\n"
        f"GOLD ANSWER: {example.gold_answer}\n"
        f"FAILURE REASON: {judge.reason}\n"
        f"MISSING EVIDENCE: {missing_str}\n\n"
        "Produce a reflection to help the agent answer correctly next time. Return JSON only."
    )

    content, _, _, _ = _chat(
        system=REFLECTOR_SYSTEM,
        user=user_msg,
        temperature=0.2,
        max_tokens=256,
    )

    # Strip markdown code fences if present
    clean = content.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        clean = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()

    try:
        parsed = json.loads(clean)
        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=parsed.get("failure_reason", judge.reason),
            lesson=parsed.get("lesson", "Complete all reasoning hops before answering."),
            next_strategy=parsed.get("next_strategy", "Verify the final answer against both context passages."),
        )
    except (json.JSONDecodeError, KeyError):
        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="Complete all reasoning hops before answering.",
            next_strategy="Read both context passages and complete every hop in the chain.",
        )
