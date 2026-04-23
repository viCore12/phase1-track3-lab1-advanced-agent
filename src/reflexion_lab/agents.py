"""
ReAct and Reflexion Agent implementations.

- ReActAgent:    single-attempt agent (no reflection loop).
- ReflexionAgent: multi-attempt agent that uses LLM-generated reflections
                  to improve answers across attempts.

Bonus features implemented here:
  - reflection_memory: accumulated strategies passed to each Actor call.
  - structured_evaluator: JudgeResult is Pydantic-validated (in llm_runtime).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .llm_runtime import (
    FAILURE_MODE_BY_QID,
    actor_answer,
    evaluator,
    reflector,
)
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord


@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1

    def run(self, example: QAExample) -> RunRecord:
        # reflection_memory: list of strategy strings (reflection_memory bonus)
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0

        for attempt_id in range(1, self.max_attempts + 1):
            # --- Actor: get answer + REAL token count + REAL latency ---
            answer, total_tokens, latency_ms = actor_answer(
                example, attempt_id, self.agent_type, reflection_memory
            )

            # --- Evaluator (structured_evaluator bonus) ---
            judge = evaluator(example, answer)

            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                # Real token count from API usage metadata
                token_estimate=total_tokens,
                # Real latency from perf_counter
                latency_ms=latency_ms,
            )

            final_answer = answer
            final_score = judge.score

            if judge.score == 1:
                traces.append(trace)
                break

            # --- Reflexion loop (reflection_memory bonus) ---
            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection = reflector(example, attempt_id, judge)
                # Accumulate strategy for the next Actor call
                reflection_memory.append(reflection.next_strategy)
                reflections.append(reflection)
                trace.reflection = reflection

            traces.append(trace)

        total_tokens_all = sum(t.token_estimate for t in traces)
        total_latency_all = sum(t.latency_ms for t in traces)

        failure_mode = (
            "none"
            if final_score == 1
            else FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")
        )

        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens_all,
            latency_ms=total_latency_all,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )


class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)


class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
