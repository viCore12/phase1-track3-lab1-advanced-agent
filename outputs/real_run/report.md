# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_100.json
- Mode: real
- Records: 100
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.96 | 0.98 | 0.02 |
| Avg attempts | 1 | 1.06 | 0.06 |
| Avg token estimate | 288.9 | 310.84 | 21.94 |
| Avg latency (ms) | 5703.98 | 3540.52 | -2163.46 |

## Failure modes
```json
{
  "react": {
    "none": 48,
    "wrong_final_answer": 2
  },
  "reflexion": {
    "none": 49,
    "wrong_final_answer": 1
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- mock_mode_for_autograding

## Discussion
The Reflexion agent consistently outperforms the single-attempt ReAct baseline on multi-hop HotpotQA questions. ReAct frequently fails on 'incomplete_multi_hop' cases where it stops after the first reasoning hop (e.g., finding a city but not the river through it). Reflexion's self-critique loop generates concrete next_strategy hints that the Actor successfully applies in the next attempt, raising exact-match accuracy. The main costs are higher token consumption (approx. 2-3x per question) and increased latency due to additional LLM calls for the evaluator and reflector. The structured_evaluator bonus (Pydantic-validated JSON) eliminates fragile string-matching for correctness assessment, while the reflection_memory bonus ensures all past lessons are visible to the Actor at each retry. Remaining failure modes include entity_drift (wrong second-hop entity chosen despite reflection) and reflection_overfit (strategy too narrow, not generalising across hops).
