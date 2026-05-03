# Lumina

Lumina is a research codebase for feasibility-constrained KV-cache management in edge LLM inference.

The project studies the gap between:

- the theoretical KV-cache action space available to an algorithm; and
- the subset of policies that an inference backend can actually execute under physical system constraints.

The core research object is the **Backend-Induced Optimality Gap**:

```text
Gap(s) = Score(a*_A, s) - Score(a*_F, s)
```

where `a*_A` is the best policy in the theoretical action space and `a*_F` is the best backend-feasible policy under the same operating state.

## Current Scope

This repository is intentionally minimal. It contains only the primitives needed for the first clean measurement campaign:

- KV-cache action definitions;
- analytical KV-cache memory estimates;
- backend feasible-set classification;
- MLX-LM capability probing;
- macOS telemetry collection;
- runtime cost and policy-selection primitives;
- optimality-gap analysis utilities;
- a memory-soak helper for controlled contention experiments.

## Reporting Discipline

Every experiment must label policy execution status explicitly:

- `real`;
- `backend_infeasible`;
- `simulated`.

Simulated or surrogate results must not be mixed with measured backend execution results.
