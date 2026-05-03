# Feasibility-Constrained KV-Cache Management

Subtitle: Characterizing the Optimality Gap in Edge LLM Backends

## Research Thesis

Efficient long-context inference on edge hardware is constrained by more than algorithmic compression quality. A theoretically attractive KV-cache policy can be unusable when the inference backend cannot execute the required cache layout, precision, or retention mechanism under real operating conditions.

This project studies that mismatch as a systems problem:

```text
theoretical action space A != backend-feasible action space F
```

The central claim is the existence of a **Backend-Induced Optimality Gap**: the task-quality difference between the best policy in the theoretical action space and the best policy that is executable by the backend under physical system constraints.

## Action Space

The theoretical KV-cache policy space is:

```text
A = { (r, b) | r in R, b in B }
```

where:

- `r` is the retained fraction of the KV cache;
- `b` is the KV precision in bits;
- `R` is a configurable retention grid;
- `B` is the set of candidate KV precisions.

The backend-feasible set is:

```text
F_backend subset A
```

An action can be excluded from `F_backend` when the backend lacks the required cache kernel, layout conversion, quantized cache path, retention path, or composition of those mechanisms.

## Research Objectives

### Objective 1: Theoretical Frontier

Map the memory-quality frontier over the action space `A`.

Requirements:

- define retention/precision actions;
- estimate KV-cache memory for each action;
- measure task quality with real backend runs when execution is possible;
- label any surrogate estimate explicitly as a surrogate.

### Objective 2: Backend Feasible Set

Empirically map `F_backend` for each backend version.

Requirements:

- probe installed backend capabilities;
- run execution tests for each `(r, b)` pair;
- classify each action as `real`, `backend_infeasible`, or `simulated`;
- store backend name, version, model, action, result, and failure reason.

### Objective 3: Physical Telemetry

Measure the operating state of the machine during inference.

Signals:

- memory pressure;
- swap usage and swap delta;
- pageouts;
- compressed memory;
- process RSS;
- available unified memory;
- thermal state;
- latency tail behavior.

The telemetry layer must produce time-series traces, not only final snapshots.

### Objective 4: Dynamic Controller

Select the best backend-feasible policy under a runtime cost threshold.

Optimization target:

```text
maximize Score(action, task)
subject to action in F_backend
subject to Cost(action, system_state) <= threshold
```

The controller must never present a theoretical or simulated policy as an executed policy.

### Objective 5: Optimality Gap

Quantify the cost of moving from theory to executable reality.

Definition:

```text
Gap(s) = Score(a*_A, s) - Score(a*_F, s)
```

where:

```text
a*_A = argmax Score(a, s), a in A
a*_F = argmax Score(a, s), a in F_backend
```

The result is meaningful only when both scores are measured or when surrogate scores are clearly labeled and separated from measured results.

## Validation Protocol

### Feasible-Set Probe

Run every action in the configured grid against the backend and record whether it executes.

Primary output:

```text
F(r, b, backend, version, model)
```

### Long-Context Retrieval

Use Needle-in-a-Haystack style prompts to measure whether retention policies preserve evidence at different positions.

Primary metrics:

- exact match;
- containment;
- latency per generated token;
- cache memory;
- telemetry trace.

### Long-Context Document QA

Use document QA prompts to measure semantic degradation under retention and precision changes.

Primary metrics:

- exact match or task-specific score;
- generated answer;
- reference answer;
- memory footprint;
- latency;
- telemetry trace.

### Thermal and Memory Soak

Run inference while applying controlled memory pressure and long-duration thermal load.

Primary metrics:

- phase transition in policy choice;
- swap and pageout response;
- thermal response;
- latency tail;
- selected action stability.

## Non-Negotiable Reporting Rules

- Every result must include backend version and hardware.
- Every policy must be labeled by execution status.
- Simulated policies must not be mixed with real measurements.
- Surrogate quality scores must not be called accuracy.
- Negative results are valid if they identify backend or physical limits.

## Phase 1 Entry Criteria

The first clean implementation phase starts with:

1. backend feasible-set probing;
2. OS telemetry time-series;
3. a reproducible memory-soak harness;
4. a minimal action grid over retention and precision;
5. a report schema that separates real, simulated, and infeasible actions.
