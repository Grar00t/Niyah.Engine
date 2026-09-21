# Documentation Style Guide

This guide keeps Niyah.Engine documentation readable without allowing presentation quality to outrun evidence quality.

## Structure

Prefer this order for major technical pages:

1. scope;
2. current implementation;
3. reproducible usage or evidence;
4. interpretation;
5. limitations/non-claims;
6. links to deeper evidence.

## Claims

Use precise language:

- **implemented** when source contains the behavior;
- **verified** when a named test/build/runtime record demonstrates it;
- **observed** for local diagnostic behavior that is not a repository-wide guarantee;
- **unestablished** when evidence is insufficient.

Do not convert:

```text
build success              → production readiness
training loss decrease     → generalization
validation loss decrease   → broad reasoning
one good generation        → stable conversational quality
architecture diagram       → runtime evidence
```

## Experiment reporting

A model result should state enough context to avoid becoming detached from its experiment:

```text
checkpoint / optimizer step
dataset role: train | validation | test
sample/token counts
metric definition
metric value
exit status
repository SHA when known
known data-quality limitations
```

If the repository SHA is unknown, say so.

## Visuals

Visuals may summarize architecture or measured results. Every chart should indicate:

- what was measured;
- units/metric;
- experiment scope;
- whether lower/higher is better where ambiguous;
- a caveat when the chart is diagnostic rather than a public benchmark.

Do not use decorative benchmark graphics to imply unsupported model competitiveness.

## Commands

Command examples must match an interface present in the documented source revision. Do not document planned commands as if they are already merged.

## Tone

Keep prose technical and direct. Avoid marketing superlatives unless a reproducible comparison actually supports them.
