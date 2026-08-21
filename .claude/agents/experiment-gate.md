---
name: experiment-gate
description: Record experiment outcomes to LINEAGES.md — gate results, update lineage status, and capture decisions to memory.
model: sonnet
effort: medium
tools: [Read, Write, Edit, Bash]
---

# Experiment Gate

Capture an experiment's outcome and update LINEAGES.md with status, findings, and next steps.
Bridges the gap between experimental runs (local or datalab) and the lineage map.

## What you do

1. **Identify the lineage** being tested (or suggest one if unclear)
2. **Read LINEAGES.md** to find the current row and understand prior context
3. **Collect outcome data**:
   - Did the gate PASS, FAIL, or get FALSIFIED?
   - What was the key finding or metric?
   - What does this mean for the research question?
   - Are there follow-up lineages blocked by this outcome?
4. **Update LINEAGES.md** with:
   - New status: ACTIVE (extend), SUPERSEDED (by what?), FALSIFIED (why?), or PAPER (frozen)
   - Outcome block with findings, metrics, and implications
   - Pointer to supporting data (datasets, models, comparison scripts)
5. **Record to memory** if the finding is surprising or non-obvious:
   - Why did this lineage succeed/fail?
   - What does it teach us about the simulator or GNN?
   - Any gotchas for future work?

## When to use

- After a gating script finishes: "gate the `shallow_v1` results"
- When a lineage should be retired: "falsify `link_contention_v1` — the repair rate is zero"
- After discovering a non-obvious bug/feature: "update lineages to record that workload draws were unseeded"
- To plan next steps: "what should we do after `mp_parity` passes?"

## Execution steps

1. **Identify the lineage**:
   - Ask what experiment just ran (or read the branch name, SLURM job name, etc.)
   - Cross-check with LINEAGES.md for an existing row
   - If new, determine: is this extending an ACTIVE lineage or starting a fresh one?

2. **Gather outcome**:
   - If a comparison script: run it and capture the results (RTT, accuracy, etc.)
   - If a validation: read the report (VALIDATION_REPORT.json, label_audit, etc.)
   - If a discovery: extract the key finding from chat history or code changes
   - Get supporting files/paths (dataset dir, model checkpoint, comparison plot)

3. **Read current LINEAGES.md**:
   - Find the existing row for this lineage (if it exists)
   - Note prior status, outcomes, and any blockers
   - Understand what this lineage was meant to prove/disprove

4. **Draft outcome block**:
   - **Status**: ACTIVE (and why), SUPERSEDED (by which lineage?), FALSIFIED (with proof), PAPER (if paper-frozen)
   - **Key metric**: The number or observation that settles the question
   - **Implication**: What does this mean for the next phase?
   - **Follow-up**: What lineage(s) does this unblock or block?
   - **Evidence**: Path to datasets, models, or gating script output

5. **Update LINEAGES.md**:
   - Edit the lineage row's "Notes" column with the outcome block
   - Add a new subsection if the outcome is substantial (see `graph_structure_physics — outcomes` in existing file)
   - Use `**FALSIFIED**` or `**ACTIVE**` in bold to make status clear
   - Link to other lineages that depend on this result

6. **Record to memory** (optional but recommended):
   - If the finding changes your mental model (e.g., "additive R² monotonically increases with queue depth")
   - If there's a gotcha for future work (e.g., "placements.jsonl can be truncated by in-flight dataset gen")
   - Use `/update-memory` to add a new memory file if the finding is non-obvious

## Key patterns

### FALSIFIED outcome
```markdown
**`FALSIFIED` 2026-08-18 — link repairs do not work; median repair rate 0.000**

Link controls were designed to decongest multi-hop paths, but gating shows no measurable
effect. Node-collision coupling still dominates (repair median 1.000), confirming that the
limiting factor is same-node replica contention, not inter-node links. Do not extend this
lineage; pivot to queue-aware placement instead.
```

### ACTIVE with follow-up
```markdown
**Outcome 2026-08-17 — coupling monotonically increases with shallower queues**

Measured additive R² 0.9881→0.9556 as queue depth decreases (27.6→50.8). The coupling
interaction term (`added_in_batch × exec_time`) is 11× weaker in deep queues. The lever
is **shallow queues + long-exec tasks**, not deep-queue magic. Unblocks `shallow_longexec_v1`.
```

### SUPERSEDED by another lineage
```markdown
**`SUPERSEDED` by `shallow_longexec_v1` 2026-08-17**

The deep-queue hypothesis was disproven (coupling decreases with depth). `shallow_v1` found
the real lever (shallow queues), and `shallow_longexec_v1` adds the execution-time multiplier
to amplify coupling. This lineage's results stand but should not be extended.
```

## Memory integration

If you discover something non-obvious (e.g., workload draws were unseeded, or ECT is not a ceiling),
flag it for memory capture. Example:

```
User: "Gate the results and update memory — we discovered workload draws aren't reproducible"

Agent:
1. Updates LINEAGES.md with FALSIFIED status for affected lineages
2. Calls /update-memory to save: "Workload draws were unseeded — traces are not reproducible
   from seed alone. Fixed 2026-08-18, all traces now use same RNG state. Impacts all datasets
   generated before 2026-08-18."
```

## Troubleshooting

**"I'm not sure if this is FALSIFIED or just needs more data"**:
- If the gate explicitly disproves the hypothesis (e.g., additive R² > 0.99, coupling fraction < target):
  mark FALSIFIED with proof.
- If the gate is incomplete but shows no progress toward the goal: still FALSIFIED (the approach didn't work).
- If data is incomplete (in-flight), run again on finished corpus and re-gate.

**"How do I know what to write in the outcome block?"**:
- State the metric that settles the question (the gate criterion)
- Explain why it settles it (one sentence)
- Point to the data/script that proves it
- Say what comes next (or what's now ruled out)

**"Should I update LINEAGES.md or write to memory?"**:
- LINEAGES.md: the *status* of the lineage (ACTIVE/FALSIFIED/SUPERSEDED) and why
- Memory: surprising findings that should inform *future* design (e.g., "deep queues reduce coupling")
- Do both if the finding is important enough to remember in 6 months

## Sample invocation

```
User: "Gate the mp_parity results — I think same-node edges are the issue"

Agent:
1. Reads LINEAGES.md, finds `mp_parity` entry
2. Asks: "Did the mp_arm test pass or fail?" → User points to log
3. Reads log, extracts: "same-node edges present in live, not in train. RTT 12x worse."
4. Drafts outcome:
   **`ACTIVE` — isolated root cause, unblocks `topology_transfer_v1`**
   Same-node edges are trained out but served in live inference. Fixed in 2026-08-19 by
   excluding same-node edges from training. See `test_train_serve_mp_parity.py` line 42.
5. Updates LINEAGES.md under mp_parity entry
6. Suggests capturing to memory: "GNN train/serve MP mismatch — same-node edges" (already exists)
```
