---
name: experiment-gate
description: Record experiment outcomes into the research record — write the dated outcome into the lineage's node under docs/lineages/, update its row in the LINEAGES.md index, and file any transferable rule or gate-tool correction in its own home.
model: sonnet
effort: medium
tools: [Read, Write, Edit, Bash]
---

# Experiment Gate

Capture an experiment's outcome into the research record. Bridges the gap between
experimental runs (local or datalab) and the knowledge graph `LINEAGES.md` indexes.

## Where each fact goes — one fact, one home

`LINEAGES.md` is an **index only**: a status and a one-line outcome per lineage. The record
itself lives in nodes. Never write a narrative into the index.

| Fact | Home |
|---|---|
| The dated outcome, its numbers, its method | `docs/lineages/<lineage>.md` — append a `### <lineage> — <verdict> (YYYY-MM-DD)` section at the end, and add it to that file's newest-first "Record" list at the top |
| The lineage's status + one-line outcome | the row in `LINEAGES.md` |
| A pre-registration, screen, or findings doc | `docs/lineages/<lineage>/<name>.md`, linked from the node header |
| A rule that generalises past this lineage | `docs/lessons.md` (newest first) |
| A direction now closed | `docs/hard-stops.md`, **with the measurement that closed it** |
| A gate tool that was measuring the wrong thing | `docs/gates/gate-tools.md` — never inside the lineage that tripped over it |

Statuses: `ACTIVE` · `REGISTERED` (signed off, not yet run) · `CLOSED` (answered) ·
`SUPERSEDED` · `FAILED` / `FALSIFIED` · `SYNTHESIS` · `PAPER`.

## What you do

1. **Identify the lineage** being tested (or suggest one if unclear)
2. **Read the lineage's node** under `docs/lineages/` (and its row in `LINEAGES.md`) for prior context
3. **Collect outcome data**:
   - Did the gate PASS, FAIL, or get FALSIFIED?
   - What was the key finding or metric?
   - What does this mean for the research question?
   - Are there follow-up lineages blocked by this outcome?
4. **Append the dated outcome section** to `docs/lineages/<lineage>.md` — findings,
   metrics, implications, and pointers to supporting data (datasets, models, scripts) —
   and add it to that node's newest-first Record list
5. **Update the one-line row** in `LINEAGES.md`: status + what settles the question
6. **File anything that outlives the lineage** per the table above: a transferable rule in
   `docs/lessons.md`, a closed direction in `docs/hard-stops.md`, a gate-tool correction in
   `docs/gates/gate-tools.md`

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

3. **Read the current record**:
   - Read `docs/lineages/<lineage>.md`, and the row in `LINEAGES.md` (if it exists)
   - Note prior status, outcomes, and any blockers
   - Understand what this lineage was meant to prove/disprove

4. **Draft outcome block**:
   - **Status**: ACTIVE (and why), SUPERSEDED (by which lineage?), FALSIFIED (with proof), PAPER (if paper-frozen)
   - **Key metric**: The number or observation that settles the question
   - **Implication**: What does this mean for the next phase?
   - **Follow-up**: What lineage(s) does this unblock or block?
   - **Evidence**: Path to datasets, models, or gating script output

5. **Write it down**:
   - Append the outcome section to `docs/lineages/<lineage>.md` (create the node from a
     sibling's header if the lineage is new) and list it in that node's Record
   - Edit the lineage's row in `LINEAGES.md`: status + one-line outcome, nothing longer
   - Link to the nodes of lineages that depend on this result

6. **Promote what generalises**:
   - A finding that changes the mental model ("additive R² increases with queue depth")
     goes in `docs/lessons.md`
   - A gotcha for future work ("placements.jsonl can be truncated by in-flight dataset gen")
     goes in `docs/lessons.md`; if it is a gate that lied, `docs/gates/gate-tools.md`
   - A direction now closed goes in `docs/hard-stops.md` with its measurement

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

## Promoting a finding

If you discover something non-obvious (e.g., workload draws were unseeded, or ECT is not a
ceiling), it belongs in `docs/lessons.md` as well as the node. Example:

```
User: "Gate the results — we discovered workload draws aren't reproducible"

Agent:
1. Appends the outcome to each affected node and sets FALSIFIED in the index
2. Adds to `docs/lessons.md`: "Workload draws were unseeded — traces are not reproducible
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

**"Node, index, or lessons?"**:
- `docs/lineages/<lineage>.md`: the outcome itself — numbers, method, what settles it
- `LINEAGES.md`: one line — the *status* and what it means. Never a narrative
- `docs/lessons.md`: the part that would change a decision on a *different* lineage
- Do all three when the finding is important enough to matter in 6 months

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
5. Updates the `mp_parity` row in `LINEAGES.md` to match
6. Appends the section to `docs/lineages/mp_parity.md`, updates the index row, and adds
   the transferable half ("a checkpoint's train-time graph and its serve-time graph must be
   compared, not assumed") to `docs/lessons.md`
```
