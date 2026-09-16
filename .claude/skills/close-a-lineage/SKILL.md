---
name: close-a-lineage
description: The checklist for moving a lineage to CLOSED (or FALSIFIED, FAILED, PARKED, SUPERSEDED) without leaving the record inconsistent. Load whenever a gate has been read and a lineage is being closed, retired, parked or superseded, and before editing a LINEAGES.md row or a node's Status header. Closing is the single moment when the index, the node header, the standing answer and the hard stop all drift apart — every status drift and every oversized index row in the 2026-09-16 audit entered here. Not a guide to running or reading a gate (see the experiment-gate agent) — this is what to write down once the answer is known.
---

# Closing a lineage

Closing is cheap to do badly. The 2026-09-16 record audit found **twelve** lineages whose
index row and node header disagreed, **44 KB** of node narrative pasted into an index that
says in its own preamble that it is an index only, and a flagship node whose head still led
with a result that three later lineages had reversed. Every one of those entered at close
time, because closing touches four files and it is easy to touch two.

Work the list in order. Each step names the defect it prevents.

## 1. Write the outcome into the node — the record lives here

Append a dated section to `docs/lineages/<lineage>.md`: the verdict, the metric that
settles it, the method, and what it unblocks or rules out. Add it to the node's
newest-first Record list.

This is the only place the full narrative goes.

## 2. Rewrite the node's head — do not append to it

The head is everything above the first dated section. It is **12% of node bytes and the
only part most sessions read**, so a stale head is a wrong answer delivered by default.

- Update `**Status:**` to the new status, and keep the registration fact next to it:
  `**Status:** \`CLOSED\` (2026-09-15) — **VERDICT-NAME**. Registered 2026-09-14; every bar
  below was signed before its data.`
- Rewrite the `**Outcome.**` paragraph so it states what is true **now**. If a later
  lineage reversed or qualified this one, the head says so, in the head, not 40 KB down.
- Carry the caveats a reader must not quote without. `peer_affinity_v1`'s head omitted
  "offline only" and "reverses live" for four days.

**The failure this prevents:** six nodes closed on 2026-09-15 or later still said
`REGISTERED` in their own header, because the outcome was appended and the head was not
touched.

## 3. Compress the index row to one line

`LINEAGES.md` gets a status and a **one-line outcome** — under 400 bytes, no embedded
tables, no per-seed numbers, no method. If you cannot say it in one line, the extra belongs
in the node.

Move the row into the section matching its new status: a `CLOSED` row does not stay under
`## Open — the current program`. Thirteen of them did.

**Before deleting anything from a row**, confirm the node already carries it. Numbers that
existed only in the index get migrated to the node first; that is what the
`### Carried over from the index row` sections are.

## 4. File what outlives the lineage — in its own home, not here

| What | Where |
|---|---|
| A rule that would have changed a decision *before* it was learned | `docs/lessons.md`, as its own `##` section |
| A direction now closed | `docs/hard-stops.md`, **with the measurement that closed it** |
| A gate that was measuring the wrong thing | `docs/gates/gate-tools.md` |

Do not write the same paragraph into two of these. One measurement was found in four
files at once. If a fact fits two homes, pick the one whose *question* it answers and link
from the other.

**Check the stop and the lesson do not contradict.** A lesson written in August asserted a
conclusion that `hard-stops.md` later recorded as falsified, and both sat in the tree for
weeks.

## 5. Update the standing answer if the program-level story moved

If this close changes what is true about the research question, rewrite CLAUDE.md's
`**Where the research question stands (rewritten ...)**` block and bump its date.
**Rewrite it — never append a dated paragraph.** That block reached 12 KB by accretion and
then stopped being maintained, which is worse than either.

## 6. Run the check

```bash
PIPENV_IGNORE_VIRTUALENVS=1 pipenv run python3 -m pytest tests/test_record_hygiene.py -q
```

It enforces steps 2, 3 and 5 mechanically: row length, row-vs-node status agreement, every
node has a row, head freshness, the CLAUDE.md stamp, and that cited scripts still exist.
It runs in about two seconds. **The routing rule was written in four places and violated in
all four; this test is the thing that actually holds it.**

## What "closed" does not mean

A lineage closes on a **live gate**, never on an offline read (CLAUDE.md rule 6). An
offline screen may order the work; a NO-GO on an offline bar is recorded and the registered
live gate still runs. If you are closing on an offline number, you are not closing.
