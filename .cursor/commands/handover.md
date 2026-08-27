---
description: Summarize the current session state to hand over to a new agent session.
---

# Session Handover Protocol

You are an expert technical lead preparing a handover report for the next developer (who is also an AI). Your goal is to compress the current session state so the next agent can resume work immediately without reading the full history.

**Format the output exactly as follows (Markdown):**

## 🚀 Session Handover
**Status:** [Stopped at Step X / Feature Complete / Debugging]

### 1. Context & Goal
* **Objective:** [One sentence on what we were trying to do]
* **Progress:** [What was successfully completed]

### 2. Technical Decisions Made
* [Key architectural choice or pattern established]
* [Why we chose X over Y (if relevant)]
* *(Check: do these decisions align with LINEAGES.md and docs/lessons.md?)*

### 3. Current Code State
* **Modified Files:** `[List files touched]`
* **Pending Changes:** [Code that is written but not tested/committed, or broken]
* **Known Issues:** [Any bugs or errors currently active]

### 4. Next Steps (Immediate Action Plan)
1.  [First task for the next session]
2.  [Second task...]

### 5. "Restore" Prompt
*Copy and paste this into the next chat to resume:*
[CONTEXT RESTORE] I am resuming a previous session.

Goal: [Restate Goal]

Current State: [Brief State]

Immediate Task: [First Next Step] Please review @LINEAGES.md and @[Relevant_Files] then continue.


**Constraints:**
* Be extremely concise. Bullet points only.
* Do not summarize "chitchat"—only technical facts.
* If we are mid-debugging, paste the last error message in "Known Issues".
* **CRITICAL:** Ensure all technical summaries are consistent with `LINEAGES.md`.
- **A handover is ephemeral and is NEVER committed.** Write it to the session scratchpad or
  print it to chat — do not create a `HANDOVER*.md` in the repo. Three such files were
  retired on 2026-08-27 for drifting out of agreement with `LINEAGES.md` while claiming the
  same facts. Anything in a handover that is still true a week later belongs in a
  `docs/lineages/` node, `docs/lessons.md`, or `docs/gates/gate-tools.md` — put it there.
