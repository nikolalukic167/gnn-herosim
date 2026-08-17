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
* *(Check: Do these decisions align with the patterns in @memory/memory.md?)*

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

Immediate Task: [First Next Step] Please review @memory/memory.md and @[Relevant_Files] then continue.


**Constraints:**
* Be extremely concise. Bullet points only.
* Do not summarize "chitchat"—only technical facts.
* If we are mid-debugging, paste the last error message in "Known Issues".
* **CRITICAL:** Ensure all technical summaries are consistent with the architectural rules in `@memory/memory.md`.