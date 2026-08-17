---
name: update-memory
description: Update memory/memory.md with strict versioning after completing work in this session. Use when the user says "update memory", "/update_memory", or asks to record progress/decisions into the project memory bank.
---

# Memory Update Protocol

You are the Project Archivist. Your goal is to update `memory/memory.md` based on the recent chat context.

**Strict Rules for Updating:**
1. **Read** the current `memory/memory.md`.
2. **Compare** the "Active Context" in the file vs. what was just accomplished in this session.
3. **Update** the file using targeted edits — do not rewrite the whole file if possible, only the sections that changed.

**Versioning Logic:**
- **Patch (0.0.x):** Small task completion, bug fix, or note added.
- **Minor (0.x.0):** New feature completed or major architecture decision.
- **Major (x.0.0):** Total refactor or project pivot.

**Execution Steps:**
1. Mark completed tasks in "Active Tasks" as done or remove them.
2. Add new tasks to "Active Tasks" (MAX 3 at a time).
3. Add any critical technical "Gotchas" to "Knowledge Graph" (only if reusable knowledge).
4. **Append** a new row to the "Changelog" table at the bottom.
5. **Update** the `**Version:**` and `**Last Updated:**` fields at the top — include hours and minutes, not only the date.

**Constraint Checklist (verify before finishing):**
- [ ] Did I bump the version number?
- [ ] Is the changelog row added?
- [ ] Are there NO paragraphs of text (bullet points only)?
- [ ] Did I preserve the Core Architecture section?

Proceed with the update using the Edit tool on `memory/memory.md`.
