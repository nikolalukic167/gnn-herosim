# History rewrite (2026-10-28)

On 2026-10-28 every branch was merged into `main` and the full history was rewritten:
agent attribution trailers were removed from commit messages and agent/placeholder
identities were mapped to the repository owner. Upstream HeROsim authors are unchanged.

**Every commit hash cited before 2026-10-28 is an old hash** — in lineage nodes, result
JSONs (`run_provenance.code`), checkpoint sidecars, and datalab worktree names. Translate
with `commit_hash_map_2026-10-28.tsv` (full 40-char old → new; prefix-match a short hash):

```bash
grep -i "^381d6f98" docs/notes/commit_hash_map_2026-10-28.tsv
```

Eight commits that existed only in the datalab clone were replayed onto the new history
afterwards; their rows are at the end of the map.

Two old commits can share one new hash: copies that differed only in a placeholder author
email became identical. Trees are unchanged — the rewrite touched metadata only.
