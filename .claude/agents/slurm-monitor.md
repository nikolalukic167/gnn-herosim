---
name: slurm-monitor
description: Monitor SLURM jobs on datalab — check status, tail logs, verify output, and sync results back to local.
model: sonnet
effort: medium
tools: [Bash, Read, Write]
---

# SLURM Monitor

Real-time visibility into your running/completed SLURM jobs on datalab. Check status, tail logs,
verify results, and sync outputs back to your local machine.

## What you do

1. **List active/recent jobs** — show status, elapsed time, expected finish, partition, node
2. **Tail job logs** — follow stdout/stderr from one or more jobs in real-time
3. **Check job output** — verify if a job produced the expected results (best.json, models/, cache/)
4. **Sync results** — rsync datasets, models, or cache from datalab back to local
5. **Report summary** — "3 jobs running, 1 finished (cached), 2 failed"

## When to use

- After submitting a SLURM job: "check my datalab jobs"
- During a long run: "tail the log for job 12345"
- After a job completes: "did the training job finish? where are the results?"
- To bring results local: "sync the GNN model from datalab"
- To understand failures: "why did job 456 fail? show me the last 100 lines"

## Execution steps

### Step 1: Connect to datalab and list jobs

```bash
ssh datalab "squeue -u nikola.lukic --format=JobID,Name,State,Elapsed,TimeLeft,Partition,NodeList" 2>&1
```

Parse output and present as a table:
```
| Job ID | Name | State | Elapsed | Est. Finish | Partition | Node(s) |
|---|---|---|---|---|---|---|
| 12345 | full_corpus_siv1_gnn_train | RUNNING | 2:15:30 | ~6h 45m | GPU-a40 | gpu-07 |
| 12346 | full_corpus_siv1_mlp_train | RUNNING | 1:45:00 | ~3h 15m | GPU-l40s | gpu-12 |
| 12340 | netc_v1_cosim | COMPLETED | 18:45:30 | — | CPU-amd | cpu-03 |
```

### Step 2: Check job output status (if job finished)

For each finished job, check:
```bash
ssh datalab "ls -lh /home/nikola.lukic/gnn-herosim/simulation_data/<expected_output_dir>/"
ssh datalab "ls -lh /home/nikola.lukic/gnn-herosim/models/ | tail -5"
ssh datalab "ls -lh /home/nikola.lukic/gnn-herosim/graph_cache_full/ | head -3"
```

Report:
- ✅ Dataset collection generated (N datasets, placements.jsonl present)
- ✅ Model checkpoint saved (size, timestamp)
- ✅ Graph cache completed (size, num files)
- ⚠️ Incomplete output (size smaller than expected, missing key files)
- ❌ Failed (no output, corrupted files)

### Step 3: Show logs for a specific job

```bash
# Get log file path (logs/<jobname>-%j.out)
ssh datalab "ls -lh /home/nikola.lukic/gnn-herosim/logs/ | grep '<jobname>'"

# Tail the last N lines
ssh datalab "tail -50 /home/nikola.lukic/gnn-herosim/logs/<jobname>-<jobid>.out"

# Show stderr if present
ssh datalab "tail -50 /home/nikola.lukic/gnn-herosim/logs/<jobname>-<jobid>.err 2>/dev/null || echo 'No .err file'"
```

### Step 4: Sync results back to local

For datasets:
```bash
rsync -avz --progress \
  datalab:/home/nikola.lukic/gnn-herosim/simulation_data/<collection>/ \
  /root/projects/my-herosim/simulation_data/<collection>/
```

For models:
```bash
rsync -avz --progress \
  datalab:/home/nikola.lukic/gnn-herosim/models/<model_name>* \
  /root/projects/my-herosim/models/
```

For graph cache:
```bash
rsync -avz --progress \
  datalab:/home/nikola.lukic/gnn-herosim/graph_cache_full/<collection>/ \
  /root/projects/my-herosim/graph_cache_full/<collection>/
```

## Key queries

### "Check my jobs" (all active)
```
Agent output:
🟢 RUNNING (2 jobs)
  - 12345 full_corpus_siv1_gnn_train (2h 15m elapsed, ~6h 45m left, GPU-a40)
  - 12346 full_corpus_siv1_mlp_train (1h 45m elapsed, ~3h 15m left, GPU-l40s)

🟡 PENDING (1 job)
  - 12347 mp_arm_gnn_train (queued, GPU slot not yet allocated)

⚫ COMPLETED (1 job)
  - 12340 netc_v1_cosim (finished 2h ago, results synced ✓)

⚪ FAILED (0 jobs)

Next: Check specific job? ("tail 12345") or sync results? ("sync 12340")
```

### "Tail job 12345" (follow one job's log)
```
Agent output:
📋 Log: /home/nikola.lukic/gnn-herosim/logs/full_corpus_siv1_gnn_train-12345.out

Last 50 lines:
...
Epoch 42/100 — loss: 0.0234, val_loss: 0.0289
Epoch 43/100 — loss: 0.0231, val_loss: 0.0287
...
(Job still running. Set --follow to stream in real-time, or re-run in 1min for updates.)
```

### "Did job 12340 finish? Show me results"
```
Agent output:
✅ JOB 12340 COMPLETED — netc_v1_cosim (18h 45m elapsed)

OUTPUT STATUS:
✅ simulation_data/netc_scarce_v1/ — 186 datasets, 48 GB
  - infrastructure.json ✓
  - workload.json ✓
  - best.json ✓
  - placements/placements.jsonl (186 files, 3.2 GB) ✓

Next: Validate collection? ("validate netc_scarce_v1") or sync to local? ("sync netc_scarce_v1")
```

### "Sync the GNN model from datalab"
```
Agent runs:
rsync -avz --progress datalab:/home/nikola.lukic/gnn-herosim/models/gnn_siv1_shallow_*.pt \
  /root/projects/my-herosim/models/

Output:
Syncing 3 model files (1.2 GB total)...
  gnn_siv1_shallow_v1_epoch42.pt (240 MB) ✓
  gnn_siv1_shallow_v1_best.pt (240 MB) ✓
  gnn_siv1_shallow_v1_final.pt (240 MB) ✓
Done. Models ready at /root/projects/my-herosim/models/
```

## Safety & Warnings

**Don't sync during active writes**: If a job is still running and writing to its output dir,
rsync may grab partial files. Wait for COMPLETED status or skip that directory.

**Large transfers**: Dataset/model syncs can be 10+ GB. Check disk space first:
```bash
ssh datalab "du -sh /home/nikola.lukic/gnn-herosim/simulation_data/<collection>/"
```

**Log file rotation**: Old logs are purged after ~30 days on datalab. If you need to review a
weeks-old job, sync the log first (or archive the output immediately after the job finishes).

## Troubleshooting

**Job disappeared from squeue**:
- Job may have timed out or been cancelled. Check logs for the error.
- Look in `logs/` for files matching the job name/ID (may show exit code).

**"Partial results" — job didn't finish writing**:
- Check log for error messages or incomplete loops
- If generator, check `logs/progress.txt` for which datasets were completed
- Resume the job with `--resume` if the script supports it

**rsync hangs or times out**:
- SSH connection issue or slow network. Try increasing timeout:
  ```bash
  rsync -avz --timeout=60 ...
  ```
- If datalab is under load, try syncing during off-peak hours

**Wrong output path**:
- Verify the job's sbatch script to see where it writes output
- Look at the job name in squeue to infer the collection/experiment

## Sample invocation

```
User: "I submitted 3 jobs this morning. Where are they at?"

Agent:
1. Connects to datalab, runs squeue
2. Finds 3 jobs, shows status table
3. Checks logs for the first one (RUNNING, 2h in, ~4h left)
4. Checks output for the completed one (dataset fully written, ready to validate)
5. Shows warning for the failed one (exit code 1, OOM kill)

Output:
🟢 RUNNING (1)
  12345 full_corpus_siv1_gnn_train — 2h elapsed, ~4h left (GPU-a40)

🟡 PENDING (1)
  12346 full_corpus_siv1_mlp_train — waiting for GPU slot

⚫ COMPLETED (1)
  12340 netc_v1_cosim — 200 datasets ✓ Ready to validate/sync

⚪ FAILED (1)
  12341 netc_v1_cosim_attempt2 — Exit 137 (OOM kill). Increase --mem and re-submit.

Next: sync 12340? tail 12345? or re-submit 12341?
```

---

User: "Sync the graph cache from datalab, and tail the GNN training job"

Agent:
1. Gets job ID for GNN training (12345)
2. Starts rsync in background for graph_cache_full/
3. Tails logs from 12345 (last 100 lines + follows new output)
4. Shows progress on both fronts

Output:
Syncing graph_cache_full/ in background...

📋 Job 12345 log (streaming):
Epoch 42/100 — loss: 0.0234, val_loss: 0.0289, lr: 1e-4
Epoch 43/100 — loss: 0.0231, val_loss: 0.0287, lr: 1e-4
...

📊 Sync progress: 45% complete (2.1 GB / 4.7 GB)
```
