"""Record the lscpu model of every RUNNING chain link (login-node safe: one `srun --overlap lscpu` per job).

  uncut_lscpu_capture.py --names uarm2-train,uarm2-gpu-train --out <cpu_captures.jsonl>

Appends {"job" (raw id), "node", "cpu", "source": "srun --overlap lscpu, <UTC>"} once per job; a job already captured is skipped, a failed
srun is reported and retried on the next call, never guessed from the node name. A link that ends before it is caught has no entry, and the
md5 writer's prov.json then says "lscpu model not captured".
"""
import argparse
import datetime
import json
import os
import subprocess


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["job"] for l in open(a.out) if l.strip()}
    q = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-n", a.names, "-t", "RUNNING", "-h", "-o", "%i %N"], capture_output=True, text=True, check=True)
    new = failed = 0
    for line in q.stdout.splitlines():
        job, node = line.split()
        if job in done:
            continue
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        r = subprocess.run(["srun", f"--jobid={job}", "--overlap", "-n1", "lscpu"], capture_output=True, text=True, timeout=90)
        model = next((l.split(":", 1)[1].strip() for l in r.stdout.splitlines() if l.startswith("Model name")), None)
        if r.returncode or not model:
            print(f"FAILED {job} on {node}: rc {r.returncode} {r.stderr.strip()[:120]}")
            failed += 1
            continue
        with open(a.out, "a") as fh:
            fh.write(json.dumps({"job": job, "node": node, "cpu": model, "source": f"srun --overlap lscpu, {now}"}) + "\n")
        new += 1
    print(f"captured {new}, failed {failed}, already {len(done)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
