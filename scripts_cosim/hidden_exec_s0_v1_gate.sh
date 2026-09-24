#!/bin/bash
# hidden_exec_s0_v1 -- live headroom screen. docs/lineages/hidden_exec_s0_v1.md.
# 64 runs: {table, oracle} x 16 + 16 unsaturated_edge_v1 environments, all under hidden_node_v1.
# usage: hidden_exec_s0_v1_gate.sh <config_dir> <workload_dir> <out_dir> [parallel=16]
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
CFG_DIR=$1; WL_DIR=$2; OUT_DIR=$3; PAR=${4:-16}
SELECTION=${SELECTION:-$REPO/simulation_data/unsaturated_edge_v1/selected.json}
[[ -f "$SELECTION" ]] || { echo "FAIL LOUD: $SELECTION missing" >&2; exit 1; }
mkdir -p "$OUT_DIR"
PY=${HEROSIM_PY:-pipenv run python3}

TASKS=$(python3 - "$SELECTION" <<'PY'
import json, sys
sel = json.load(open(sys.argv[1]))
if sel.get("verdict") != "DESIGN-READY":
    raise SystemExit(f"FAIL LOUD: selection verdict {sel.get('verdict')!r}")
for nc in ("40", "80"):
    envs = sel["rungs"][nc]["environments"]
    if len(envs) != 16:
        raise SystemExit(f"FAIL LOUD: C{nc} has {len(envs)} environments")
    for _r, t, w in envs:
        for arm in ("table", "oracle"):
            print(nc, t, w, arm)
PY
)

run_one() {
  local nc=$1 topo=$2 w=$3 arm=$4
  local wl=drainable_${w}_n50000; [[ $w == w0 ]] && wl=drainable_f4000_n50000
  local name="cc${nc}s${topo}__${w}__${arm}"
  local summary="$OUT_DIR/$name.summary.json"
  [[ -f "$summary" ]] && { echo "[skip] $name"; return 0; }
  (
    export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2
    export PYTHONHASHSEED=0 HEROSIM_GNN_DEVICE=cpu SIM_FORCE_FULL_STATS=1 OMP_NUM_THREADS=1
    export PYTHONPATH="$REPO"
    export HEROSIM_EXEC_PHYSICS=hidden_node_v1 HEROSIM_EXEC_SEED=$topo
    [[ $arm == oracle ]] && export HEROSIM_PG_EXEC_KNOWLEDGE=oracle
    $PY "$REPO/src/executesimulation.py" --config "$CFG_DIR/cc${nc}s${topo}.json" \
      --workload "$WL_DIR/$wl.json" --policy peer_greedy_selfpredict_network \
      --output "$OUT_DIR/$name.raw.json" > "$OUT_DIR/$name.log" 2>&1
  )
  python3 - "$OUT_DIR/$name.raw.json" "$summary" "$name" "$nc" "$topo" "$w" "$arm" <<'PY'
import json, sys
raw, summary, name, nc, topo, w, arm = sys.argv[1:8]
doc = json.load(open(raw)); st = doc.get("stats") or doc; prov = doc.get("run_provenance") or {}
if int(st.get("num_tasks") or 0) != 50000:
    raise SystemExit(f"FAIL LOUD: {name} num_tasks={st.get('num_tasks')!r}")
want = "oracle" if arm == "oracle" else "table"
if prov.get("exec_physics") != "hidden_node_v1" or prov.get("pg_exec_knowledge") != want \
        or int(prov.get("exec_seed", -1)) != int(topo):
    raise SystemExit(f"FAIL LOUD: {name} provenance {prov.get('exec_physics')}/{prov.get('pg_exec_knowledge')}/{prov.get('exec_seed')}")
c = st.get("schedulerCounters") or {}
if int(c.get("pg_decisions") or 0) != 50000 or int(c.get("pg_lookahead_priced") or 0) == 0:
    raise SystemExit(f"FAIL LOUD: {name} rule instrument off: {c.get('pg_decisions')}/{c.get('pg_lookahead_priced')}")
out = {k: st.get(k) for k in ("total_rtt", "averageElapsedTime", "averageQueueTime",
                              "averageExecutionTime", "averagePeerExchangeTime", "totalPeerRendezvousWait")}
out.update(name=name, rung=f"C{nc}", topology=int(topo), window=w, arm=arm,
           code=prov.get("code"), exec_physics=prov.get("exec_physics"), exec_seed=prov.get("exec_seed"))
json.dump(out, open(summary + ".partial", "w"), indent=1)
PY
  mv "$summary.partial" "$summary"
  rm -f "$OUT_DIR/$name.raw.json"
  echo "[done] $name"
}
export -f run_one
export OUT_DIR CFG_DIR WL_DIR REPO PY
echo "$TASKS" | xargs -P "$PAR" -L 1 bash -c 'run_one "$@"' _
n=$(ls "$OUT_DIR"/*.summary.json | wc -l)
(( n == 64 )) || { echo "FAIL LOUD: $n/64 summaries" >&2; exit 1; }
echo "=== hidden_exec_s0_v1 gate complete: 64/64 ==="
