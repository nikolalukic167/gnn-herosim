#!/bin/bash
# physics_audit_v1 I8: one cell, same seed, two runs (plus a third with HS=1, i.e. another PYTHONHASHSEED, informational; the runner must pass it through).
# usage: run_determinism.sh <runner> <topo> <rung> <events> <outdir>
#   <runner> is a script taking: topo window rung policy out.json [max_events], with the R1 environment set
#   (the scratchpad r1run.sh is the reference). Prints one line per run.
runner=$1; topo=$2; rung=$3; ev=$4; out=$5; mkdir -p "$out"
for tag in a b; do
  "$runner" "$topo" g0 "$rung" peer_greedy_network_cd "$out/${topo}_${rung}_$tag.json" "$ev" > "$out/${topo}_${rung}_$tag.log" 2>&1
  echo "$topo $rung $tag rc=$?"
done
HS=1 "$runner" "$topo" g0 "$rung" peer_greedy_network_cd "$out/${topo}_${rung}_hs1.json" "$ev" > "$out/${topo}_${rung}_hs1.log" 2>&1
echo "$topo $rung hs1 rc=$?"
