#!/bin/bash
# scale_d_v1 launcher wrapper (docs/lineages/scale_d_v1.md, v3). Wraps ~/llf_cd_launch_48h.sh (S4's generic cell launcher: option D, cap200 inputs, nosplit k100, r1a_gate.sbatch at the
# pinned worktrees, one unpinned job per (window, rung, group), a 44-job cap check and a refusal to reuse an output dir) with PAR / CELL_MEM from the environment.
# DRY (the default) prints the sbatch lines and submits nothing; GO=1 submits.
#   scale_d_launch.sh stage    <SC>                    build the IN dir $U/scale_d/<SC>/in from the minted cap200 inputs (llf_cd_in.sh); no job
#   scale_d_launch.sh smoke    <SC>                    SC = SA|SB: g0 moderate, bar + HEFT (s0 job) and gnn / wide MLP / MLP seed 1 (static_s1 job), one cell at a time, 24 h per cell
#   scale_d_launch.sh bar      <SC> <ids,csv>          SC = A|B: the bar-only admission wave, g0 g1 x moderate heavy  (these are also the comparison cells)
#   scale_d_launch.sh heft     <SC> <ids,csv>          SC = A|B: HEFT, g0 g1 x moderate heavy
#   scale_d_launch.sh learned  <SC> <ids,csv>          SC = A|B: gnn / wide MLP / MLP, seeds 1-3 (one job per seed, window, rung); ids = the ADMITTED ids
#   scale_d_launch.sh quick    <SC>                    SC = QA|QB (disposable quick-look ids, never study ids): gnn s1, wide MLP s1, bar s0, g0 x moderate heavy -> $U/scale_d/quicklook/<SC>
#   scale_d_launch.sh identity                         the two identity cells at the serving pins (parent topology 30327, g0 moderate); output under $U/scale_d/identity
# env: PAR (default 8) and CELLMEM (default 16G) per cell for bar / heft / learned: set both from the smoke's MaxRSS (learned and heft refuse to run unset); EXCL; CPUS
set -euo pipefail
KIND=${1:?stage|smoke|bar|heft|learned|identity}; SC=${2:-}; IDS=${3:-}
U=/share/nikola.lukic/uncut_groups; SD=$U/scale_d; GO=${GO:-0}; DRY=1; [ "$GO" = 1 ] && DRY=0
BARPIN=$(git -C $HOME/gnn-herosim-wt/s6_lk_0ec7cdcb rev-parse HEAD); STATICPIN=$(git -C $HOME/gnn-herosim-wt/s6_lk_02668ecf rev-parse HEAD)
[[ $BARPIN == 0ec7cdcb* && $STATICPIN == 02668ecf* ]] || { echo "FAIL LOUD: pin worktrees"; exit 2; }
L=$SD/launcher.sh
mkdir -p $SD
sed 's/PAR=8 CELL_MEM=16G/PAR=${PAR_N:-8} CELL_MEM=${CELLMEM:-16G}/' $HOME/llf_cd_launch_48h.sh > $L.tmp
grep -q 'PAR=${PAR_N:-8}' $L.tmp || { echo "FAIL LOUD: launcher sed did not apply"; exit 2; }
mv $L.tmp $L
run() { env ROOTS_FILE=$ROOTS OUT_ROOT=$OUT PIN=$BARPIN PIN_STATIC=$STATICPIN DRY=$DRY "$@" bash $L; }
need_sc() { [[ $SC =~ ^(A|B)$ ]] || { echo "FAIL LOUD: scale A|B"; exit 2; }; [[ -d $SD/$SC/in && -n $IDS ]] || { echo "FAIL LOUD: stage $SC first and give ids"; exit 2; }; }
case $KIND in
  stage)
    [[ $SC =~ ^(A|B|SA|SB|QA|QB)$ ]] || { echo "FAIL LOUD: scale"; exit 2; }
    [ "$GO" = 1 ] || { echo "DRY: would run llf_cd_in.sh $SD/$SC/in $SD/$SC/gate_inputs_cap200/inputs_group"; exit 0; }
    bash $HOME/llf_cd_in.sh $SD/$SC/in $SD/$SC/gate_inputs_cap200/inputs_group;;
  smoke)
    [[ $SC =~ ^(SA|SB)$ && -d $SD/$SC/in ]] || { echo "FAIL LOUD: SA|SB staged"; exit 2; }
    SID=$(cat $SD/$SC/ids.txt); ROOTS=$SD/$SC/roots.txt; OUT=$SD/$SC/cells
    printf 'smoke %s %s\n' $SD/$SC/in $SID > $ROOTS
    run JN=sdsmk WINDOWS=g0 RUNGS=moderate GROUPS_ONLY="s0 static_s1" S0_ARMS=cd_ib0_inflightfull_k100,heft LIMIT=86400 PAR_N=1 CELLMEM=64G;;
  smokebar)
    [[ $SC =~ ^(SA|SB)$ && -d $SD/$SC/in ]] || { echo "FAIL LOUD: SA|SB staged"; exit 2; }
    SID=$(cat $SD/$SC/ids.txt); ROOTS=$SD/$SC/roots.txt; OUT=$SD/$SC/cells_s0
    printf 'smoke %s %s\n' $SD/$SC/in $SID > $ROOTS
    run JN=sdsmk WINDOWS=g0 RUNGS=moderate GROUPS_ONLY="s0" S0_ARMS=cd_ib0_inflightfull_k100,heft LIMIT=86400 PAR_N=1 CELLMEM=64G;;
  bar|heft|learned)
    need_sc
    [ "$KIND" = bar ] || { : "${PAR_N:?FAIL LOUD: set PAR_N from the smoke}"; : "${CELLMEM:?FAIL LOUD: set CELLMEM from the smoke MaxRSS}"; }
    ROOTS=$SD/$SC/roots_$KIND.txt; OUT=$SD/$SC/cells_$KIND
    printf 'study %s %s\n' $SD/$SC/in $IDS > $ROOTS
    case $KIND in
      bar)     run JN=sd$SC-bar WINDOWS="g0 g1" GROUPS_ONLY="s0" S0_ARMS=cd_ib0_inflightfull_k100 LIMIT=${LIMIT:-43200} PAR_N=${PAR_N:-8} CELLMEM=${CELLMEM:-16G};;
      heft)    run JN=sd$SC-heft WINDOWS="g0 g1" GROUPS_ONLY="s0" S0_ARMS=heft_k100 LIMIT=${LIMIT:-43200};;
      learned) run JN=sd$SC-ln WINDOWS="g0 g1" GROUPS_ONLY="static_s1 static_s2 static_s3" LIMIT=${LIMIT:-43200};;
    esac;;
  quick)
    [[ $SC =~ ^(QA|QB)$ && -d $SD/$SC/in ]] || { echo "FAIL LOUD: QA|QB staged"; exit 2; }
    : "${PAR_N:?FAIL LOUD: set PAR_N from the smoke}"; : "${CELLMEM:?FAIL LOUD: set CELLMEM from the smoke MaxRSS}"
    ROOTS=$SD/$SC/roots_quick.txt; OUT=$SD/quicklook/$SC
    printf 'quick %s %s\n' $SD/$SC/in "$(cat $SD/$SC/ids.txt | tr ' ' ',')" > $ROOTS
    run JN=sdq-$SC WINDOWS=g0 GROUPS_ONLY="s0 static_s1" S0_ARMS=cd_ib0_inflightfull_k100 STATIC_ARMS=ra_gnn_eng_combfl_linkstatic_e40_nosplit_k100,ra_mlp_wide_combfl_linkstatic_e40_nosplit_k100 LIMIT=${LIMIT:-43200};;
  identity)
    mkdir -p $SD/identity
    ROOTS=$SD/identity/roots.txt; printf 'idn %s 30327\n' $U/xs/in_llf_cd > $ROOTS
    OUT=$SD/identity/cells_bar;  run JN=sdidb WINDOWS=g0 RUNGS=moderate GROUPS_ONLY="s0" S0_ARMS=cd_ib0_inflightfull_k100 LIMIT=43200
    OUT=$SD/identity/cells_gnn;  run JN=sdidg WINDOWS=g0 RUNGS=moderate GROUPS_ONLY="static_s1" STATIC_ARMS=ra_gnn_eng_combfl_linkstatic_e40_nosplit_k100 LIMIT=43200;;
  *) echo "FAIL LOUD: kind $KIND"; exit 2;;
esac
