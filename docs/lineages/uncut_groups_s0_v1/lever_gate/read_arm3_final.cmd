# The exact command that produced lever_gate/read_arm3_final.json (md5 7acc051fd90c2e9ba02794fb062761e3), run on datalab 2026-10-17 with the
# reader of rp/uncut-lever ba344dc3 (worktree ~/gnn-herosim-wt/s4_reader_ba344dc3). Re-running it with another reader or another --out only
# changes the output only if the reader's logic changed or the arguments embedded in the JSON differ (see the last lines).
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
U=/share/nikola.lukic/uncut_groups
R=$HOME/gnn-herosim-wt/s4_reader_ba344dc3
A=$U/final/reads/read_arm1b_curba344_nocompare.json     # S6's Arm 1b final read, md5 301b11c37f3cda559ff392234303e53d
TOPOS=$(seq -s, 19537 19548)                             # 19537,19538,...,19548 (sha256 553ef17de81af2bd61a72b716e8f48163c6b0e8f8050dd71323183750b29a5d7)
cd $R
python3 scripts_cosim/uncut_arm_gate_read.py \
  --gate $U/final/cells_arm3,$U/arm1_k1_bar_d6ae87c2 \
  --topos $TOPOS --topos-sha256 $(echo -n $TOPOS | sha256sum | cut -d" " -f1) \
  --pin d6ae87c2 --primary ra_gnn_eng_arm3_nosplit \
  --checkpoint-md5 arm3=$R/experiments/uncut_arm3_ckpt_md5.json \
  --repo $HOME/gnn-herosim --w-pins 1847a484 \
  --gate-change bf79865a,e8c87702,114cc0e7,8dfd7e0c,3df0cce6 \
  --after-read $A \
  --out $U/final/reads/read_arm3_curba344_nocompare.json
# no --twin, no --mlp, no --compare, default --suffix _k100 and --cotenancy required.
# The JSON embeds the --after-read path and md5 (sequence.after_read, after_read_md5 = 301b11c3...), --pin, --gate-change, --topos and the w-pin shas; a different
# --after-read file (another path or another content, e.g. a later regenerated Arm 1b read) or another --gate-change / --w-pins string changes the bytes. The cell
# directories themselves are not embedded.
