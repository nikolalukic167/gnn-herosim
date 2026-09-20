"""Regenerate every figure of paper/professor_summary/summary.tex from data/ and the record.

    cd /root/projects/my-herosim
    PIPENV_IGNORE_VIRTUALENVS=1 PYTHONPATH=$PWD pipenv run python3 paper/professor_summary/make_figures.py

Every number drawn here is either read from a gate summary under data/results/ (fetched from
datalab, stripped to scalars) or copied from a lineage node with the node named in the comment.
The statistics are computed with the registered readers (scripts_cosim/unsaturated_edge_v1_read.py,
scripts_cosim/unsaturated_scale_v2_read.py) so the figures show exactly the registered numbers.
Figures whose data has not landed yet (9, 11) are skipped with a message, never faked.
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA, FIG = HERE / "data", HERE / "figures"
FIG.mkdir(exist_ok=True)

from scripts_cosim.unsaturated_edge_v1_read import (  # noqa: E402
    E_ARMS_BY_RUNG, E_RUNGS, E_TOPOLOGY_CANDIDATES, E_WINDOWS, checkpoint_stats, read_m0,
    select_topologies,
)
from scripts_cosim.unsaturated_edge_v1_gate_read import screen_shares, study_tables  # noqa: E402
from scripts_cosim.unsaturated_scale_v2_read import M_TOPOLOGY_CANDIDATES  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "figure.dpi": 100, "savefig.dpi": 180})
C = {"reactive": "#444444", "gnnedge0": "#c0392b", "peeronly": "#e67e22", "mpoff": "#2980b9",
     "gnn": "#8e44ad", "random": "#7f8c8d", "ect": "#27ae60", "rpi": "#e74c3c", "xavier": "#2ecc71",
     "pyngFpga": "#3498db"}
LABEL = {"be1670_gnnedge0": "gnnedge0", "1670_peeronly": "peeronly", "1670_mpoff": "mpoff",
         "1670_gnn": "gnn (sum)", "516_mpoff": "mpoff (516)"}


def _load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(str(DATA / pattern)))]


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    print(f"[fig] {name}")


# --- Fig 1: a 6-server / 40-client cell -------------------------------------------------------

def fig1_topology():
    from src.generate_infrastructure import generate_network_topology_deterministic
    cfg = json.load(open(DATA / "configs/cc40s9001.json"))
    types = list(cfg["pci"].keys())
    nodes = []
    for i in range(int(cfg["nodes"]["client_nodes"]["count"])):
        nodes.append({**cfg["pci"][types[i % len(types)]]["specs"],
                      "node_name": f"client_node{i}", "type": types[i % len(types)]})
    for i in range(int(cfg["nodes"]["server_nodes"]["count"])):
        nodes.append({**cfg["pci"][types[i % len(types)]]["specs"],
                      "node_name": f"node{i}", "type": types[i % len(types)]})
    for n in nodes:
        n["network_map"] = {}
    rng = random.Random(int(cfg["network"]["topology"]["seed"]))
    maps = generate_network_topology_deterministic(nodes, cfg, rng)
    clients = [n for n in nodes if n["node_name"].startswith("client")]
    servers = [n for n in nodes if not n["node_name"].startswith("client")]
    G = nx.Graph()
    G.add_nodes_from(n["node_name"] for n in nodes)
    for c in clients:
        for s, lat in maps.get(c["node_name"], {}).items():
            if s in {x["node_name"] for x in servers}:
                G.add_edge(c["node_name"], s, lat=float(lat))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    pos = {}
    for i, c in enumerate(clients):
        pos[c["node_name"]] = (i / max(1, len(clients) - 1), 0.0)
    for i, s in enumerate(servers):
        pos[s["node_name"]] = ((i + 0.5) / len(servers), 1.0)
    for u, v, d in G.edges(data=True):
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#bbbbbb",
                lw=0.25 + 1.5 * (0.15 - d["lat"]) / 0.12, alpha=0.7, zorder=1)
    for n in clients:
        ax.scatter(*pos[n["node_name"]], s=28, color=C[n["type"]], edgecolor="k", lw=0.4, zorder=3)
    for n in servers:
        plats = n.get("platforms", [])
        ax.scatter(*pos[n["node_name"]], s=90 + 12 * len(plats), color=C[n["type"]], edgecolor="k",
                   lw=0.8, marker="s", zorder=3)
        ax.annotate(f"{n['node_name']}\n{len(plats)} platf.", (pos[n["node_name"]][0], 1.0),
                    xytext=(0, 10), textcoords="offset points", ha="center", fontsize=7)
    deg = [G.degree(s["node_name"]) for s in servers]
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.15, 1.35); ax.axis("off")
    ax.set_title(f"cc40s9001: 40 clients (bottom) x 6 servers (top), seed 9001, p(link) = 0.6; "
                 f"server degree {min(deg)}-{max(deg)}; line width = 1/latency", fontsize=9)
    ax.legend(handles=[Patch(color=C["rpi"], label="Raspberry Pi 4 (4x CPU)"),
                       Patch(color=C["xavier"], label="Nvidia Xavier (CPU/GPU/DLA)"),
                       Patch(color=C["pyngFpga"], label="PYNQ (Artix-7 FPGA)")],
              loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12), frameon=False)
    save(fig, "fig1_topology.png")


# --- Fig 2: where a task's response time goes ---------------------------------------------------

def _po_medians(rung, corpus, kind):
    rows = [r for r in _load("results/po_v1_clients/*.summary.json")
            if r["rung"] == rung and r["arm_kind"] == kind and (corpus is None or r["corpus"] == corpus)]
    if not rows:
        return None
    e = median(float(r["averageElapsedTime"]) for r in rows)
    q = median(float(r["averageQueueTime"]) for r in rows)
    w = median(float(r["averageWaitTime"] or 0) for r in rows)
    return {"elapsed": e, "queue": q, "wait": w, "service": max(0.0, e - q - w), "n": len(rows)}


def fig2_composition():
    arms = [("reactive Knative", _po_medians("C40", None, "reactive"), C["reactive"]),
            ("gnnedge0", _po_medians("C40", "be1670", "gnnedge0"), C["gnnedge0"]),
            ("mpoff (twin)", _po_medians("C40", "1670", "mpoff"), C["mpoff"])]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for i, (name, m, col) in enumerate(arms):
        left = 0.0
        for key, hatch, lab in (("wait", "//", "scheduler wait (batch assembly)"),
                                ("queue", "", "queue behind other tasks' service"),
                                ("service", "..", "own service: latency, pull, peer exchange, execution")):
            ax.barh(i, m[key], left=left, color=col, alpha=0.35 if key == "queue" else 0.85 if key == "wait" else 0.6,
                    hatch=hatch, edgecolor="k", lw=0.5, label=lab if i == 0 else None)
            if m[key] > 1.5:
                ax.text(left + m[key] / 2, i, f"{m[key]:.1f} s", ha="center", va="center", fontsize=8)
            left += m[key]
        ax.text(left + 0.4, i, f"{m['elapsed']:.1f} s", va="center", fontsize=8, fontweight="bold")
    ax.set_yticks(range(len(arms))); ax.set_yticklabels([a[0] for a in arms])
    ax.set_xlabel("mean per-task response time, s  (6 servers / 40 clients, window w0; medians over 4 cells x 16 checkpoints)", fontsize=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False, fontsize=7)
    ax.set_xlim(0, 34)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig2_composition.png")


# --- Fig 3: peer groups and the batch window -----------------------------------------------------

def fig3_peer_groups():
    st = json.load(open(DATA / "ps_workload_stats.json"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1.6]})
    # left: one group of 10, each task with 2 partners (a representative draw, seeded)
    rng = random.Random(7)
    G = nx.Graph(); G.add_nodes_from(range(10))
    for i in range(10):
        for j in rng.sample([k for k in range(10) if k != i], 2):
            G.add_edge(i, j)
    pos = nx.circular_layout(G)
    nx.draw_networkx_edges(G, pos, ax=a, edge_color="#999999", width=1.2)
    nx.draw_networkx_nodes(G, pos, ax=a, node_color="#f5b7b1", edgecolors="k", node_size=180)
    nx.draw_networkx_labels(G, pos, ax=a, font_size=7)
    a.set_title("one peer group: 10 consecutive arrivals,\n2 partners each, 200 MB x 10^U(-1,1) per pair", fontsize=7.5, pad=2)
    a.axis("off")
    spans = np.array(st["span_s"])
    b.hist(spans, bins=np.arange(0, 80, 2), color="#95a5a6", edgecolor="k", lw=0.3)
    for w, col, dy in ((2, "#2ecc71", 0.97), (4, "#27ae60", 0.89), (8, "#f39c12", 0.81), (16, "#c0392b", 0.97)):
        b.axvline(w, color=col, lw=1.4, ls="--")
        b.text(w + 0.4, b.get_ylim()[1] * dy, f"{w} s", color=col, fontsize=7, ha="left")
    b.set_xlabel("arrival span of a group (first to last member), s\nwindow w0, 0.46 arrivals/s", fontsize=8)
    b.set_ylabel("groups")
    b.set_title(f"{st['n_groups']:,} groups: median span {np.median(spans):.1f} s, p90 "
                f"{np.percentile(spans, 90):.1f} s\nthe 16 s window sees "
                f"{100 * np.mean(spans <= 16):.0f} % of groups complete", fontsize=7.5, pad=2)
    b.spines[["top", "right"]].set_visible(False)
    save(fig, "fig3_peer_groups.png")


# --- Fig 4: co-simulation pipeline --------------------------------------------------------------

def _box(ax, x, y, w, h, text, fc="#ecf0f1", fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec="k", lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", lw=0.9))


def fig4_cosim():
    fig, ax = plt.subplots(figsize=(7.2, 2.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3)
    steps = ["1. seed -> topology,\nreplicas, queues", "2. warm-up episode\n(first event only);\ncapture state S",
             "3. batch T: candidate\nsets C_t = reachable\nreplicas of type(t)",
             "4. enumerate every\nplan in  prod_t |C_t|\n(72 ... 3,777 per snapshot)",
             "5. simulate EVERY\nplan to completion:\ncost = sum_t (done_t - dispatch_t)",
             "6. label = argmin;\nper task: index of its\noptimal candidate"]
    for i, s in enumerate(steps):
        _box(ax, 0.1 + i * 1.66, 1.25, 1.5, 1.5, s, fs=7)
        if i:
            _arrow(ax, 0.1 + i * 1.66 - 0.16, 2.0, 0.1 + i * 1.66, 2.0)
    _box(ax, 2.6, 0.15, 4.9, 0.75,
         "placements.jsonl: every plan with its cost (mandatory, ADR 0003) -> exact regret of any decoded plan, "
         "auditable optimum", fc="#fdf2e9", fs=7)
    _arrow(ax, 7.4, 1.25, 6.5, 0.9)
    save(fig, "fig4_cosim.png")


# --- Fig 5: the model ----------------------------------------------------------------------------

def fig5_model():
    fig = plt.figure(figsize=(7.2, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.6])
    a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1])
    # left: the graph
    tasks, plats = ["t1", "t2", "t3"], ["p1", "p2", "p3", "p4"]
    pos = {t: (0.0, 2 - i) for i, t in enumerate(tasks)}
    pos.update({p: (1.0, 2.25 - 1.5 * i / 3 * 2 / 1.5) for i, p in enumerate(plats)})
    pos = {**{t: (0.0, 2.0 - i) for i, t in enumerate(tasks)}, **{p: (1.0, 2.5 - i * 5 / 6) for i, p in enumerate(plats)}}
    G = nx.Graph()
    bip = [("t1", "p1"), ("t1", "p2"), ("t2", "p2"), ("t2", "p3"), ("t2", "p4"), ("t3", "p1"), ("t3", "p4")]
    peer = [("t1", "t2"), ("t2", "t3")]
    G.add_edges_from(bip + peer)
    nx.draw_networkx_edges(G, pos, ax=a, edgelist=bip, edge_color="#7f8c8d", width=1.2)
    nx.draw_networkx_edges(G, pos, ax=a, edgelist=peer, edge_color=C["gnnedge0"], width=1.8, style="dashed",
                           connectionstyle="arc3,rad=-0.6")
    nx.draw_networkx_nodes(G, pos, ax=a, nodelist=tasks, node_color="#f5b7b1", edgecolors="k", node_size=420)
    nx.draw_networkx_nodes(G, pos, ax=a, nodelist=plats, node_color="#aed6f1", edgecolors="k", node_size=420, node_shape="s")
    nx.draw_networkx_labels(G, pos, ax=a, font_size=8)
    a.text(0.0, 2.75, "tasks (3 feats)", ha="center", fontsize=8)
    a.text(1.0, 2.95, "candidate platforms (16 feats)", ha="center", fontsize=8)
    a.text(0.5, -0.65, "solid: task-platform edge [exec, latency, warm, energy, comm]\n"
                       "dashed: peer edge [log bytes]", ha="center", fontsize=7)
    a.set_xlim(-0.35, 1.35); a.set_ylim(-0.9, 3.1); a.axis("off")
    # right: the stages
    b.axis("off"); b.set_xlim(0, 10); b.set_ylim(0, 10)
    stages = [("encoders: task MLP, platform MLP  (64-d)", "#ecf0f1"),
              ("PeerConv over peer edges  (mean, residual)\n-- off in mpoff", "#fadbd8"),
              ("BipartiteEdgeConv over task-platform edges\n(mean; the GIN 'sum' variant is gnn)  -- off in peeronly, mpoff", "#fadbd8"),
              ("prefix block (22-d, size-free): what is already committed\non each candidate's machine, re-read at every decode step", "#fdf2e9"),
              ("EdgeScorer: [h_task, h_platform, edge, prefix] -> one logit;\nsoftmax per task over its candidates (CE vs the oracle)", "#ecf0f1"),
              ("masked topological decode: commit tasks one by one,\ncapacity-masked, re-score after every commit", "#d5f5e3")]
    y = 9.3
    for text, fc in stages:
        _box(b, 1.3, y - 1.35, 8.6, 1.3, text, fc=fc, fs=6.3)
        y -= 1.55
    b.annotate("", (1.3, 9.3 - 1.55 * 3 - 0.7), (1.3, 9.3 - 1.55 * 5 - 0.7),
               arrowprops=dict(arrowstyle="->", lw=1.0, connectionstyle="arc3,rad=0.8", color="#27ae60"))
    b.text(0.05, 9.3 - 1.55 * 4 - 0.7, "commit one task,\nrefresh the prefix,\nre-score", fontsize=6, color="#27ae60", va="center")
    save(fig, "fig5_model.png")


# --- Fig 7: reactive's capacity does not grow with the cluster ----------------------------------

def fig7_load_ladder():
    # docs/lineages/unsaturated_scale_v1.md, S1 (80 servers, 4 cells; f1000 unreadable on 3 cells)
    s80 = [(0.230, 0.480), (0.460, 0.660), (0.920, 0.955), (1.841, 0.978), (6.137, 0.988)]
    s6 = [(0.460, 0.634)]   # peer_only_v1 B5 R0 / unsaturated_scale_v1 text: 6 servers at f4000
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    ax.plot([x for x, _ in s80], [y for _, y in s80], "o-", color=C["reactive"], label="80 servers (unsaturated_scale_v1 S1)")
    ax.plot([x for x, _ in s6], [y for _, y in s6], "s", color=C["gnnedge0"], ms=8, label="6 servers, same rate")
    ax.axhline(0.80, color="#e67e22", ls="--", lw=1); ax.text(0.24, 0.81, "admissible <= 0.80", fontsize=7, color="#e67e22")
    ax.axhline(0.90, color="#c0392b", ls="--", lw=1); ax.text(0.24, 0.91, "saturated >= 0.90", fontsize=7, color="#c0392b")
    ax.set_xscale("log"); ax.set_xlabel("arrival rate, tasks/s (50,000-event window, rescaled)")
    ax.set_ylabel("reactive queue share\n(queue / response time)")
    ax.set_ylim(0.4, 1.02); ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig7_load_ladder.png")


# --- Fig 8: topology x window heat maps ---------------------------------------------------------

def fig8_windows():
    ue = screen_shares(str(DATA / "results/ue_v1_screen"))
    v2 = {}
    for r in _load("results/us_v2_screen/*.summary.json"):
        if int(r.get("num_tasks") or 0) == 50000:
            v2[(int(r["topology"]), r["window"])] = float(r["queue_share"])
    panels = [("6 servers / 40 clients", E_TOPOLOGY_CANDIDATES, lambda t, w: ue.get((40, t, w))),
              ("6 servers / 80 clients", E_TOPOLOGY_CANDIDATES, lambda t, w: ue.get((80, t, w))),
              ("80 servers / 20 clients", M_TOPOLOGY_CANDIDATES, lambda t, w: v2.get((t, w)))]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.6), sharey=True)
    for ax, (title, topos, get) in zip(axes, panels):
        M = np.full((len(topos), 4), np.nan)
        for i, t in enumerate(topos):
            for j, w in enumerate(E_WINDOWS):
                v = get(t, w)
                M[i, j] = np.nan if v is None else v
        im = ax.imshow(M, cmap="RdYlGn_r", vmin=0.4, vmax=1.0, aspect="auto")
        for i in range(len(topos)):
            for j in range(4):
                v = M[i, j]
                ax.text(j, i, "hang" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="k" if np.isnan(v) or v < 0.85 else "w")
        ax.set_xticks(range(4)); ax.set_xticklabels(E_WINDOWS)
        ax.set_yticks(range(len(topos))); ax.set_yticklabels([str(t) for t in topos], fontsize=7)
        ax.set_title(title, fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.8, label="reactive queue share (admissible <= 0.80)")
    save(fig, "fig8_windows.png")


# --- Fig 9: the headline at 16 environments per rung ------------------------------------------

def fig9_headline():
    study = DATA / "results/ue_v1"
    if len(list(study.glob("*.summary.json"))) < 1338:
        print("[skip] fig9: study not on disk"); return None
    m0 = read_m0(screen_shares(str(DATA / "results/ue_v1_screen")))
    arms, reactive = study_tables(str(study), str(DATA / "results/ue_v1_screen"))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.6, 0.6, 1.4]})
    out = {}
    # panels 1-2: per-checkpoint statistic per arm, C40 and C80
    def _complete(tab, envs):
        # DISCLOSED: checkpoints whose (environment) set is complete; a hung arm drops its checkpoint
        by = {}
        for (e, s), v in tab.items():
            by.setdefault(s, {})[e] = v
        keep = {s for s, d in by.items() if all(e in d for e in envs)}
        return {(e, s): v for (e, s), v in tab.items() if s in keep}
    for ax, nc in zip(axes[:2], E_RUNGS):
        sel = select_topologies(m0, nc); envs = [tuple(e) for e in sel["environments"]]
        for i, a in enumerate(E_ARMS_BY_RUNG[nc]):
            tab = _complete(arms.get(a, {}), envs)
            st = checkpoint_stats(tab, reactive, envs)
            vals = np.array([st[s] for s in sorted(st)])
            if len(vals) < 16:
                ax.text(i, max(vals) + 1.0, f"n={len(vals)}\n(disclosed)", ha="center", fontsize=6, color="#7f8c8d")
            rng = np.random.default_rng(1)
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(vals)), vals, s=14, color=C[LABEL[a].split()[0]], alpha=0.8, zorder=3)
            ax.plot([i - 0.3, i + 0.3], [np.median(vals)] * 2, color="k", lw=1.6, zorder=4)
            out[(nc, a)] = st
        ax.axhline(0, color="k", lw=0.6); ax.axhline(-5, color="#c0392b", ls="--", lw=0.8)
        ax.set_xticks(range(len(E_ARMS_BY_RUNG[nc]))); ax.set_xticklabels([LABEL[a] for a in E_ARMS_BY_RUNG[nc]], rotation=25, fontsize=7)
        ax.set_title(f"C{nc}: per checkpoint\n(median over 16 environments)", fontsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("% vs Knative, paired per environment\n(negative = faster)", fontsize=8)
    # panel 3: per window, C40
    ax = axes[2]; sel = select_topologies(m0, 40); envs = [tuple(e) for e in sel["environments"]]
    for a in E_ARMS_BY_RUNG[40]:
        pw = []
        tab = _complete(arms[a], envs)
        for w in E_WINDOWS:
            st = checkpoint_stats(tab, reactive, [e for e in envs if e[2] == w]); pw.append(median(st.values()))
        ax.plot(E_WINDOWS, pw, "o-", color=C[LABEL[a].split()[0]], label=LABEL[a])
    ax.axhline(0, color="k", lw=0.6); ax.axhline(-5, color="#c0392b", ls="--", lw=0.8)
    ax.set_title("C40 by arrival window\n(median over checkpoints)", fontsize=7.5); ax.legend(frameon=False, fontsize=6.5)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig9_headline.png")
    return out


# --- Fig 10: direction from a small design, magnitude from a large one --------------------------

def fig10_v1_vs_v2():
    # docs/lineages/unsaturated_scale_v2.md, head table (80 servers): v1 = 4 environments, v2 = 16
    rows = [("gnnedge0 vs reactive", -6.46, 0.47, -1.62, 0.0023),
            ("peeronly vs reactive", -7.97, 0.12, -1.23, 0.0013),
            ("gnn vs reactive", +16.98, 0.044, +2.31, 0.0052),
            ("gnnedge0 vs mpoff (model class)", -6.10, 0.47, -2.38, 0.0004),
            ("sum vs mean (gnn vs gnnedge0)", +26.69, 0.039, +2.70, 0.0019)]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    y = np.arange(len(rows))
    ax.barh(y + 0.18, [r[1] for r in rows], height=0.34, color="#bdc3c7", label="4 environments (v1, all window w0)")
    ax.barh(y - 0.18, [r[3] for r in rows], height=0.34, color=C["gnnedge0"], label="16 environments (v2)")
    for i, r in enumerate(rows):
        ax.text(r[1] + (0.6 if r[1] > 0 else -0.6), i + 0.18, f"{r[1]:+.1f} %  p={r[2]:.2f}", va="center", ha="left" if r[1] > 0 else "right", fontsize=7)
        ax.text(r[3] + (0.6 if r[3] > 0 else -0.6), i - 0.18, f"{r[3]:+.2f} %  p={r[4]:.4f}", va="center", ha="left" if r[3] > 0 else "right", fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8); ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.6); ax.set_xlim(-16, 36)
    ax.set_xlabel("median % vs the reference, 80 servers (negative = first arm faster)")
    ax.legend(frameon=False, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig10_v1_vs_v2.png")


# --- Fig 11: the batch tax and the window -------------------------------------------------------

def fig11_batch_window():
    bw = DATA / "results/bw_v1"
    if not bw.exists() or len(list(bw.glob("*.summary.json"))) < 180:
        print("[skip] fig11: batch-window results not on disk"); return
    from scripts_cosim.batch_window_edge_v1_gate_read import tables, W_GRAPH
    from scripts_cosim.batch_window_edge_v1_read import split_environments
    m0 = read_m0(screen_shares(str(DATA / "results/ue_v1_screen")))
    sel40 = select_topologies(m0, 40); sp = split_environments(sel40["topologies"])
    reactive, random_, elapsed, wait = tables(str(DATA / "results/ue_v1_screen"), str(DATA / "results/ue_v1"), str(bw))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.1, 1.4]})
    envs = sp["screen"]; xs, ys, ns, ws = [], [], [], []
    for (w, arm), tab in sorted(elapsed.items()):
        if arm != W_GRAPH:
            continue
        by = {}
        for (e, s), v in tab.items():
            by.setdefault(s, {})[e] = v
        keep = {s for s, d in by.items() if all(e in d for e in envs)}
        t = {(e, s): v for (e, s), v in tab.items() if s in keep}
        st = checkpoint_stats(t, reactive, envs)
        xs.append(w); ys.append(median(st.values())); ns.append(len(keep))
        ws.append(median(v for (e, s), v in wait[(w, W_GRAPH)].items() if e in envs and s in keep))
    a.plot(xs, ys, "o-", color=C["gnnedge0"], label="vs Knative, %")
    for x, y, n, wt in zip(xs, ys, ns, ws):
        a.annotate(f"wait {wt:.1f} s" + ("" if n == 16 else f", n={n}"), (x, y), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=6.5)
    a.set_xscale("log", base=2); a.set_xticks(xs); a.set_xticklabels([f"{x:g}" for x in xs])
    a.axhline(0, color="k", lw=0.6); a.axhline(-5, color="#c0392b", ls="--", lw=0.8); a.set_ylim(-7, 12)
    a.set_xlabel("batch window, s"); a.set_ylabel("gnnedge0 vs Knative, %\n(screen topology, 4 arrival windows)", fontsize=8)
    a.set_title("the window moves the wait, not the total", fontsize=8); a.spines[["top", "right"]].set_visible(False)
    # right: decomposition on the 16 study environments
    envs16 = set(tuple(e) for e in sel40["environments"])
    scr = [r for r in _load("results/ue_v1_screen/cc40*.summary.json") if (40, int(r["topology"]), r["window"]) in envs16]
    st = [r for r in _load("results/ue_v1/cc40*.summary.json") if (40, int(r["topology"]), r["window"]) in envs16]
    arms = [("Knative", scr, C["reactive"]), ("random", [r for r in st if r["arm_kind"] == "random_network"], C["random"]),
            ("gnnedge0", [r for r in st if r["arm_kind"] == "gnnedge0" and r["corpus"] == "be1670"], C["gnnedge0"]),
            ("peeronly", [r for r in st if r["arm_kind"] == "peeronly"], C["peeronly"]),
            ("mpoff (twin)", [r for r in st if r["arm_kind"] == "mpoff" and r["corpus"] == "1670"], C["mpoff"])]
    for i, (name, rows, col) in enumerate(arms):
        e = median(float(r["averageElapsedTime"]) for r in rows); wt = median(float(r["averageWaitTime"] or 0) for r in rows)
        q = median(float(r["averageQueueTime"]) for r in rows); px = median(float(r["totalPeerExchangeTime"]) / 50000 for r in rows)
        rd = median(float(r["totalPeerRendezvousWait"]) / 50000 for r in rows)
        left = 0
        for v, hatch, alpha in ((wt, "//", 0.9), (q, "", 0.3), (px, "..", 0.6), (rd, "xx", 0.45)):
            b.barh(i, v, left=left, color=col, alpha=alpha, hatch=hatch, edgecolor="k", lw=0.5); left += v
        b.text(e + 0.3, i, f"{e:.1f} s", va="center", fontsize=7)
    b.set_yticks(range(len(arms))); b.set_yticklabels([x[0] for x in arms], fontsize=7); b.invert_yaxis(); b.set_xlim(0, 23)
    b.set_xlabel("per task, s: wait (//) | queue | exchange (..) | rendezvous (xx)", fontsize=7.5)
    b.set_title("C40, 16 environments: where the time goes", fontsize=8); b.spines[["top", "right"]].set_visible(False)
    save(fig, "fig11_batch_window.png")


if __name__ == "__main__":
    fig1_topology(); fig2_composition(); fig3_peer_groups(); fig4_cosim(); fig5_model()
    fig7_load_ladder(); fig8_windows(); fig9_headline(); fig10_v1_vs_v2(); fig11_batch_window()
