#!/usr/bin/env python3
"""Render bipartite topology sketches from real config JSON + infra generator."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generate_infrastructure import (  # noqa: E402
    apply_degree_skew_core_server_device_types,
    generate_network_topology_deterministic,
)

OUT_DIR = PROJECT_ROOT / "simulation_data" / "topology_sketches"

CONFIGS = {
    "default_uniform": PROJECT_ROOT / "simulation_data" / "space_with_network.json",
    "sparse_no_hub": (
        PROJECT_ROOT
        / "simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/05_sparse_40_40_p25.json"
    ),
    "degree_skew_20x20": (
        PROJECT_ROOT
        / "simulation_data/normal_sim_sweeps/atomic21_skew_configs/default_20_20_degree_skew.json"
    ),
    "hub_k2_seek30": (
        PROJECT_ROOT
        / "simulation_data/normal_sim_sweeps/tiered_hub_gnn_mlp_125225_20260610/configs/hub_k2_seek30.json"
    ),
    "hub_k2_seek80": (
        PROJECT_ROOT
        / "simulation_data/normal_sim_sweeps/tiered_hub_gnn_mlp_125225_20260610/configs/hub_k2_seek80.json"
    ),
    "hub_k6_seek80": (
        PROJECT_ROOT
        / "simulation_data/normal_sim_sweeps/tiered_hub_gnn_mlp_125225_20260610/configs/hub_k6_seek80.json"
    ),
}


def load_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_nodes(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    n_clients = int(config["nodes"]["client_nodes"]["count"])
    n_servers = int(config["nodes"]["server_nodes"]["count"])
    clients = [{"node_name": f"client_node{i}", "type": "rpi"} for i in range(n_clients)]
    servers = [{"node_name": f"server_node{i}", "type": "rpi"} for i in range(n_servers)]
    nodes = clients + servers
    apply_degree_skew_core_server_device_types(nodes, config)
    return nodes


def hub_seeker_labels(
    clients: List[Dict[str, Any]],
    servers: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, bool]:
    topo = config.get("network", {}).get("topology", {})
    if topo.get("type") != "degree_skewed_core":
        return {c["node_name"]: False for c in clients}

    seed = int(topo.get("seed", 42))
    hub_frac = float(topo.get("hub_seeker_fraction", 0.4))
    k_core = int(topo.get("k_core", 4))
    p_core = float(topo.get("p_core", 0.95))
    p_periphery = float(topo.get("p_periphery", 0.15))
    lat_s = float(topo.get("latency_core_ms", 5.0)) / 1000.0
    core_servers = {s["node_name"] for s in servers[:k_core]}

    rng = random.Random(seed)
    labels: Dict[str, bool] = {}
    for client in clients:
        is_seeker = rng.random() < hub_frac
        labels[client["node_name"]] = is_seeker
        for server in servers:
            s_name = server["node_name"]
            in_core = s_name in core_servers
            if is_seeker and in_core:
                p_conn = p_core
            elif not is_seeker and not in_core:
                p_conn = p_periphery
            elif is_seeker and not in_core:
                p_conn = p_periphery * 0.5
            else:
                p_conn = p_core * 0.3
            rng.random()  # same connection draw as generator
    return labels


def generate_maps(config: Dict[str, Any]) -> Tuple[List[Dict], List[Dict], Dict[str, Dict[str, float]], Dict[str, bool]]:
    nodes = build_nodes(config)
    clients = [n for n in nodes if n["node_name"].startswith("client_node")]
    servers = [n for n in nodes if not n["node_name"].startswith("client_node")]
    topo = config.get("network", {}).get("topology", {})
    seed = int(topo.get("seed", 101))
    rng = random.Random(seed)
    network_maps = generate_network_topology_deterministic(nodes, config, rng)
    seekers = hub_seeker_labels(clients, servers, config)
    return clients, servers, network_maps, seekers


def server_degrees_to_core(
    clients: List[Dict],
    servers: List[Dict],
    network_maps: Dict[str, Dict[str, float]],
    k_core: int,
) -> Tuple[List[int], List[int]]:
    core = {s["node_name"] for s in servers[:k_core]}
    core_deg, peri_deg = [], []
    for server in servers:
        s_name = server["node_name"]
        deg = sum(1 for c in clients if s_name in network_maps.get(c["node_name"], {}))
        if s_name in core:
            core_deg.append(deg)
        else:
            peri_deg.append(deg)
    return core_deg, peri_deg


def draw_bipartite(
    ax: plt.Axes,
    title: str,
    clients: List[Dict],
    servers: List[Dict],
    network_maps: Dict[str, Dict[str, float]],
    seekers: Dict[str, bool],
    k_core: int = 0,
    *,
    max_nodes: int = 24,
) -> None:
    topo_type = "uniform" if k_core == 0 else "hub"
    if len(clients) + len(servers) > max_nodes:
        clients = clients[: min(len(clients), max_nodes // 2)]
        servers = servers[: min(len(servers), max_nodes // 2)]

    core_names = {s["node_name"] for s in servers[:k_core]} if k_core else set()

    y_c = [i for i in range(len(clients))]
    y_s = [i for i in range(len(servers))]
    x_c, x_s = 0.0, 1.0

    for client in clients:
        c_name = client["node_name"]
        seeker = seekers.get(c_name, False)
        color = "#e67e22" if seeker else "#3498db"
        ax.scatter([x_c], [y_c[int(c_name.replace("client_node", ""))]], s=80, c=color, zorder=3, edgecolors="white", linewidths=0.5)

    for server in servers:
        s_name = server["node_name"]
        idx = int(s_name.replace("server_node", ""))
        is_core = s_name in core_names
        color = "#c0392b" if is_core else "#95a5a6"
        size = 120 if is_core else 70
        ax.scatter([x_s], [y_s[idx]], s=size, c=color, marker="s", zorder=3, edgecolors="white", linewidths=0.5)

    for client in clients:
        c_name = client["node_name"]
        c_idx = int(c_name.replace("client_node", ""))
        seeker = seekers.get(c_name, False)
        for s_name in network_maps.get(c_name, {}):
            s_idx = int(s_name.replace("server_node", ""))
            if s_idx >= len(servers) or c_idx >= len(clients):
                continue
            is_core = s_name in core_names
            if topo_type == "hub":
                lw = 1.8 if (seeker and is_core) else 0.35
                alpha = 0.85 if (seeker and is_core) else 0.15
                color = "#e74c3c" if is_core else "#bdc3c7"
            else:
                lw = 0.6
                alpha = 0.25
                color = "#7f8c8d"
            ax.plot([x_c, x_s], [y_c[c_idx], y_s[s_idx]], color=color, linewidth=lw, alpha=alpha, zorder=1)

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-1, max(len(clients), len(servers)))
    ax.set_xticks([x_c, x_s])
    ax.set_xticklabels(["clients", "servers"])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)


def subtitle_for(name: str, config: Dict[str, Any]) -> str:
    nodes = config["nodes"]
    nc = nodes["client_nodes"]["count"]
    ns = nodes["server_nodes"]["count"]
    topo = config.get("network", {}).get("topology", {})
    ttype = topo.get("type", "sparse")
    if ttype == "degree_skewed_core":
        return (
            f"{nc}×{ns} · k_core={topo.get('k_core')} · seek={int(float(topo.get('hub_seeker_fraction', 0))*100)}%"
        )
    p = topo.get("connection_probability", 0.5)
    return f"{nc}×{ns} · sparse p={p}"


def plot_overview() -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    panels = [
        ("default_uniform", "Uniform (default_20_20_p50)"),
        ("sparse_no_hub", "Sparse, no hub (05_sparse)"),
        ("degree_skew_20x20", "Degree-skew (skew-4)"),
        ("hub_k2_seek80", "Tiered hub k2 seek80"),
    ]
    for ax, (key, title) in zip(axes.flat, panels):
        cfg = load_config(CONFIGS[key])
        clients, servers, maps, seekers = generate_maps(cfg)
        k_core = int(cfg.get("network", {}).get("topology", {}).get("k_core", 0))
        draw_bipartite(ax, f"{title}\n{subtitle_for(key, cfg)}", clients, servers, maps, seekers, k_core)
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db", markersize=8, label="client"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e67e22", markersize=8, label="hub-seeker client"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#c0392b", markersize=10, label="hub core server"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#95a5a6", markersize=8, label="periphery server"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle("Topology families (bipartite client–server graphs from real JSON)", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = OUT_DIR / "overview_4panel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_seek_comparison() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    keys = [
        ("hub_k2_seek30", "k2 seek30 (30% hub-seekers)"),
        ("hub_k2_seek80", "k2 seek80 (80% hub-seekers)"),
        ("hub_k6_seek80", "k6 seek80 (larger core)"),
    ]
    for ax, (key, title) in zip(axes, keys):
        cfg = load_config(CONFIGS[key])
        clients, servers, maps, seekers = generate_maps(cfg)
        k_core = int(cfg["network"]["topology"]["k_core"])
        draw_bipartite(ax, title, clients, servers, maps, seekers, k_core, max_nodes=40)
    fig.suptitle("Tiered-hub: low vs high seek + larger k_core", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "tiered_hub_seek_compare.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_degree_bars() -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    panels = [
        ("default_uniform", "Uniform"),
        ("sparse_no_hub", "Sparse"),
        ("degree_skew_20x20", "Degree-skew k4"),
        ("hub_k2_seek80", "Hub k2 seek80"),
    ]
    for ax, (key, title) in zip(axes.flat, panels):
        cfg = load_config(CONFIGS[key])
        clients, servers, maps, _ = generate_maps(cfg)
        topo = cfg.get("network", {}).get("topology", {})
        k_core = int(topo.get("k_core", 0))
        if k_core:
            core_deg, peri_deg = server_degrees_to_core(clients, servers, maps, k_core)
            labels = [f"H{i}" for i in range(len(core_deg))] + [f"P{i}" for i in range(len(peri_deg))]
            colors = ["#c0392b"] * len(core_deg) + ["#95a5a6"] * len(peri_deg)
            vals = core_deg + peri_deg
            ax.bar(range(len(vals)), vals, color=colors, width=0.7)
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_ylabel("client links")
            ax.set_title(f"{title}: server degree (hub vs periphery)")
        else:
            degs = [
                sum(1 for c in clients if s["node_name"] in maps.get(c["node_name"], {}))
                for s in servers
            ]
            ax.hist(degs, bins=range(0, max(degs) + 2), color="#3498db", edgecolor="white")
            ax.set_xlabel("links per server")
            ax.set_ylabel("count")
            ax.set_title(f"{title}: server degree distribution")
    fig.suptitle("Why skew matters — hub servers attract many more client links", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "server_degree_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_schematic() -> Path:
    """Small hand-layout concept diagram (not RNG) for intuition."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def panel_uniform(ax: plt.Axes) -> None:
        clients_y = [3, 2, 1, 0]
        servers_y = [2.5, 1.5, 0.5]
        for i, y in enumerate(clients_y):
            ax.scatter(0, y, s=200, c="#3498db", zorder=3)
            ax.text(-0.08, y, f"C{i}", ha="right", va="center", fontsize=9)
        for i, y in enumerate(servers_y):
            ax.scatter(1, y, s=200, c="#95a5a6", marker="s", zorder=3)
            ax.text(1.08, y, f"S{i}", ha="left", va="center", fontsize=9)
        edges = [(0, 3, 0), (0, 2, 0), (0, 1, 1), (0, 0, 1), (1, 3, 1), (1, 2, 2), (1, 1, 2), (2, 3, 2), (2, 2, 1), (3, 1, 0)]
        for cx, sy_c, sy_s in edges:
            ax.plot([0, 1], [clients_y[cx], servers_y[sy_s]], color="#7f8c8d", alpha=0.5, lw=1.2)
        ax.set_title("Uniform / sparse\n~equal random wiring", fontweight="bold")
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-0.5, 3.5)
        ax.axis("off")

    def panel_hub(ax: plt.Axes, seek: str, frac: float) -> None:
        clients_y = [4, 3, 2, 1, 0]
        hub_y = [3, 1]
        peri_y = [2.5, 0.5]
        n_seek = int(round(frac * len(clients_y)))
        for i, y in enumerate(clients_y):
            seeker = i < n_seek
            ax.scatter(0, y, s=220, c="#e67e22" if seeker else "#3498db", zorder=3)
            label = f"C{i}{'*' if seeker else ''}"
            ax.text(-0.1, y, label, ha="right", va="center", fontsize=9)
        ax.scatter([1, 1], hub_y, s=350, c="#c0392b", marker="s", zorder=3)
        ax.text(1.12, 3, "H0", fontsize=9, fontweight="bold", color="#c0392b")
        ax.text(1.12, 1, "H1", fontsize=9, fontweight="bold", color="#c0392b")
        ax.scatter([1, 1], peri_y, s=180, c="#95a5a6", marker="s", zorder=3)
        for i, y in enumerate(peri_y):
            ax.text(1.12, y, f"P{i}", ha="left", va="center", fontsize=8, color="#7f8c8d")
        for i, cy in enumerate(clients_y):
            seeker = i < n_seek
            if seeker:
                for hy in hub_y:
                    ax.plot([0, 1], [cy, hy], color="#e74c3c", lw=2.5, alpha=0.9, zorder=1)
                if frac < 0.7:
                    ax.plot([0, 1], [cy, peri_y[0]], color="#bdc3c7", lw=0.6, alpha=0.3, zorder=1)
            else:
                for py in peri_y:
                    ax.plot([0, 1], [cy, py], color="#95a5a6", lw=1.0, alpha=0.5, zorder=1)
                ax.plot([0, 1], [cy, hub_y[0]], color="#bdc3c7", lw=0.5, alpha=0.25, zorder=1)
        ax.set_title(f"Tiered hub k2 {seek}\n* = hub-seeker client", fontweight="bold")
        ax.set_xlim(-0.35, 1.45)
        ax.set_ylim(-0.5, 4.5)
        ax.axis("off")

    panel_uniform(axes[0])
    panel_hub(axes[1], "seek30", 0.3)
    panel_hub(axes[2], "seek80", 0.8)
    fig.suptitle("Concept sketches (toy 5×4 graphs)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "concept_schematic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        plot_schematic(),
        plot_overview(),
        plot_seek_comparison(),
        plot_degree_bars(),
    ]
    print("Wrote topology sketches:")
    for p in outputs:
        print(f"  {p.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
