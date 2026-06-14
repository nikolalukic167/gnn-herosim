"""Grid preset sanity checks for skew_warmth_v2."""

from scripts_cosim.generate_gnn_datasets_fast import (
    grid_total_datasets,
    grid_topology_variants,
    resolve_grid_preset,
)


def test_skew_warmth_v2_grid_count():
    preset = resolve_grid_preset("skew_warmth_v2")
    assert grid_total_datasets(preset) == 288
    assert len(grid_topology_variants(preset)) == 9  # 3 k_core × 3 seek


def test_skew_warmth_v2_output_subdir():
    preset = resolve_grid_preset("skew_warmth_v2")
    assert preset["default_output_subdir"] == "gnn_datasets_4tasks_skew_warmth_v2"


def test_skew_warmth_v2_placement_combos_nonzero():
    """Skew infra + load_deterministic must agree on xavier core nodes (platform_id parity)."""
    import json
    import shutil
    from pathlib import Path

    from scripts_cosim.generate_gnn_datasets_fast import (
        NUM_WORKLOAD_TEMPLATES,
        create_config_for_iteration,
        generate_workload_templates,
        grid_topology_variants,
        resolve_grid_preset,
    )
    from src.executecosimulation import (
        determine_replica_placement,
        flatten_workloads,
        generate_brute_force_placement_combinations,
        load_simulation_inputs,
        prepare_simulation_config,
        prepare_workloads,
    )
    from src.generate_infrastructure import generate_deterministic_infrastructure
    from src.sample_loader import load_primary_sample_and_mapping

    root = Path("simulation_data/gnn_datasets_4tasks_skew_warmth_v2_test")
    if not (root / "ds_00000" / "infrastructure.json").exists():
        preset = resolve_grid_preset("skew_warmth_v2")
        topo_label, topo_kwargs = grid_topology_variants(preset)[0]
        base_config = json.load(open("simulation_data/space_with_network.json"))
        config = create_config_for_iteration(
            base_config,
            preset["replica_configs"][0],
            preset["seeds"][0],
            preset["queue_distributions"][0],
            batch_size=4,
            **topo_kwargs,
        )
        out = root / "ds_00000"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "space_with_network.json", "w") as f:
            f.write(json.dumps(config, indent=2))
        templates = generate_workload_templates(
            Path("data/nofs-ids/traces/workload-10.json"),
            Path("data/nofs-ids/traces/gnn_templates_pytest"),
            NUM_WORKLOAD_TEMPLATES,
            quiet=True,
        )
        shutil.copy2(templates[0], out / "workload.json")
        generate_deterministic_infrastructure(
            str(out / "space_with_network.json"),
            Path("data/nofs-ids"),
            str(out / "infrastructure.json"),
            preset["seeds"][0],
        )

    out = root / "ds_00000"
    sample, mapping, _ = load_primary_sample_and_mapping(
        Path("simulation_data/sample_simple.json"),
        Path("simulation_data/lhs_samples_simple.npy"),
        Path("simulation_data/lhs_samples_simple_mapping.pkl"),
    )
    sim_inputs = load_simulation_inputs(Path("data/nofs-ids"))
    with open(out / "space_with_network.json") as f:
        infra_config = json.load(f)
    with open(out / "workload.json") as f:
        workload_base = json.load(f)
    apps = list(infra_config["wsc"].keys())
    flat = flatten_workloads(prepare_workloads(sample, mapping, workload_base, apps))
    sim_config = prepare_simulation_config(
        sample, mapping, infra_config, infrastructure_file=out / "infrastructure.json"
    )
    replica_plan = determine_replica_placement(sim_config, sim_inputs)
    combos = generate_brute_force_placement_combinations(
        flat["events"],
        sim_config,
        sim_inputs,
        replica_plan,
        use_all_replicas=True,
        allow_non_unique_replicas=True,
    )
    assert len(combos) > 0


def test_workload_templates_dir_isolated_per_job(monkeypatch, tmp_path):
    from scripts_cosim.generate_gnn_datasets_fast import workload_templates_dir_for_run

    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "2")
    d1 = workload_templates_dir_for_run(tmp_path)
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    d2 = workload_templates_dir_for_run(tmp_path)
    assert d1 != d2
    assert "gnn_templates_12345_2" in str(d1)
