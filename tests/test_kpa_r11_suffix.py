"""kpa_scaleout_v1 readers: --suffix redirects the gate directories, default unchanged."""
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts_cosim"


def test_suffix_parameter_defaults_to_published_directories():
    for name in ("kpa_scaleout_v1_read.py", "kpa_shared_control_read.py"):
        src = (SCRIPTS / name).read_text()
        assert 'add_argument("--suffix", default=""' in src
    assert "load(d + a.suffix)" in (SCRIPTS / "kpa_scaleout_v1_read.py").read_text()
    shared = (SCRIPTS / "kpa_shared_control_read.py").read_text()
    assert "load([SHARED[cond] + a.suffix]), load([KPA[cond] + a.suffix])" in shared and "LEGACY[cond] + a.suffix" not in shared


def test_sbatch_guards():
    text = (SCRIPTS / "datalab" / "kpa_scaleout_v1_r11.sbatch").read_text()
    assert "unset GATE_FIXED_POLICY_TIME_SCALE" in text
    assert '[[ ! -e "$OUT" ]]' in text and "_kpa_r11" in text and "_legacy_shared_r11" in text
    assert subprocess.run(["bash", "-n", str(SCRIPTS / "datalab" / "kpa_scaleout_v1_r11.sbatch")]).returncode == 0
