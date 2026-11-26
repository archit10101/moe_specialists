"""The SLURM submitter: every job must reach sbatch with its full command (dry run, no SLURM needed)."""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("profile_flag", ["--smoke", None])
def test_every_submitted_job_has_its_command(tmp_path, profile_flag):
    env = dict(os.environ, REPRO_NO_CONDA="1", REPRO_WORK=str(tmp_path))
    cmd = ["bash", "slurm/submit_all.sh", "--dry-run"] + ([profile_flag] if profile_flag else [])
    subprocess.run(cmd, cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    lines = (tmp_path / "logs" / "dry_run.txt").read_text().splitlines()
    assert lines, "nothing was submitted"
    for line in lines:
        name = re.search(r"-J (\S+)", line).group(1)
        args = line.split("slurm/job.sbatch", 1)[1].split()
        assert args, f"{name} was submitted without a command: {line}"
    by_name = {re.search(r"-J (\S+)", l).group(1): l.split("slurm/job.sbatch", 1)[1].split() for l in lines}
    prefix = "repro_smoke_" if profile_flag else "repro_"
    assert by_name[prefix + "prepare"] == ["prepare"]
    assert by_name[prefix + "cluster"] == ["cluster"]
    assert by_name[prefix + "base_base.edl_s0"] == ["base", "--exp", "base.edl", "--seed", "0"]
    assert "--dependency" not in " ".join(l for l in lines if f"-J {prefix}prepare " in l)
