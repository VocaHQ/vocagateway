from __future__ import annotations

import subprocess
import sys


def test_wps_rules_cover_the_complete_application_package() -> None:
    command = [sys.executable, "-m", "flake8", "app"]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stdout
