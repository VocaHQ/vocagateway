from __future__ import annotations

import subprocess
import sys

REMEDIATED_FILES: tuple[str, ...] = (
    "app/templating.py",
    "app/context.py",
    "app/storage.py",
    "app/readiness.py",
    "app/models/mlx_audio.py",
    "app/models/whisper_cpp.py",
    "app/routes/admin_status.py",
    "app/routes/admin_tokens.py",
    "app/routes/pairing.py",
    "app/fragments/tokens.py",
    "app/fragments/pairing.py",
    "app/metrics.py",
    "app/audio.py",
    "app/config.py",
    "app/pairing_view.py",
    "app/routes/sessions.py",
    "app/runtime_config.py",
    "app/models/warmup.py",
    "app/routes/admin_config.py",
    "app/models/faster_whisper.py",
    "app/tokens.py",
    "app/models/handy.py",
    "app/models/moonshine.py",
    "tests/conftest.py",
    "tests/test_readiness.py",
    "tests/test_harvest_model_pins.py",
    "tests/test_cli_entry_points.py",
    "tests/test_diagnostics.py",
    "tests/test_faster_whisper.py",
    "tests/test_metrics.py",
    "tests/test_moonshine.py",
    "tests/test_whisper_cpp.py",
    "tests/test_whisperkit.py",
)


def test_wps_rules_compliance_vector() -> None:
    command = [sys.executable, "-m", "flake8", *REMEDIATED_FILES]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stdout
