from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

MACOS = "macOS"
APPLE_SILICON = "Apple silicon"

# Engines that cannot run on every host, and the requirement the WebUI shows.
# The desktop-app adapters are the strictest: Handy ships for macOS, and VocaMac
# is Apple-silicon-only, so neither exists on Linux or inside a container.
ENGINE_HOST_REQUIREMENTS = {
    "vocamac": APPLE_SILICON,
    "handy": MACOS,
    "whisperkit": MACOS,
    "mlx-audio": APPLE_SILICON,
}


@dataclass(frozen=True, slots=True)
class SystemInfo:
    os_name: str
    os_version: str
    arch: str
    chip: str
    ram_gb: float
    is_apple_silicon: bool
    ffmpeg_path: str | None
    whisper_cpp_path: str | None
    whisperkit_cli_path: str | None
    handy_installed: bool
    vocamac_installed: bool
    logical_cpus: int
    effective_cpus: float
    containerized: bool
    accelerators: tuple[str, ...]
    cpu_features: tuple[str, ...]


def detect_system(
    *,
    whisper_binary: Path,
    whisperkit_binary: str,
    handy_binary: Path,
    vocamac_app: Path = Path("/Applications/VocaMac.app"),
) -> SystemInfo:
    arch = platform.machine()
    is_mac = platform.system() == "Darwin"
    chip = _sysctl("machdep.cpu.brand_string") if is_mac else _linux_cpu_brand()
    ram_gb = _ram_gb(is_mac)
    logical_cpus = os.cpu_count() or 1
    return SystemInfo(
        os_name=platform.system(),
        os_version=platform.release(),
        arch=arch,
        chip=chip or arch,
        ram_gb=ram_gb,
        is_apple_silicon=is_mac and arch == "arm64",
        ffmpeg_path=shutil.which("ffmpeg"),
        whisper_cpp_path=(
            str(whisper_binary) if whisper_binary.is_file() else shutil.which("whisper-cli")
        ),
        whisperkit_cli_path=_resolve_binary(whisperkit_binary),
        handy_installed=handy_binary.is_file(),
        vocamac_installed=vocamac_app.exists(),
        logical_cpus=logical_cpus,
        effective_cpus=_effective_cpu_count(logical_cpus),
        containerized=_is_containerized(),
        accelerators=_accelerators(is_mac, arch),
        cpu_features=_cpu_features(is_mac),
    )


def engine_requirement(engine: str) -> str | None:
    """The host an engine needs, or None when it runs anywhere."""
    return ENGINE_HOST_REQUIREMENTS.get(engine)


def engine_runs_on(engine: str, *, is_mac: bool, is_apple_silicon: bool) -> bool:
    requirement = ENGINE_HOST_REQUIREMENTS.get(engine)
    if requirement is None:
        return True
    return is_apple_silicon if requirement == APPLE_SILICON else is_mac


def engine_runs_here(engine: str) -> bool:
    """The same check for the running host, without a full `detect_system` probe."""
    is_mac = platform.system() == "Darwin"
    return engine_runs_on(
        engine,
        is_mac=is_mac,
        is_apple_silicon=is_mac and platform.machine() == "arm64",
    )


def _resolve_binary(binary: str) -> str | None:
    candidate = Path(binary).expanduser()
    if candidate.is_file():
        return str(candidate)
    return shutil.which(binary)


def _sysctl(key: str) -> str:
    try:
        result = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _ram_gb(is_mac: bool) -> float:
    if is_mac:
        raw = _sysctl("hw.memsize")
        return round(int(raw) / (1024**3), 1) if raw.isdigit() else 0.0
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return 0.0
    if not isinstance(pages, int) or not isinstance(page_size, int):
        return 0.0
    detected_bytes = pages * page_size
    try:
        memory_limit = Path("/sys/fs/cgroup/memory.max").read_text(encoding="utf-8").strip()
        if memory_limit != "max":
            detected_bytes = min(detected_bytes, int(memory_limit))
    except (OSError, ValueError):
        pass
    return round(detected_bytes / (1024**3), 1)


def _linux_cpu_brand() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith(("model name", "hardware")) and ":" in line:
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _effective_cpu_count(logical_cpus: int) -> float:
    limits = [float(logical_cpus)]
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").split()[:2]
        if quota != "max" and int(period) > 0:
            limits.append(max(0.1, int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    for candidate in (
        Path("/sys/fs/cgroup/cpuset.cpus.effective"),
        Path("/sys/fs/cgroup/cpuset/cpuset.cpus"),
    ):
        try:
            count = _count_cpu_set(candidate.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if count:
            limits.append(float(count))
            break
    return round(min(limits), 2)


def _count_cpu_set(value: str) -> int:
    count = 0
    for part in value.split(","):
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            count += max(0, int(end) - int(start) + 1)
        else:
            int(part)
            count += 1
    return count


def _is_containerized() -> bool:
    if Path("/.dockerenv").exists() or os.environ.get("CONTAINER"):
        return True
    try:
        cgroup = Path("/proc/1/cgroup").read_text(encoding="utf-8").lower()
    except OSError:
        return False
    return any(marker in cgroup for marker in ("docker", "containerd", "kubepods", "podman"))


def _accelerators(is_mac: bool, arch: str) -> tuple[str, ...]:
    """Capability labels for the WebUI. Prefer named GPUs when tools are present."""
    values: list[str] = ["CPU"]
    if is_mac and arch == "arm64":
        values.append("Metal / Core ML")
    nvidia = _nvidia_gpu_labels()
    if nvidia:
        values.extend(nvidia)
    elif Path("/dev/nvidia0").exists() or shutil.which("nvidia-smi"):
        values.append("NVIDIA CUDA")
    amd = _amd_gpu_labels()
    if amd:
        values.extend(amd)
    elif Path("/dev/kfd").exists():
        values.append("AMD ROCm")
    # Only mention a generic DRM device when no named GPU was found above.
    if (
        Path("/dev/dri/renderD128").exists()
        and not nvidia
        and not amd
        and not (is_mac and arch == "arm64")
    ):
        values.append("Vulkan / VAAPI device")
    return tuple(values)


def _nvidia_gpu_labels() -> list[str]:
    """One label per NVIDIA GPU, e.g. 'NVIDIA GeForce RTX 5080 (16 GB)'."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0 or not result.stdout.strip():
        return []
    labels: list[str] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if not parts or not parts[0]:
            continue
        name = parts[0]
        if not name.upper().startswith("NVIDIA"):
            name = f"NVIDIA {name}"
        if len(parts) > 1:
            try:
                mem_mib = float(parts[1])
                mem_gb = round(mem_mib / 1024, 1)
                labels.append(f"{name} ({mem_gb:g} GB)")
                continue
            except ValueError:
                pass
        labels.append(name)
    return labels


def _amd_gpu_labels() -> list[str]:
    """Best-effort AMD product names from rocm-smi, lspci, or DRM sysfs."""
    labels = _rocm_gpu_labels()
    if labels:
        return labels
    labels = _lspci_gpu_labels(vendor="AMD")
    if labels:
        return labels
    # Sysfs often only has a PCI device id; skip opaque "AMD device 0x...." labels.
    return [label for label in _drm_amd_labels() if "device 0x" not in label.lower()]


def _lspci_gpu_labels(*, vendor: str) -> list[str]:
    if not shutil.which("lspci"):
        return []
    try:
        result = subprocess.run(
            ["lspci", "-mm"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    labels: list[str] = []
    vendor_key = vendor.lower()
    for line in result.stdout.splitlines():
        # lspci -mm: Slot "Class" "Vendor" "Device" ...
        if '"VGA compatible controller"' not in line and '"3D controller"' not in line:
            continue
        if vendor_key not in line.lower():
            continue
        # Pull the Device field (4th quoted token).
        parts = line.split('"')
        # ["00:00.0 ", "Class", " ", "Vendor", " ", "Device", ...]
        device = parts[5].strip() if len(parts) > 5 else ""
        if not device:
            continue
        label = device if device.upper().startswith(vendor.upper()) else f"{vendor} {device}"
        if label not in labels:
            labels.append(label)
    return labels


def _rocm_gpu_labels() -> list[str]:
    if not shutil.which("rocm-smi"):
        return []
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    labels: list[str] = []
    for line in result.stdout.splitlines():
        # Typical: "GPU[0] : Card series: Radeon RX 7900 XTX"
        if ":" not in line or "GPU[" not in line.upper():
            continue
        name = line.split(":", 1)[1].strip()
        for prefix in ("Card series:", "Card model:", "Device Name:"):
            if name.startswith(prefix):
                name = name[len(prefix) :].strip()
        if name and name not in labels:
            labels.append(name if name.upper().startswith("AMD") else f"AMD {name}")
    return labels


def _drm_amd_labels() -> list[str]:
    """Fallback: product names under /sys/class/drm for AMD PCI devices."""
    root = Path("/sys/class/drm")
    if not root.is_dir():
        return []
    labels: list[str] = []
    seen: set[str] = set()
    for card in sorted(root.glob("card[0-9]*")):
        if "-" in card.name:
            continue
        vendor_path = card / "device" / "vendor"
        label_path = card / "device" / "label"
        uevent_path = card / "device" / "uevent"
        try:
            vendor = vendor_path.read_text(encoding="utf-8").strip().lower()
        except OSError:
            continue
        # 0x1002 is AMD
        if vendor not in {"0x1002", "1002"}:
            continue
        name = ""
        try:
            name = label_path.read_text(encoding="utf-8").strip()
        except OSError:
            pass
        if not name:
            try:
                for line in uevent_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith("PCI_ID=") or line.startswith("DRIVER="):
                        continue
                    if line.startswith("MODALIAS="):
                        continue
            except OSError:
                pass
        # product_name is common on some stacks
        for candidate in (
            card / "device" / "product_name",
            card / "device" / "marketing_name",
        ):
            if name:
                break
            try:
                name = candidate.read_text(encoding="utf-8").strip()
            except OSError:
                continue
        if not name:
            try:
                device_id = (card / "device" / "device").read_text(encoding="utf-8").strip()
                name = f"device {device_id}"
            except OSError:
                name = card.name
        label = name if name.upper().startswith("AMD") else f"AMD {name}"
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def _cpu_features(is_mac: bool) -> tuple[str, ...]:
    if is_mac:
        raw = " ".join((_sysctl("machdep.cpu.features"), _sysctl("machdep.cpu.leaf7_features")))
    else:
        try:
            raw = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        except OSError:
            raw = ""
    lowered = set(raw.lower().replace(":", " ").split())
    wanted = ("avx", "avx2", "avx512f", "fma", "neon", "asimd")
    return tuple(feature.upper() for feature in wanted if feature in lowered)
