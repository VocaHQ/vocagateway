from __future__ import annotations

import contextlib
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

MACOS = "macOS"
APPLE_SILICON = "Apple silicon"
ARM64_ARCHITECTURE = "arm64"
UTF8_ENCODING = "utf-8"
KEY_VALUE_SEPARATOR = ":"
DEVICE_DIRECTORY_NAME = "device"
DEFAULT_VOCAMAC_APPLICATION_PATH = Path("/Applications/VocaMac.app")
GIBIBYTE = 1024**3
MEBIBYTE = 1024
# ggml and CTranslate2 split work evenly across the threads they are given, so a
# thread pinned to an efficiency core (or descheduled by a cgroup quota) holds the
# whole batch back. Threads are therefore counted in physical performance cores,
# clamped by the container quota, and capped: past this point the per-layer sync
# cost outweighs the extra core on every model this catalog ships.
MAXIMUM_INFERENCE_THREADS = 8
# No CPU this gateway targets runs more than two threads per core, so a
# reported topology claiming fewer cores than that is not believable and is
# treated as a floor rather than a count. See `_linux_physical_cpus`.
MAXIMUM_THREADS_PER_CORE = 2

# Engines that cannot run on every host, and the requirement the WebUI shows.
# The desktop-app adapters are the strictest: Handy ships for macOS, and VocaMac
# is Apple-silicon-only, so neither exists on Linux or inside a container.
ENGINE_HOST_REQUIREMENTS = MappingProxyType(
    {
        "vocamac": APPLE_SILICON,
        "handy": MACOS,
        "whisperkit": MACOS,
        "mlx-audio": APPLE_SILICON,
    }
)


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
    transcribe_cli_path: str | None
    whisperkit_cli_path: str | None
    handy_installed: bool
    vocamac_installed: bool
    logical_cpus: int
    effective_cpus: float
    containerized: bool
    accelerators: tuple[str, ...]
    cpu_features: tuple[str, ...]


class _EngineHost:
    @classmethod
    def requirement(cls, engine: str) -> str | None:
        """The host an engine needs, or None when it runs anywhere."""
        return ENGINE_HOST_REQUIREMENTS.get(engine)

    @classmethod
    def runs_on(cls, engine: str, *, is_mac: bool, is_apple_silicon: bool) -> bool:
        requirement = ENGINE_HOST_REQUIREMENTS.get(engine)
        if requirement is None:
            return True
        return is_apple_silicon if requirement == APPLE_SILICON else is_mac

    @classmethod
    def runs_here(cls, engine: str) -> bool:
        """The same check for the running host, without a full `detect_system` probe."""
        is_mac = platform.system() == "Darwin"
        return cls.runs_on(
            engine,
            is_mac=is_mac,
            is_apple_silicon=is_mac and platform.machine() == ARM64_ARCHITECTURE,
        )

    @classmethod
    def drm_labels(cls) -> list[str]:
        root = Path("/sys/class/drm")
        if not root.is_dir():
            return []
        labels: list[str] = []
        seen: set[str] = set()
        for card in sorted(root.glob("card[0-9]*")):
            label = cls._drm_card_label(card)
            if label and label not in seen:
                seen.add(label)
                labels.append(label)
        return labels

    @classmethod
    def _drm_card_label(cls, card: Path) -> str:
        if "-" in card.name:
            return ""
        device = card / DEVICE_DIRECTORY_NAME
        if not cls._is_amd_vendor(device / "vendor"):
            return ""
        name = cls._drm_product_name(device, card.name)
        if name.upper().startswith("AMD"):
            return name
        return f"AMD {name}"

    @classmethod
    def _is_amd_vendor(cls, vendor_path: Path) -> bool:
        try:
            vendor = vendor_path.read_text(encoding=UTF8_ENCODING).strip().lower()
        except OSError:
            return False
        return vendor in {"0x1002", "1002"}

    @classmethod
    def _drm_product_name(cls, device: Path, fallback: str) -> str:
        for candidate in (device / "label", device / "product_name", device / "marketing_name"):
            try:
                name = candidate.read_text(encoding=UTF8_ENCODING).strip()
            except OSError:
                name = ""
            if name:
                return name
        try:
            device_id = (device / DEVICE_DIRECTORY_NAME).read_text(encoding=UTF8_ENCODING).strip()
        except OSError:
            return fallback
        return f"device {device_id}" if device_id else fallback


class _SysProbe:
    @classmethod
    def resolve_binary(cls, binary: str) -> str | None:
        """An operator's binary setting resolved to a path, or None if absent.

        `shutil.which` alone is not enough: it neither expands `~` nor accepts a
        file without the execute bit, so an engine probing with it would report
        "not installed" for a binary the Libraries panel just showed as present.
        Every caller — the panel, the model cards, and the engines themselves —
        must resolve through here so they cannot disagree.
        """
        candidate = Path(binary).expanduser()
        if candidate.is_file():
            return str(candidate)
        return shutil.which(binary)

    @classmethod
    def sysctl(cls, key: str) -> str:
        try:
            command_result = subprocess.run(
                ["sysctl", "-n", key],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        if command_result.returncode != 0:
            return ""
        return command_result.stdout.strip()

    @classmethod
    def ram_gb(cls, is_mac: bool) -> float:
        if is_mac:
            raw = cls.sysctl("hw.memsize")
            if raw.isdigit():
                return round(int(raw) / GIBIBYTE, 1)
            return 0
        return cls._linux_ram_gb()

    @classmethod
    def linux_cpu_brand(cls) -> str:
        try:
            lines = Path("/proc/cpuinfo").read_text(encoding=UTF8_ENCODING).splitlines()
        except OSError:
            return platform.processor() or platform.machine()
        for line in lines:
            if line.lower().startswith(("model name", "hardware")) and KEY_VALUE_SEPARATOR in line:
                return line.split(KEY_VALUE_SEPARATOR, 1)[1].strip()
        return platform.processor() or platform.machine()

    @classmethod
    def is_containerized(cls) -> bool:
        if Path("/.dockerenv").exists() or os.environ.get("CONTAINER"):
            return True
        try:
            cgroup = Path("/proc/1/cgroup").read_text(encoding=UTF8_ENCODING).lower()
        except OSError:
            return False
        return any(marker in cgroup for marker in ("docker", "containerd", "kubepods", "podman"))

    @classmethod
    def _linux_ram_gb(cls) -> float:
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
        except (OSError, ValueError):
            return 0
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError):
            return 0
        if not isinstance(pages, int) or not isinstance(page_size, int):
            return 0
        detected_bytes = pages * page_size
        with contextlib.suppress(OSError, ValueError):
            memory_limit = Path("/sys/fs/cgroup/memory.max").read_text(encoding=UTF8_ENCODING)
            stripped = memory_limit.strip()
            if stripped != "max":
                detected_bytes = min(detected_bytes, int(stripped))
        return round(detected_bytes / GIBIBYTE, 1)


class _CpuSets:
    @classmethod
    def inference_threads(cls, configured: int = 0) -> int:
        """Threads an inference engine should use on this host.

        `configured` is the operator's explicit override from the WebUI; 0 means
        "decide for me", which is the default every engine starts with.
        """
        if configured > 0:
            return configured
        logical = os.cpu_count() or 1
        cores = min(cls.physical_cpu_count(logical), int(cls.effective_cpu_count(logical)) or 1)
        return max(1, min(cores, MAXIMUM_INFERENCE_THREADS))

    @classmethod
    def physical_cpu_count(cls, logical_cpus: int) -> int:
        """Physical cores, preferring performance cores on Apple silicon."""
        if platform.system() == "Darwin":
            return cls._darwin_physical_cpus(logical_cpus)
        return cls._linux_physical_cpus(logical_cpus)

    @classmethod
    def effective_cpu_count(cls, logical_cpus: int) -> float:
        limits = [float(logical_cpus)]
        cls._append_quota(limits)
        cls._append_cpuset(limits)
        return round(min(limits), 2)

    @classmethod
    def _darwin_physical_cpus(cls, logical_cpus: int) -> int:
        for key in ("hw.perflevel0.physicalcpu", "hw.physicalcpu"):
            raw = _SysProbe.sysctl(key)
            if raw.isdigit() and int(raw) > 0:
                return int(raw)
        return logical_cpus

    @classmethod
    def _linux_physical_cpus(cls, logical_cpus: int) -> int:
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text(encoding=UTF8_ENCODING)
        except OSError:
            return logical_cpus
        cores = cls._cpuinfo_cores(cpuinfo)
        if not cores:
            return logical_cpus
        # Some hypervisors report one identical (physical id, core id) pair for
        # every vCPU. Believing that would run a 16-vCPU guest single-threaded,
        # so a count that implies more than two threads per core is discarded.
        return max(len(cores), logical_cpus // MAXIMUM_THREADS_PER_CORE)

    @classmethod
    def _cpuinfo_cores(cls, cpuinfo: str) -> set[tuple[str, str]]:
        """Unique (physical id, core id) pairs, which is one entry per real core."""
        cores: set[tuple[str, str]] = set()
        package = ""
        for line in cpuinfo.splitlines():
            key, _, entry = line.partition(KEY_VALUE_SEPARATOR)
            label = key.strip().lower()
            if label == "processor":
                package = ""
            elif label == "physical id":
                package = entry.strip()
            elif label == "core id":
                cores.add((package, entry.strip()))
        return cores

    @classmethod
    def _append_quota(cls, limits: list[float]) -> None:
        try:
            quota, period = (
                Path("/sys/fs/cgroup/cpu.max").read_text(encoding=UTF8_ENCODING).split()[:2]
            )
        except (OSError, ValueError):
            return
        if quota != "max" and int(period) > 0:
            limits.append(max(0.1, int(quota) / int(period)))

    @classmethod
    def _append_cpuset(cls, limits: list[float]) -> None:
        for candidate in (
            Path("/sys/fs/cgroup/cpuset.cpus.effective"),
            Path("/sys/fs/cgroup/cpuset/cpuset.cpus"),
        ):
            count = cls._cpu_set_count(candidate)
            if count:
                limits.append(float(count))
                return

    @classmethod
    def _cpu_set_count(cls, candidate: Path) -> int:
        try:
            return cls._count_cpu_set(candidate.read_text(encoding=UTF8_ENCODING).strip())
        except (OSError, ValueError):
            return 0

    @classmethod
    def _count_cpu_set(cls, cpu_set_text: str) -> int:
        return sum(cls._cpu_set_part(part) for part in cpu_set_text.split(","))

    @classmethod
    def _cpu_set_part(cls, part: str) -> int:
        if not part:
            return 0
        if "-" not in part:
            int(part)
            return 1
        start, end = part.split("-", 1)
        return max(0, int(end) - int(start) + 1)


class _NvidiaGpus:
    @classmethod
    def labels(cls) -> list[str]:
        if not shutil.which("nvidia-smi"):
            return []
        try:
            command_result = subprocess.run(
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
        if command_result.returncode != 0 or not command_result.stdout.strip():
            return []
        return cls._parse(command_result.stdout)

    @classmethod
    def _parse(cls, stdout: str) -> list[str]:
        labels: list[str] = []
        for line in stdout.splitlines():
            label = cls._line_label(line)
            if label:
                labels.append(label)
        return labels

    @classmethod
    def _line_label(cls, line: str) -> str:
        parts = [part.strip() for part in line.split(",")]
        if not parts or not parts[0]:
            return ""
        name = parts[0]
        if not name.upper().startswith("NVIDIA"):
            name = f"NVIDIA {name}"
        if len(parts) <= 1:
            return name
        with contextlib.suppress(ValueError):
            mem_gb = round(float(parts[1]) / MEBIBYTE, 1)
            return f"{name} ({mem_gb} GB)"
        return name


class _AmdGpus:
    @classmethod
    def labels(cls) -> list[str]:
        rocm = cls._rocm()
        if rocm:
            return rocm
        lspci = cls._lspci()
        if lspci:
            return lspci
        return [label for label in _EngineHost.drm_labels() if "device 0x" not in label.lower()]

    @classmethod
    def _lspci(cls) -> list[str]:
        if not shutil.which("lspci"):
            return []
        try:
            command_result = subprocess.run(
                ["lspci", "-mm"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return []
        if command_result.returncode != 0:
            return []
        return cls._lspci_labels(command_result.stdout)

    @classmethod
    def _rocm(cls) -> list[str]:
        if not shutil.which("rocm-smi"):
            return []
        try:
            command_result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return []
        if command_result.returncode != 0:
            return []
        return cls._rocm_labels(command_result.stdout)

    @classmethod
    def _lspci_labels(cls, stdout: str) -> list[str]:
        labels: list[str] = []
        for line in stdout.splitlines():
            label = cls._lspci_line(line)
            if label and label not in labels:
                labels.append(label)
        return labels

    @classmethod
    def _lspci_line(cls, line: str) -> str:
        if '"VGA compatible controller"' not in line and '"3D controller"' not in line:
            return ""
        if "amd" not in line.lower():
            return ""
        parts = line.split('"')
        device = parts[5].strip() if len(parts) > 5 else ""
        if not device:
            return ""
        if device.upper().startswith("AMD"):
            return device
        return f"AMD {device}"

    @classmethod
    def _rocm_labels(cls, stdout: str) -> list[str]:
        labels: list[str] = []
        for line in stdout.splitlines():
            name = cls._rocm_name(line)
            if name and name not in labels:
                labels.append(name)
        return labels

    @classmethod
    def _rocm_name(cls, line: str) -> str:
        if KEY_VALUE_SEPARATOR not in line or "GPU[" not in line.upper():
            return ""
        name = line.split(KEY_VALUE_SEPARATOR, 1)[1].strip()
        for prefix in ("Card series:", "Card model:", "Device Name:"):
            if name.startswith(prefix):
                name = name[len(prefix) :].strip()
        if not name:
            return ""
        if name.upper().startswith("AMD"):
            return name
        return f"AMD {name}"


class _HostDetector:
    @classmethod
    def detect(
        cls,
        *,
        whisper_binary: Path,
        whisperkit_binary: str,
        handy_binary: Path,
        transcribe_binary: str,
        vocamac_app: Path = DEFAULT_VOCAMAC_APPLICATION_PATH,
    ) -> SystemInfo:
        arch = platform.machine()
        is_mac = platform.system() == "Darwin"
        chip = (
            _SysProbe.sysctl("machdep.cpu.brand_string") if is_mac else _SysProbe.linux_cpu_brand()
        )
        logical_cpus = os.cpu_count() or 1
        whisper_cli = (
            str(whisper_binary) if whisper_binary.is_file() else shutil.which("whisper-cli")
        )
        return SystemInfo(
            os_name=platform.system(),
            os_version=platform.release(),
            arch=arch,
            chip=chip or arch,
            ram_gb=_SysProbe.ram_gb(is_mac),
            is_apple_silicon=is_mac and arch == ARM64_ARCHITECTURE,
            ffmpeg_path=shutil.which("ffmpeg"),
            whisper_cpp_path=whisper_cli,
            transcribe_cli_path=_SysProbe.resolve_binary(transcribe_binary),
            whisperkit_cli_path=_SysProbe.resolve_binary(whisperkit_binary),
            handy_installed=handy_binary.is_file(),
            vocamac_installed=vocamac_app.exists(),
            logical_cpus=logical_cpus,
            effective_cpus=_CpuSets.effective_cpu_count(logical_cpus),
            containerized=_SysProbe.is_containerized(),
            accelerators=cls._accelerators(is_mac, arch),
            cpu_features=cls._cpu_features(is_mac),
        )

    @classmethod
    def _accelerators(cls, is_mac: bool, arch: str) -> tuple[str, ...]:
        accelerator_labels: list[str] = ["CPU"]
        apple = is_mac and arch == ARM64_ARCHITECTURE
        if apple:
            accelerator_labels.append("Metal / Core ML")
        nvidia = _NvidiaGpus.labels()
        amd = _AmdGpus.labels()
        cls._extend_named(accelerator_labels, nvidia, amd, apple)
        return tuple(accelerator_labels)

    @classmethod
    def _extend_named(
        cls,
        labels: list[str],
        nvidia: list[str],
        amd: list[str],
        apple: bool,
    ) -> None:
        if nvidia:
            labels.extend(nvidia)
        elif Path("/dev/nvidia0").exists() or shutil.which("nvidia-smi"):
            labels.append("NVIDIA CUDA")
        if amd:
            labels.extend(amd)
        elif Path("/dev/kfd").exists():
            labels.append("AMD ROCm")
        has_drm = Path("/dev/dri/renderD128").exists()
        unnamed = not nvidia and not amd and not apple
        if has_drm and unnamed:
            labels.append("Vulkan / VAAPI device")

    @classmethod
    def _cpu_features(cls, is_mac: bool) -> tuple[str, ...]:
        if is_mac:
            raw = " ".join(
                (
                    _SysProbe.sysctl("machdep.cpu.features"),
                    _SysProbe.sysctl("machdep.cpu.leaf7_features"),
                )
            )
        else:
            raw = cls._linux_cpuinfo()
        lowered = set(raw.lower().replace(KEY_VALUE_SEPARATOR, " ").split())
        wanted = ("avx", "avx2", "avx512f", "fma", "neon", "asimd")
        return tuple(feature.upper() for feature in wanted if feature in lowered)

    @classmethod
    def _linux_cpuinfo(cls) -> str:
        try:
            return Path("/proc/cpuinfo").read_text(encoding=UTF8_ENCODING)
        except OSError:
            return ""


detect_system = _HostDetector.detect
inference_thread_count = _CpuSets.inference_threads
resolve_binary = _SysProbe.resolve_binary
engine_requirement = _EngineHost.requirement
engine_runs_on = _EngineHost.runs_on
engine_runs_here = _EngineHost.runs_here
