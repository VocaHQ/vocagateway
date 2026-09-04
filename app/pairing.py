"""Phone pairing payload and QR for the authenticated WebUI.

The payload is a compact JSON document the iPhone and Android apps scan:

    {"v":1,"url":"http://192.168.1.20:8765","token":"..."}

The QR must encode a host the phone can reach (LAN / Tailscale), not loopback.
"""

from __future__ import annotations

import ipaddress
import json
import os
import socket
import subprocess
from contextlib import suppress
from dataclasses import dataclass
from io import StringIO
from urllib.parse import ParseResult, urlparse

import qrcode
from qrcode.image import svg as qrcode_svg

PAIRING_VERSION = 1
PAIRING_SCHEME_HINT = "vocaphone-pair-v1"
_HTTP_SCHEMES = frozenset(("http", "https"))
_AMBIENT_LAN_NETWORKS = (
    ipaddress.IPv4Network("100.64.0.0/10"),  # Tailscale / CGNAT, not flagged by is_private
)
_UDP_PROBES = ("1.1.1.1", "8.8.8.8")
_RANKED_NETWORKS = (
    (ipaddress.IPv4Network("192.168.0.0/16"), 0),
    (ipaddress.IPv4Network("172.16.0.0/12"), 1),
    (ipaddress.IPv4Network("100.64.0.0/10"), 2),
    (ipaddress.IPv4Network("10.0.0.0/8"), 3),
)


@dataclass(frozen=True, slots=True)
class PairingPayload:
    url: str
    token: str
    version: int = PAIRING_VERSION

    def encode(self) -> str:
        if self.version != PAIRING_VERSION:
            raise ValueError(f"Unsupported pairing version: {self.version}")
        if not self.token.strip():
            raise ValueError("Pairing token must not be empty.")
        url = normalize_gateway_url(self.url)
        return json.dumps(
            {"v": self.version, "url": url, "token": self.token},
            separators=(",", ":"),
            ensure_ascii=True,
        )


class _PayloadCodec:
    @classmethod
    def encode(cls, url: str, token: str, *, version: int = PAIRING_VERSION) -> str:
        return PairingPayload(url=url, token=token, version=version).encode()

    @classmethod
    def decode(cls, raw: str) -> PairingPayload:
        payload_data = cls._parse_object(raw)
        version = payload_data.get("v", payload_data.get("version"))
        cls._require_version(version)
        url = cls._require_text(payload_data.get("url"), "Pairing code is missing a gateway URL.")
        token = cls._require_text(
            payload_data.get("token"),
            "Pairing code is missing a bearer token.",
        )
        gateway = normalize_gateway_url(url)
        return PairingPayload(url=gateway, token=token.strip(), version=PAIRING_VERSION)

    @classmethod
    def _parse_object(cls, raw: str) -> dict[str, object]:
        text = raw.strip()
        if not text:
            raise ValueError("Pairing code is empty.")
        try:
            payload_data = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError("Pairing code is not valid JSON.") from error
        if not isinstance(payload_data, dict):
            raise ValueError("Pairing code must be a JSON object.")
        return payload_data

    @classmethod
    def _require_version(cls, version: object) -> None:
        if version != PAIRING_VERSION:
            raise ValueError(f"Unsupported pairing version: {version!r}")

    @classmethod
    def _require_text(cls, field: object, message: str) -> str:
        if isinstance(field, str) and field.strip():
            return field
        raise ValueError(message)


class _GatewayUrls:
    @classmethod
    def normalize(cls, url: str) -> str:
        parsed = urlparse(url.strip())
        cls._validate_scheme_host(parsed)
        cls._validate_safe_url(parsed)
        host = cls._bracket_host(parsed.hostname or "")
        port = f":{parsed.port}" if parsed.port else ""
        path = (parsed.path or "").rstrip("/")
        return f"{parsed.scheme}://{host}{port}{path}"

    @classmethod
    def normalize_input(cls, raw: str, default_port: int) -> str:
        trimmed = raw.strip()
        if not trimmed:
            raise ValueError("Address must not be empty.")
        if "://" not in trimmed:
            trimmed = f"http://{trimmed}"
        parsed = urlparse(trimmed)
        if parsed.hostname and parsed.port is None:
            trimmed = cls._with_default_port(parsed, default_port)
        return cls.normalize(trimmed)

    @classmethod
    def is_ambient_lan(cls, url: str) -> bool:
        host = urlparse(url).hostname or ""
        try:
            ip_address = ipaddress.IPv4Address(host)
        except ValueError:
            return False
        return ip_address.is_private or any(
            ip_address in network for network in _AMBIENT_LAN_NETWORKS
        )

    @classmethod
    def _validate_scheme_host(cls, parsed: ParseResult) -> None:
        if parsed.scheme not in _HTTP_SCHEMES:
            raise ValueError("Gateway URL must use http:// or https://.")
        if not parsed.hostname:
            raise ValueError("Gateway URL is missing a host.")

    @classmethod
    def _validate_safe_url(cls, parsed: ParseResult) -> None:
        if parsed.username or parsed.password:
            raise ValueError("Gateway URL must not include credentials.")
        if parsed.query or parsed.fragment:
            raise ValueError("Gateway URL must not include a query or fragment.")

    @classmethod
    def _bracket_host(cls, host: str) -> str:
        if ":" in host and not host.startswith("["):
            return f"[{host}]"
        return host

    @classmethod
    def _with_default_port(cls, parsed: ParseResult, default_port: int) -> str:
        host = cls._bracket_host(parsed.hostname or "")
        path = parsed.path or ""
        return f"{parsed.scheme}://{host}:{default_port}{path}"


class _Ipv4Policy:
    @classmethod
    def is_reachable(cls, address: str) -> bool:
        try:
            ip_address = ipaddress.IPv4Address(address)
        except ValueError:
            return False
        blocked = (
            ip_address.is_loopback
            or ip_address.is_link_local
            or ip_address.is_unspecified
            or ip_address.is_multicast
        )
        return not blocked

    @classmethod
    def rank(cls, address: str) -> tuple[int, str]:
        try:
            ip_address = ipaddress.IPv4Address(address)
        except ValueError:
            return (90, address)
        for network, rank in _RANKED_NETWORKS:
            if ip_address in network:
                return (rank, address)
        fallback = 4 if ip_address.is_private else 5
        return (fallback, address)

    @classmethod
    def parse_ifconfig(cls, output: str) -> list[str]:
        addresses = []
        for line in output.splitlines():
            address = cls._ifconfig_line_address(line)
            if address is not None:
                addresses.append(address)
        return addresses

    @classmethod
    def _ifconfig_line_address(cls, line: str) -> str | None:
        if not line[:1].isspace():
            return None
        stripped = line.strip()
        if not stripped.startswith("inet "):
            return None
        parts = stripped.split()
        if len(parts) < 2:
            return None
        address = parts[1].removeprefix("addr:")
        return address if cls.is_reachable(address) else None


class _AddressCollector:
    def __init__(self) -> None:
        self.found: set[str] = set()

    def collect(self) -> list[str]:
        self._from_hostname()
        for probe in _UDP_PROBES:
            self._from_udp_probe(probe)
        self._from_ip_command()
        self._from_ifconfig()
        return list(self.found)

    def _add(self, address: str) -> None:
        if _Ipv4Policy.is_reachable(address):
            self.found.add(address)

    def _from_hostname(self) -> None:
        try:
            hostname = socket.gethostname()
        except OSError:
            return
        try:
            infos = socket.getaddrinfo(hostname, None, socket.AF_INET, socket.SOCK_STREAM)
        except OSError:
            return
        for address_info in infos:
            sockaddr = address_info[4]
            address = sockaddr[0]
            if isinstance(address, str):
                self._add(address)

    def _from_udp_probe(self, probe: str) -> None:
        with suppress(OSError):
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect((probe, 80))
                address = sock.getsockname()[0]
            if isinstance(address, str):
                self._add(address)

    def _from_ip_command(self) -> None:
        try:
            command_result = subprocess.run(
                ["ip", "-4", "-o", "addr", "show", "scope", "global"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return
        if command_result.returncode != 0:
            return
        for line in command_result.stdout.splitlines():
            parts = line.split()
            if "inet" in parts:
                position = parts.index("inet")
                if position + 1 < len(parts):
                    self._add(parts[position + 1].split("/")[0])

    def _from_ifconfig(self) -> None:
        try:
            command_result = subprocess.run(
                ["ifconfig"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return
        if command_result.returncode == 0:
            self.found.update(_Ipv4Policy.parse_ifconfig(command_result.stdout))


def is_phone_reachable_gateway_url(url: str) -> bool:
    """True when a phone on LAN/VPN can open this gateway base URL.

    Rejects empty hosts, ``localhost``, IPv4/IPv6 loopback, link-local,
    unspecified, and multicast. Non-IP hostnames (MagicDNS, custom DNS) pass.
    """
    host = urlparse(url).hostname
    if not host:
        return False
    if host.lower() == "localhost":
        return False
    try:
        ip_address = ipaddress.ip_address(host)
    except ValueError:
        return True
    if isinstance(ip_address, ipaddress.IPv4Address):
        return _Ipv4Policy.is_reachable(host)
    blocked = (
        ip_address.is_loopback
        or ip_address.is_link_local
        or ip_address.is_unspecified
        or ip_address.is_multicast
    )
    return not blocked


def unreachable_pairing_override() -> tuple[str, str] | None:
    """Return ``(env_key, raw_value)`` when a pairing override is set but unusable.

    Checks ``VOCAGATEWAY_PUBLIC_URL`` then ``VOCAGATEWAY_PAIRING_URL``. A value
    that fails to normalize is skipped; a value that normalizes to a
    non-phone-reachable URL is returned so callers can explain the misconfig.
    """
    for key in ("VOCAGATEWAY_PUBLIC_URL", "VOCAGATEWAY_PAIRING_URL"):
        raw = os.environ.get(key, "").strip()
        if not raw:
            continue
        try:
            normalized = _GatewayUrls.normalize(raw)
        except ValueError:
            continue
        if not is_phone_reachable_gateway_url(normalized):
            return (key, raw)
    return None


class _GatewayDiscovery:
    @classmethod
    def discover(cls, port: int) -> list[str]:
        ranked = sorted(_AddressCollector().collect(), key=_Ipv4Policy.rank)
        discovered = [f"http://{ip_address}:{port}" for ip_address in ranked]
        return cls._unique(cls._overrides() + discovered)

    @classmethod
    def primary(cls, port: int) -> str | None:
        urls = cls.discover(port)
        return urls[0] if urls else None

    @classmethod
    def default_url(cls, port: int, *, saved_pairing_url: str | None = None) -> str | None:
        if saved_pairing_url and cls._keep_saved(port, saved_pairing_url):
            return saved_pairing_url
        return cls.primary(port)

    @classmethod
    def _keep_saved(cls, port: int, saved_pairing_url: str) -> bool:
        if not is_ambient_lan_address(saved_pairing_url):
            return True
        return saved_pairing_url in cls.discover(port)

    @classmethod
    def _overrides(cls) -> list[str]:
        overrides = []
        for key in ("VOCAGATEWAY_PUBLIC_URL", "VOCAGATEWAY_PAIRING_URL"):
            configured_url = os.environ.get(key, "").strip()
            normalized = cls._normalized_override(configured_url)
            if normalized:
                overrides.append(normalized)
        return overrides

    @classmethod
    def _normalized_override(cls, configured_url: str) -> str | None:
        if not configured_url:
            return None
        try:
            normalized = normalize_gateway_url(configured_url)
        except ValueError:
            return None
        if not is_phone_reachable_gateway_url(normalized):
            return None
        return normalized

    @classmethod
    def _unique(cls, candidates: list[str]) -> list[str]:
        ordered: list[str] = []
        for candidate in candidates:
            if candidate not in ordered:
                ordered.append(candidate)
        return ordered


class _QrCodes:
    @classmethod
    def svg(cls, payload: str, *, box_size: int = 6, border: int = 2) -> str:
        qr_code = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=box_size,
            border=border,
            image_factory=qrcode_svg.SvgPathImage,
        )
        cls._fill(qr_code, payload)
        image = qr_code.make_image()
        return str(image.to_string(encoding="unicode"))

    @classmethod
    def ascii(cls, payload: str, *, border: int = 1, invert: bool = True) -> str:
        qr_code = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            border=border,
        )
        cls._fill(qr_code, payload)
        buffer = StringIO()
        qr_code.print_ascii(out=buffer, invert=invert)
        return buffer.getvalue()

    @classmethod
    def _fill(cls, qr_code: qrcode.QRCode[str], payload: str) -> None:
        qr_code.add_data(payload)
        qr_code.make(fit=True)


encode_pairing_payload = _PayloadCodec.encode
decode_pairing_payload = _PayloadCodec.decode
normalize_gateway_url = _GatewayUrls.normalize
normalize_gateway_input = _GatewayUrls.normalize_input
is_ambient_lan_address = _GatewayUrls.is_ambient_lan
discover_gateway_base_urls = _GatewayDiscovery.discover
primary_gateway_base_url = _GatewayDiscovery.primary
default_pairing_url = _GatewayDiscovery.default_url
qr_svg_for_payload = _QrCodes.svg
qr_ascii_for_payload = _QrCodes.ascii
_parse_ifconfig_ipv4_addresses = _Ipv4Policy.parse_ifconfig
