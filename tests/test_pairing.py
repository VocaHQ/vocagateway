from __future__ import annotations

import json

import pytest

from app.pairing import (
    PAIRING_VERSION,
    _parse_ifconfig_ipv4_addresses,
    decode_pairing_payload,
    default_pairing_url,
    encode_pairing_payload,
    is_ambient_lan_address,
    normalize_gateway_input,
    primary_gateway_base_url,
    qr_ascii_for_payload,
    qr_svg_for_payload,
)


def test_is_ambient_lan_address_matches_private_and_tailnet_ips() -> None:
    assert is_ambient_lan_address("http://192.168.1.20:8765") is True
    assert is_ambient_lan_address("http://10.0.0.5:8765") is True
    assert is_ambient_lan_address("http://172.16.5.5:8765") is True
    assert is_ambient_lan_address("http://100.101.102.103:8765") is True  # Tailscale/CGNAT


def test_is_ambient_lan_address_leaves_hostnames_and_public_ips_alone() -> None:
    assert is_ambient_lan_address("https://homelabone.tail1234.ts.net:8765") is False
    assert is_ambient_lan_address("https://flow.example.com") is False
    assert is_ambient_lan_address("http://8.8.8.8:8765") is False


def test_round_trip_encode_decode() -> None:
    raw = encode_pairing_payload(
        "http://192.168.1.20:8765/",
        "test-token-with-at-least-thirty-two-characters",
    )
    data = json.loads(raw)
    assert data["v"] == PAIRING_VERSION
    assert data["url"] == "http://192.168.1.20:8765"
    assert data["token"] == "test-token-with-at-least-thirty-two-characters"
    decoded = decode_pairing_payload(raw)
    assert decoded.url == "http://192.168.1.20:8765"
    assert decoded.token == "test-token-with-at-least-thirty-two-characters"


def test_decode_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        decode_pairing_payload("not-json")
    with pytest.raises(ValueError, match="empty"):
        decode_pairing_payload("   ")
    with pytest.raises(ValueError, match="version"):
        decode_pairing_payload(
            '{"v":99,"url":"http://192.168.1.1:8765",'
            '"token":"test-token-with-at-least-thirty-two-characters"}'
        )
    with pytest.raises(ValueError, match="URL"):
        decode_pairing_payload('{"v":1,"token":"test-token-with-at-least-thirty-two-characters"}')
    with pytest.raises(ValueError, match="token"):
        decode_pairing_payload('{"v":1,"url":"http://192.168.1.1:8765","token":""}')


def test_encode_rejects_public_credentials_and_query() -> None:
    with pytest.raises(ValueError, match="credentials"):
        encode_pairing_payload(
            "http://user:pass@192.168.1.1:8765",
            "test-token-with-at-least-thirty-two-characters",
        )
    with pytest.raises(ValueError, match="query"):
        encode_pairing_payload(
            "http://192.168.1.1:8765?x=1",
            "test-token-with-at-least-thirty-two-characters",
        )


def test_qr_svg_contains_path_and_is_svg() -> None:
    payload = encode_pairing_payload(
        "http://192.168.1.75:8765",
        "test-token-with-at-least-thirty-two-characters",
    )
    svg = qr_svg_for_payload(payload)
    assert svg.lstrip().startswith("<?xml") or "<svg" in svg
    assert "path" in svg.lower() or "rect" in svg.lower()
    assert len(svg) > 200


def test_qr_ascii_is_multiline_and_dense() -> None:
    payload = encode_pairing_payload(
        "http://192.168.1.75:8765",
        "test-token-with-at-least-thirty-two-characters",
    )
    ascii_qr = qr_ascii_for_payload(payload)
    lines = [line for line in ascii_qr.splitlines() if line.strip()]
    assert len(lines) >= 10
    assert len(ascii_qr) > 200
    # Half-block / full-block glyphs from qrcode.print_ascii(invert=True).
    assert any(ch in ascii_qr for ch in ("█", "▀", "▄", "#", "*"))


def test_primary_gateway_base_url_prefers_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOCAGATEWAY_PUBLIC_URL", "http://homelab.example:8765")
    assert primary_gateway_base_url(8765) == "http://homelab.example:8765"


def test_default_pairing_url_prefers_saved_non_ambient_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VOCAGATEWAY_PUBLIC_URL", "http://192.168.1.20:8765")
    assert (
        default_pairing_url(8765, saved_pairing_url="https://dictation.example.com")
        == "https://dictation.example.com"
    )


def test_default_pairing_url_drops_stale_ambient_lan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VOCAGATEWAY_PUBLIC_URL", "http://192.168.9.9:8765")
    # 10.0.0.1 is ambient LAN and not in discovered set (override is the only hit).
    assert (
        default_pairing_url(8765, saved_pairing_url="http://10.0.0.1:8765")
        == "http://192.168.9.9:8765"
    )


def test_default_pairing_url_keeps_live_ambient_lan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VOCAGATEWAY_PUBLIC_URL", "http://192.168.1.20:8765")
    assert (
        default_pairing_url(8765, saved_pairing_url="http://192.168.1.20:8765")
        == "http://192.168.1.20:8765"
    )


def test_normalize_gateway_input_adds_scheme_and_default_port() -> None:
    assert normalize_gateway_input("100.101.102.103", 8765) == "http://100.101.102.103:8765"


def test_normalize_gateway_input_keeps_explicit_port() -> None:
    assert normalize_gateway_input("100.101.102.103:9000", 8765) == "http://100.101.102.103:9000"


def test_normalize_gateway_input_accepts_tailscale_hostname() -> None:
    assert (
        normalize_gateway_input("phone.tailnet-name.ts.net", 8765)
        == "http://phone.tailnet-name.ts.net:8765"
    )


def test_normalize_gateway_input_passes_through_full_url() -> None:
    assert normalize_gateway_input("http://192.168.1.5:8765", 8765) == "http://192.168.1.5:8765"


def test_normalize_gateway_input_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        normalize_gateway_input("   ", 8765)


def test_parse_ifconfig_ipv4_addresses_finds_macos_tailscale_interface() -> None:
    # Trimmed macOS `ifconfig` output: a LAN NIC plus a Tailscale utun.
    output = """
lo0: flags=8049<UP,LOOPBACK,RUNNING,MULTICAST> mtu 16384
\tinet 127.0.0.1 netmask 0xff000000
en0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tinet 192.168.0.211 netmask 0xffffff00 broadcast 192.168.0.255
utun2: flags=8051<UP,POINTOPOINT,RUNNING,MULTICAST> mtu 1280
\tinet 100.89.197.121 --> 100.89.197.121 netmask 0xffffff00
"""
    assert _parse_ifconfig_ipv4_addresses(output) == ["192.168.0.211", "100.89.197.121"]


def test_parse_ifconfig_ipv4_addresses_finds_linux_net_tools_interface() -> None:
    output = """
eth0      Link encap:Ethernet  HWaddr 00:00:00:00:00:00
          inet addr:10.0.0.5  Bcast:10.0.0.255  Mask:255.255.255.0
tailscale0 Link encap:UNSPEC
          inet addr:100.64.1.2  P-t-P:100.64.1.2  Mask:255.255.255.255
"""
    assert _parse_ifconfig_ipv4_addresses(output) == ["10.0.0.5", "100.64.1.2"]


def test_parse_ifconfig_ipv4_addresses_ignores_loopback() -> None:
    output = "lo0: flags=8049<UP,LOOPBACK>\n\tinet 127.0.0.1 netmask 0xff000000\n"
    assert _parse_ifconfig_ipv4_addresses(output) == []
