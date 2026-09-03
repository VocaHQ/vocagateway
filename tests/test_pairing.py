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

DEFAULT_GATEWAY_PORT = 8765
MINIMUM_QR_RENDER_LENGTH = 200
LOCAL_GATEWAY_URL = "http://192.168.1.20:8765"
PAIRING_TEST_TOKEN = "test-token-with-at-least-thirty-two-characters"
PUBLIC_URL_ENVIRONMENT_VARIABLE = "VOCAGATEWAY_PUBLIC_URL"
TOKEN_KEY = "token"
VERSION_KEY = "v"


def test_is_ambient_lan_address_matches_pri_aa() -> None:
    assert is_ambient_lan_address(LOCAL_GATEWAY_URL) is True
    assert is_ambient_lan_address("http://10.0.0.5:8765") is True
    assert is_ambient_lan_address("http://172.16.5.5:8765") is True
    assert is_ambient_lan_address("http://100.101.102.103:8765") is True  # Tailscale/CGNAT


def test_is_ambient_lan_address_leaves_host_aaa() -> None:
    assert is_ambient_lan_address("https://homelabone.tail1234.ts.net:8765") is False
    assert is_ambient_lan_address("https://flow.example.com") is False
    assert is_ambient_lan_address("http://8.8.8.8:8765") is False


def test_round_trip_encode_decode() -> None:
    raw = encode_pairing_payload(
        "http://192.168.1.20:8765/",
        PAIRING_TEST_TOKEN,
    )
    payload = json.loads(raw)
    assert payload[VERSION_KEY] == PAIRING_VERSION
    assert payload["url"] == LOCAL_GATEWAY_URL
    assert payload[TOKEN_KEY] == PAIRING_TEST_TOKEN
    decoded = decode_pairing_payload(raw)
    assert decoded.url == LOCAL_GATEWAY_URL
    assert decoded.token == PAIRING_TEST_TOKEN


def test_decode_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        decode_pairing_payload("not-json")
    with pytest.raises(ValueError, match="empty"):
        decode_pairing_payload("   ")
    with pytest.raises(ValueError, match="version"):
        decode_pairing_payload(
            json.dumps(
                {VERSION_KEY: 99, "url": "http://192.168.1.1:8765", TOKEN_KEY: PAIRING_TEST_TOKEN}
            )
        )
    with pytest.raises(ValueError, match="URL"):
        decode_pairing_payload(json.dumps({VERSION_KEY: 1, TOKEN_KEY: PAIRING_TEST_TOKEN}))
    with pytest.raises(ValueError, match=TOKEN_KEY):
        decode_pairing_payload(
            json.dumps({VERSION_KEY: 1, "url": "http://192.168.1.1:8765", TOKEN_KEY: ""})
        )


def test_encode_rejects_public_credentials_aaaa() -> None:
    with pytest.raises(ValueError, match="credentials"):
        encode_pairing_payload(
            "http://user:pass@192.168.1.1:8765",
            PAIRING_TEST_TOKEN,
        )
    with pytest.raises(ValueError, match="query"):
        encode_pairing_payload(
            "http://192.168.1.1:8765?x=1",
            PAIRING_TEST_TOKEN,
        )


def test_qr_svg_contains_path_and_is_svg() -> None:
    payload = encode_pairing_payload(
        "http://192.168.1.75:8765",
        PAIRING_TEST_TOKEN,
    )
    svg = qr_svg_for_payload(payload)
    assert svg.lstrip().startswith("<?xml") or "<svg" in svg
    assert "path" in svg.lower() or "rect" in svg.lower()
    assert len(svg) > MINIMUM_QR_RENDER_LENGTH


def test_qr_ascii_is_multiline_and_dense() -> None:
    payload = encode_pairing_payload(
        "http://192.168.1.75:8765",
        PAIRING_TEST_TOKEN,
    )
    ascii_qr = qr_ascii_for_payload(payload)
    lines = [line for line in ascii_qr.splitlines() if line.strip()]
    assert len(lines) >= 10
    assert len(ascii_qr) > MINIMUM_QR_RENDER_LENGTH
    # Half-block / full-block glyphs from qrcode.print_ascii(invert=True).
    assert any(ch in ascii_qr for ch in ("█", "▀", "▄", "#", "*"))


def test_primary_gateway_base_url_prefers_o_aaaaa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PUBLIC_URL_ENVIRONMENT_VARIABLE, "http://homelab.example:8765")
    assert primary_gateway_base_url(DEFAULT_GATEWAY_PORT) == "http://homelab.example:8765"


def test_default_pairing_url_prefers_saved_a(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PUBLIC_URL_ENVIRONMENT_VARIABLE, LOCAL_GATEWAY_URL)
    assert (
        default_pairing_url(DEFAULT_GATEWAY_PORT, saved_pairing_url="https://dictation.example.com")
        == "https://dictation.example.com"
    )


def test_default_pairing_url_drops_stale_am_a7a5a(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PUBLIC_URL_ENVIRONMENT_VARIABLE, "http://192.168.9.9:8765")
    # 10.0.0.1 is ambient LAN and not in discovered set (override is the only hit).
    assert (
        default_pairing_url(DEFAULT_GATEWAY_PORT, saved_pairing_url="http://10.0.0.1:8765")
        == "http://192.168.9.9:8765"
    )


def test_default_pairing_url_keeps_live_amb_aa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PUBLIC_URL_ENVIRONMENT_VARIABLE, LOCAL_GATEWAY_URL)
    assert (
        default_pairing_url(DEFAULT_GATEWAY_PORT, saved_pairing_url=LOCAL_GATEWAY_URL)
        == LOCAL_GATEWAY_URL
    )


def test_normalize_gateway_input_adds_schem_af165() -> None:
    assert (
        normalize_gateway_input("100.101.102.103", DEFAULT_GATEWAY_PORT)
        == "http://100.101.102.103:8765"
    )


def test_normalize_gateway_input_keeps_expl_aaa() -> None:
    assert (
        normalize_gateway_input("100.101.102.103:9000", DEFAULT_GATEWAY_PORT)
        == "http://100.101.102.103:9000"
    )


def test_normalize_gateway_input_accepts_ta_cd888() -> None:
    assert (
        normalize_gateway_input("phone.tailnet-name.ts.net", DEFAULT_GATEWAY_PORT)
        == "http://phone.tailnet-name.ts.net:8765"
    )


def test_normalize_gateway_input_passes_thr_ee5ba() -> None:
    assert (
        normalize_gateway_input("http://192.168.1.5:8765", DEFAULT_GATEWAY_PORT)
        == "http://192.168.1.5:8765"
    )


def test_normalize_gateway_input_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        normalize_gateway_input("   ", DEFAULT_GATEWAY_PORT)


def test_parse_ifconfig_ipv4_addresses_find_b716f() -> None:
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


def test_parse_ifconfig_ipv4_addresses_find_aaaa() -> None:
    output = """
eth0      Link encap:Ethernet  HWaddr 00:00:00:00:00:00
          inet addr:10.0.0.5  Bcast:10.0.0.255  Mask:255.255.255.0
tailscale0 Link encap:UNSPEC
          inet addr:100.64.1.2  P-t-P:100.64.1.2  Mask:255.255.255.255
"""
    assert _parse_ifconfig_ipv4_addresses(output) == ["10.0.0.5", "100.64.1.2"]


def test_parse_ifconfig_ipv4_addresses_igno_aaaaa() -> None:
    output = "lo0: flags=8049<UP,LOOPBACK>\n\tinet 127.0.0.1 netmask 0xff000000\n"
    assert _parse_ifconfig_ipv4_addresses(output) == []
