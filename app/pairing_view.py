from __future__ import annotations

from typing import Literal

from starlette.status import HTTP_400_BAD_REQUEST, HTTP_503_SERVICE_UNAVAILABLE

from app.context import BOOTSTRAP_TOKEN_ID, GatewayContext
from app.errors import APIProblem
from app.fragments.pairing import PairingFragmentData, pairing_fragment, redact_token
from app.pairing import (
    discover_gateway_base_urls,
    encode_pairing_payload,
    is_ambient_lan_address,
    normalize_gateway_input,
    primary_gateway_base_url,
    qr_svg_for_payload,
    unreachable_pairing_override,
)


def resolve_pairing_token(
    ctx: GatewayContext, token_id: str | None
) -> tuple[str, str, Literal["ok", "stale", "unknown"], str | None]:
    """Return (resolved_id, plaintext, status, requested_label).

    `stale` means the token still exists (see it under Settings) but this
    process never cached its plaintext — normally because it was created
    before the gateway last restarted. `unknown` means no such token
    exists at all (typically already revoked). Both fall back to the
    bootstrap token for the QR actually shown; only `stale` can be fixed
    by rotating the token to give it a fresh, displayable secret.
    """
    if token_id and token_id != BOOTSTRAP_TOKEN_ID:
        cached = ctx.token_store.cached_plaintext(token_id)
        if cached is not None:
            return token_id, cached, "ok", None
        existing = ctx.token_store.get(token_id)
        if existing is not None:
            return BOOTSTRAP_TOKEN_ID, ctx.settings.token, "stale", existing.label
        return BOOTSTRAP_TOKEN_ID, ctx.settings.token, "unknown", None
    return BOOTSTRAP_TOKEN_ID, ctx.settings.token, "ok", None


def pairing_token_options(ctx: GatewayContext) -> list[tuple[str, str]]:
    cached_ids = {token.id for token in ctx.token_store.cached_entries()}
    options = [(BOOTSTRAP_TOKEN_ID, "Bootstrap token")]
    options.extend(
        (
            token.id,
            token.label if token.id in cached_ids else f"{token.label} (paired; no live QR)",
        )
        for token in reversed(ctx.token_store.all())
    )
    return options


def forget_stale_lan_addresses(ctx: GatewayContext, discovered: list[str]) -> None:
    """Drop any remembered LAN/tailnet IP no longer part of fresh discovery.

    A bare LAN or Tailscale IP reflects whichever network the gateway was
    on when it was picked; keeping it around after switching Wi-Fi just
    clutters the address list and dropdown with a dead entry. Hostnames
    (MagicDNS names, custom domains) and public IPs are never touched —
    the user chose those deliberately and they aren't tied to one network.
    """
    if ctx.engine_manager is None:
        return
    pairing_config = ctx.pairing_config
    changed = False
    if (
        pairing_config.pairing_url
        and pairing_config.pairing_url not in discovered
        and is_ambient_lan_address(pairing_config.pairing_url)
    ):
        pairing_config.pairing_url = None
        changed = True
    remaining = [
        url
        for url in pairing_config.pairing_urls
        if url in discovered or not is_ambient_lan_address(url)
    ]
    if len(remaining) != len(pairing_config.pairing_urls):
        pairing_config.pairing_urls = remaining
        changed = True
    if changed:
        pairing_config.save(ctx.config_path)


class PairingPresenter:
    def __init__(self, ctx: GatewayContext) -> None:
        self.ctx = ctx
        self._discovered = discover_gateway_base_urls(ctx.settings.port)
        forget_stale_lan_addresses(ctx, self._discovered)

    def render_html(
        self,
        selected_url: str | None = None,
        token_id: str | None = None,
        *,
        persist: bool = False,
    ) -> str:
        candidates = self._candidates()
        selected, candidates = self._select_url(selected_url, candidates, persist)
        token_info = resolve_pairing_token(self.ctx, token_id)
        token = token_info[1]
        svg = qr_svg_for_payload(encode_pairing_payload(selected, token)) if selected else ""
        return pairing_fragment(
            PairingFragmentData(
                selected_url=selected,
                candidates=candidates,
                token_redacted=redact_token(token),
                token_plaintext=token,
                qr_svg=svg,
                saved_urls=self.ctx.pairing_config.pairing_urls,
                token_options=pairing_token_options(self.ctx),
                selected_token_id=token_info[0],
                token_status=token_info[2],
                requested_token_id=token_id or "",
                requested_token_label=token_info[3],
            )
        )

    def forget_url(self, url: str) -> str:
        try:
            normalized = normalize_gateway_input(url, self.ctx.settings.port)
        except ValueError as error:
            raise APIProblem(HTTP_400_BAD_REQUEST, "invalid_pairing_url", str(error)) from error
        cfg = self.ctx.pairing_config
        if self.ctx.engine_manager is not None and normalized in cfg.pairing_urls:
            cfg.pairing_urls.remove(normalized)
            if cfg.pairing_url == normalized:
                cfg.pairing_url = None
            cfg.save(self.ctx.config_path)
        return self.render_html()

    def _candidates(self) -> list[str]:
        candidates = list(self._discovered)
        for saved in self.ctx.pairing_config.pairing_urls:
            if saved not in candidates:
                candidates.append(saved)
        return candidates

    def _select_url(
        self, selected_url: str | None, candidates: list[str], persist: bool
    ) -> tuple[str | None, list[str]]:
        cfg = self.ctx.pairing_config
        selected: str | None = None
        if selected_url:
            try:
                selected = normalize_gateway_input(selected_url, self.ctx.settings.port)
            except ValueError:
                selected = None
            if selected and persist:
                self._persist_url(selected)
        if not selected:
            selected = cfg.pairing_url or primary_gateway_base_url(self.ctx.settings.port)
        if selected and selected not in candidates:
            candidates.insert(0, selected)
        return selected, candidates

    def _persist_url(self, selected: str) -> None:
        if self.ctx.engine_manager is None:
            return
        cfg = self.ctx.pairing_config
        if selected not in self._discovered and selected not in cfg.pairing_urls:
            cfg.pairing_urls.append(selected)
        if cfg.pairing_url != selected:
            cfg.pairing_url = selected
            cfg.save(self.ctx.config_path)


def pairing_html(
    ctx: GatewayContext,
    selected_url: str | None = None,
    token_id: str | None = None,
    *,
    persist: bool = False,
) -> str:
    return PairingPresenter(ctx).render_html(selected_url, token_id, persist=persist)


def forget_pairing_url(ctx: GatewayContext, url: str) -> str:
    return PairingPresenter(ctx).forget_url(url)


def resolve_pairing_url(ctx: GatewayContext, url: str | None) -> str:
    candidates = discover_gateway_base_urls(ctx.settings.port)
    if url:
        try:
            return normalize_gateway_input(url, ctx.settings.port)
        except ValueError as error:
            raise APIProblem(HTTP_400_BAD_REQUEST, "invalid_pairing_url", str(error)) from error
    if not candidates:
        detail = "No phone-reachable gateway address was detected."
        unreachable = unreachable_pairing_override()
        if unreachable is not None:
            key, value = unreachable
            detail = (
                f"{detail} {key}={value} is loopback/link-local and cannot be used for pairing."
            )
        detail = f"{detail} Set a phone-reachable VOCAGATEWAY_PUBLIC_URL and retry."
        raise APIProblem(
            HTTP_503_SERVICE_UNAVAILABLE,
            "pairing_unavailable",
            detail,
        )
    return candidates[0]
