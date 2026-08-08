from __future__ import annotations

from html import escape

from app.config import WILDCARD_BIND_HOSTS, format_host_port, local_webui_url
from app.fragments.engine import ENGINE_HINTS, _engine_option_label
from app.fragments.shared import _facts, _select_options
from app.schemas import ConfigResponse


def settings_fragment(
    config: ConfigResponse,
    paths: list[tuple[str, str]],
    bind_host: str,
    port: int,
    tokens_html: str,
) -> str:
    options = "".join(
        f'<option value="{escape(value)}"{" selected" if value == config.engine else ""}>'
        f"{escape(_engine_option_label(value))}</option>"
        for value in config.available_engines
    )
    hint = escape(ENGINE_HINTS.get(config.engine, ""))
    facts = _facts(paths)
    listener = escape(format_host_port(bind_host, port))
    local_url = escape(local_webui_url(bind_host, port))
    device_options = _select_options(
        [("auto", "Auto"), ("cpu", "CPU"), ("cuda", "NVIDIA CUDA")],
        config.compute_device,
    )
    precision_options = _select_options(
        [
            ("auto", "Auto (INT8 CPU / FP16 CUDA)"),
            ("int8", "INT8"),
            ("int8_float16", "INT8 + FP16"),
            ("float16", "FP16"),
            ("float32", "FP32"),
        ],
        config.compute_type,
    )
    exposure_notice = ""
    if bind_host in WILDCARD_BIND_HOSTS:
        exposure_notice = """
          <div class="callout warning compact">
            <strong>All-interface listener</strong>
            <span>Devices on reachable networks can contact the gateway. They still need
              the bearer token for transcription and administration.</span>
          </div>
        """
    return f"""
      <div class="card">
        <h2>Speech engine</h2>
        <p class="muted">Which local engine transcribes incoming audio. Selecting a
          model in the Models tab sets this for you. Only engines this host can run
          are listed; the VocaMac and Handy entries reuse those Mac apps and their
          own downloaded models.</p>
        <form hx-put="/ui/partials/config" hx-target="#engine-result" hx-swap="innerHTML">
          <div class="settings-grid">
            <label><span>Engine</span>
            <select name="engine">{options}</select>
            </label>
            <label><span>Compute device</span>
              <select name="compute_device">
                {device_options}
              </select>
            </label>
            <label><span>Precision</span>
              <select name="compute_type">
                {precision_options}
              </select>
            </label>
            <label><span>CPU threads</span>
              <input name="cpu_threads" type="number" min="0" max="256"
                     value="{config.cpu_threads}" aria-describedby="threads-hint" />
            </label>
          </div>
          <p id="threads-hint" class="muted small">
            Use 0 for an automatic, conservative thread count.
            INT8 is normally fastest on Linux CPUs.</p>
          <div class="row">
            <button type="submit" class="primary">Apply</button>
          </div>
        </form>
        <p id="engine-result" class="muted">{hint}</p>
      </div>
      <div class="card">
        <h2>Network and storage</h2>
        <dl class="facts">
          <dt>Listener</dt><dd>{listener}</dd>
          <dt>On this host</dt><dd>{local_url}</dd>
          {facts}
        </dl>
        {exposure_notice}
      </div>
      {tokens_html}
      <div class="card">
        <h2>This browser</h2>
        <p class="muted">Diagnostics are a redacted snapshot of setup, dependencies, hardware
          and counters for a bug report &mdash; never the token, recordings or transcripts.
          The token itself is stored only here, in this browser.</p>
        <div class="row">
          <button id="download-diagnostics" type="button" class="ghost">
            Download diagnostics</button>
          <button id="forget-token" class="ghost">Forget token</button>
        </div>
      </div>
    """
