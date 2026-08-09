from __future__ import annotations

from html import escape

from app.config import format_host_port, local_webui_url
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
    return f"""
      <div class="page-head">
        <div>
          <h2>Settings</h2>
          <p>Engine, network paths, device tokens, and browser tools for this gateway.</p>
        </div>
      </div>

      <div class="card" id="engine-settings-card">
        <div class="section-heading">
          <h2>Speech engine</h2>
        </div>
        <p class="muted">Local engine that transcribes audio. Choosing a model under
           Models usually sets this for you. Only engines this machine can run are listed.</p>
        <form hx-put="/ui/partials/config" hx-target="#engine-result" hx-swap="innerHTML">
          <div class="settings-grid">
            <label class="settings-field">
              <span>Engine</span>
              <select name="engine">{options}</select>
            </label>
            <label class="settings-field">
              <span>Compute device</span>
              <select name="compute_device">{device_options}</select>
            </label>
            <label class="settings-field">
              <span>Precision</span>
              <select name="compute_type">{precision_options}</select>
            </label>
            <label class="settings-field">
              <span>CPU threads</span>
              <input name="cpu_threads" type="number" min="0" max="256"
                     value="{config.cpu_threads}" aria-describedby="threads-hint" />
            </label>
          </div>
          <p id="threads-hint" class="muted small settings-hint">0 means automatic
             thread count. INT8 is usually fastest on Linux CPUs. VocaMac and Handy
             reuse those Mac apps and their own models.</p>
          <div class="row settings-actions">
            <button type="submit" class="primary">Apply</button>
          </div>
        </form>
        <p id="engine-result" class="muted settings-engine-result">{hint}</p>
      </div>

      <div class="card" id="network-settings-card">
        <div class="section-heading">
          <h2>Network and storage</h2>
        </div>
        <p class="muted">Where this process listens and where it keeps config and
           models on disk.</p>
        <dl class="facts settings-facts">
          <dt>Listener</dt><dd><code>{listener}</code></dd>
          <dt>On this host</dt><dd><code>{local_url}</code></dd>
          {facts}
        </dl>
      </div>

      {tokens_html}

      <div class="card" id="browser-settings-card">
        <div class="section-heading">
          <h2>This browser</h2>
        </div>
        <p class="muted">Diagnostics are a redacted snapshot for bug reports
           (setup, dependencies, hardware, counters). No token, recordings, or
           transcripts. The gateway token lives only in this browser&rsquo;s storage.</p>
        <div class="row settings-actions">
          <button id="download-diagnostics" type="button" class="ghost">
            Download diagnostics
          </button>
          <button id="forget-token" type="button" class="ghost danger">Forget token</button>
        </div>
      </div>
    """
