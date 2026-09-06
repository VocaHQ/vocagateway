"""Optional, local, conservative transcript cleanup.

Off by default, and off unless an operator installs a model and turns it on.
Audio never reaches this package — only recognised text does — and nothing here
can turn a successful transcription into a failed one.

Import from the modules directly rather than from this package: `base` for the
vocabulary, `service` for the single decision every entry point shares,
`manager` for the runtime that owns the process, `validation` for the checks
that decide whether a candidate is safe to insert.
"""
