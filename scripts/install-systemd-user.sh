#!/bin/sh
# Install a systemd --user unit that keeps the native Linux gateway running.
# Requires: uv sync already done in this checkout (creates .venv/bin/vocagateway).
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
# Repo root whether this is a standalone clone or a submodule at .../server.
repository=$(CDPATH= cd -- "$script_dir/.." && pwd)
unit_name="com.vocahq.vocagateway.service"
template="$script_dir/$unit_name"
program="$repository/.venv/bin/vocagateway"
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
destination="$unit_dir/$unit_name"

if [ ! -x "$program" ]; then
  printf 'Gateway executable not found at %s\nRun: uv sync --all-groups --extra engines\n' \
    "$program" >&2
  exit 1
fi

if ! command -v systemctl >/dev/null 2>&1; then
  printf 'systemctl not found; this helper needs systemd.\n' >&2
  exit 1
fi

mkdir -p "$unit_dir"
escaped_repository=$(printf '%s' "$repository" | sed 's/[\/&]/\\&/g')
temporary=$(mktemp "${TMPDIR:-/tmp}/vocagateway-systemd.XXXXXX")
trap 'rm -f "$temporary"' EXIT
sed -e "s/__REPOSITORY__/$escaped_repository/g" "$template" > "$temporary"
mv "$temporary" "$destination"
trap - EXIT

# Drop older unit names so they cannot keep restarting a missing binary.
for legacy_unit in \
  com.vocahq.vocaphone.gateway.service \
  com.example.localflow.gateway.service
do
  legacy_destination="$unit_dir/$legacy_unit"
  if systemctl --user cat "$legacy_unit" >/dev/null 2>&1 || [ -f "$legacy_destination" ]; then
    systemctl --user disable --now "$legacy_unit" 2>/dev/null || true
    rm -f "$legacy_destination"
    printf 'Removed obsolete systemd unit %s\n' "$legacy_unit"
  fi
done

systemctl --user daemon-reload
systemctl --user enable --now "$unit_name"

printf 'Installed and started %s\n' "$destination"
printf 'Status:  systemctl --user status %s\n' "$unit_name"
printf 'Logs:    journalctl --user -u %s -f\n' "$unit_name"
printf 'Stop:    systemctl --user stop %s\n' "$unit_name"
printf '\nTo keep the gateway running after logout, once per user:\n'
printf '  loginctl enable-linger %s\n' "$(id -un)"
