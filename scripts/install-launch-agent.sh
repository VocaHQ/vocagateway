#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
# Repo root whether this is a standalone clone or a submodule at .../server.
repository=$(CDPATH= cd -- "$script_dir/.." && pwd)
destination="$HOME/Library/LaunchAgents/com.vocahq.vocagateway.plist"
log_dir="$HOME/Library/Logs/VocaGateway"
template="$script_dir/com.vocahq.vocagateway.plist"
program="$repository/.venv/bin/vocagateway"
domain="gui/$(id -u)"
service="$domain/com.vocahq.vocagateway"

if [ ! -x "$program" ]; then
  printf 'Gateway executable not found at %s\nRun: uv sync in the repository root first.\n' "$program" >&2
  exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents" "$log_dir"
escaped_repository=$(printf '%s' "$repository" | sed 's/[\/&]/\\&/g')
escaped_home=$(printf '%s' "$HOME" | sed 's/[\/&]/\\&/g')
temporary=$(mktemp "${TMPDIR:-/tmp}/vocagateway-launch-agent.XXXXXX")
trap 'rm -f "$temporary"' EXIT
sed \
  -e "s/__REPOSITORY__/$escaped_repository/g" \
  -e "s/__HOME__/$escaped_home/g" \
  "$template" > "$temporary"
mv "$temporary" "$destination"
trap - EXIT

# Drop older LaunchAgent labels so they cannot keep crash-looping after the
# rename (or after localflow-server disappeared from the venv).
for legacy_label in \
  com.vocahq.vocaphone.gateway \
  com.example.localflow.gateway
do
  legacy_service="$domain/$legacy_label"
  legacy_plist="$HOME/Library/LaunchAgents/${legacy_label}.plist"
  if launchctl print "$legacy_service" >/dev/null 2>&1 || [ -f "$legacy_plist" ]; then
    launchctl bootout "$legacy_service" 2>/dev/null || true
    rm -f "$legacy_plist"
    printf 'Removed obsolete LaunchAgent %s\n' "$legacy_label"
  fi
done

launchctl bootout "$service" 2>/dev/null || true
remaining=50
while launchctl print "$service" >/dev/null 2>&1; do
  if [ "$remaining" -eq 0 ]; then
    printf 'Timed out waiting for the previous LaunchAgent to stop.\n' >&2
    exit 1
  fi
  remaining=$((remaining - 1))
  sleep 0.1
done
launchctl bootstrap "$domain" "$destination"
printf 'Installed and started %s\n' "$destination"
