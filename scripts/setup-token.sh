#!/bin/sh
set -eu

config_dir="${XDG_CONFIG_HOME:-$HOME/.config}/vocagateway"
token_file="$config_dir/token"

# Strip leading/trailing whitespace so " " is treated like an unset value,
# matching Settings.from_env().
trim() {
  printf '%s' "$1" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

read_trimmed_file() {
  if [ -f "$1" ]; then
    # Avoid swallowing trailing newlines incorrectly: read whole file then trim.
    trim "$(cat "$1")"
  fi
}

mkdir -p "$config_dir"
chmod 700 "$config_dir"

existing="$(read_trimmed_file "$token_file")"

if [ -z "$existing" ]; then
  umask 077
  openssl rand -base64 48 > "$token_file"
fi

chmod 600 "$token_file"
printf 'Token is stored at %s\n' "$token_file"
