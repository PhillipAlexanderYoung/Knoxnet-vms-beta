#!/usr/bin/env bash
#
# no_leak_check.sh — PUBLIC-repo guardrail.
#
# Fails the build if private control-plane / fleet / remote-access code (or an
# obvious secret) reappears in the PUBLIC Knoxnet VMS repo. This protects
# against recurrence of Stage 1 finding S1 (the leaked WireGuard desktop client
# + the dangling private `remote_access_routes` import).
#
# Design notes:
#   * Scans only git-TRACKED text files via `git ls-files`, so venv/, .venv*/,
#     node_modules/, portal/dist/, data/, models/, captures/ etc. are excluded
#     automatically (they are gitignored / not tracked).
#   * The Knoxnet Post planning docs under docs/knoxnet-post/ intentionally
#     DESCRIBE the private architecture (audit + split plan); they are excluded,
#     as are this script and its workflow (they necessarily contain the
#     patterns we search for).
#   * Patterns target SPECIFIC leaked-code constructs, not generic words. e.g.
#     README.md may legitimately mention "WireGuard" as general security advice,
#     so we match the private *endpoint contract* and *client symbols*, never the
#     bare word "wireguard".
#
# Upgrading the secret scan:
#   The secret scan below is a dependency-free grep fallback covering the most
#   common high-signal patterns. To upgrade to a full scanner later, add a
#   gitleaks step to .github/workflows/no-leak-guard.yml using the official
#   action `gitleaks/gitleaks-action@v2` (it requires a GITLEAKS_LICENSE secret
#   for orgs, or runs free on public repos). Keep this script as the fast,
#   zero-dependency first line of defense.
#
# Exit status: 0 = clean, 1 = leak/secret found.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# Tracked paths intentionally excluded from scanning.
EXCLUDE_REGEX='^(docs/knoxnet-post/|scripts/ci/no_leak_check\.sh$|\.github/workflows/no-leak-guard\.yml$)'

mapfile -t FILES < <(git ls-files | grep -vE "$EXCLUDE_REGEX" || true)

if [ "${#FILES[@]}" -eq 0 ]; then
  echo "no-leak-check: no tracked files to scan"
  exit 0
fi

status=0

# scan <human-label> <ERE-pattern>
# grep flags: -I skip binary files, -n line numbers, -E extended regex.
scan() {
  local label="$1" pat="$2" matches
  if matches=$(grep -InE -- "$pat" "${FILES[@]}" 2>/dev/null); then
    echo "::error::no-leak-check FAILED: ${label}"
    echo "${matches}"
    echo ""
    status=1
  fi
}

# ── A. Private remote-access / WireGuard desktop client leak (Stage 1 S1) ──
scan "private remote_access_routes module reference" 'remote_access_routes'
scan "private RA endpoint contract"                  '/api/remote-access'
scan "leaked WireGuard peer-config dialog"           '_PeerConfigDialog'
scan "leaked WireGuard RA client helpers"            '_ra_(api|headers|poll_status|apply_status|apply_peer_bundle|fetch_latest_peer_bundle|on_toggle_enabled|set_enabled|add_peer|open_bundle_dir|open_knoxnet_vpn|windows_try_autostart|host_path|last_status|last_peer_bundle|peer_autocreate|peer_qr_lbl)'
scan "WireGuard client payload / endpoint shape"     'qr_png_base64|wireguard/(status|enable|disable|peers)'

# ── B. Control-plane / fleet code markers (belong in knoxnet-control) ──
scan "control-plane fleet code marker"               'desired_peer_state|agent_checkins?|configuration_revisions|enrollment_token[s]?'

# ── C. Obvious secrets (dependency-free fallback; upgrade to gitleaks later) ──
scan "PEM private key block"                         '-----BEGIN ([A-Z]+ )?PRIVATE KEY-----'
scan "AWS access key id"                             'AKIA[0-9A-Z]{16}'
scan "GitHub token"                                  'gh[pousr]_[A-Za-z0-9]{36,}'
scan "Slack token"                                   'xox[baprs]-[A-Za-z0-9-]{10,}'
scan "WireGuard private key value"                   'PrivateKey[[:space:]]*=[[:space:]]*[A-Za-z0-9+/]{42,}='

if [ "${status}" -ne 0 ]; then
  cat <<'MSG'

no-leak-check: FAILED — private/control-plane or secret patterns found in tracked files.

These patterns must not live in the PUBLIC repo. If a match is a legitimate
public reference (e.g. generic documentation), refine the specific pattern in
scripts/ci/no_leak_check.sh. Otherwise, move the offending code to the private
knoxnet-vms-post (Post agent / WireGuard orchestration) or knoxnet-control
(control plane) repo. See docs/knoxnet-post/STAGE2_SPLIT.md.
MSG
  exit 1
fi

echo "no-leak-check: OK — no private/control-plane leaks or obvious secrets in tracked files."
