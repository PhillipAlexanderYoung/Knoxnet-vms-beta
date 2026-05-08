# Public Release Safety Checklist

Run this before publishing a beta snapshot.

## Repository Shape

- Public repo/folder is named `Knoxnet-vms-beta`.
- Private `Knoxnet-VMS-Collab` repo remains separate and private.
- No `.git` history from the private repo was copied into the public tree.

## Exclusions

Confirm these are absent from the public repo:

- Cloudflare Worker source, Wrangler configs, D1/KV/R2 identifiers, and deployment scripts.
- Private admin tools, customer deployment scripts, internal bootstrap scripts, and release build/signing scripts.
- Paid license, payment, and cloud entitlement implementation.
- PyInstaller specs, installer scripts, code-signing files, and auto-updater/staging code.
- `.env`, local databases, camera credentials, device identity files, logs, captures, recordings, and customer data.

## Secret Scan

Search for these before publishing:

```bash
rg -n "BEGIN (RSA|EC|OPENSSH|PRIVATE) KEY|api[_-]?key|secret|token|password|wrangler|Cloudflare|D1|KV|R2|stripe|customer" .
```

Review matches manually. Example config placeholders are acceptable only when they contain no real secrets.

## Functional Checks

- `VERSION` contains the intended beta version.
- `core/entitlements.py` reports a 4-camera beta entitlement.
- Startup update check points to the public beta endpoint only.
- `install.sh`, `run.sh`, `install.ps1`, `run.ps1`, `install.bat`, and `run.bat` are present.
- README manual update instructions point to `https://github.com/PhillipAlexanderYoung/Knoxnet-vms-beta`.
