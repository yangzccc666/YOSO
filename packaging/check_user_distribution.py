"""Fail release builds that accidentally contain owner-only tools or secrets."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PATHS = (
    "license_admin.py", "backend/license_issuer.py",
    "frontend/src/components/LicenseAdminDialog.tsx",
)
FORBIDDEN_CODE = ("LicenseIssuer", "LicenseAdminDialog", "configureLicenseIssuer", "issueLicense", "license_issuer")
SECRET = re.compile(r"YOLO1\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+|-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----|\"privateKey\"\s*:")


def main() -> None:
    failures = [name for name in FORBIDDEN_PATHS if (ROOT / name).exists()]
    for directory in ("backend", "frontend/src", "frontend/dist"):
        for path in (ROOT / directory).rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".ts", ".tsx", ".js", ".css", ".json", ".html"}:
                continue
            text = path.read_text(encoding="utf-8")
            if any(marker in text for marker in FORBIDDEN_CODE) or SECRET.search(text):
                failures.append(str(path.relative_to(ROOT)))
    if failures:
        raise SystemExit("Owner-only content found in user distribution:\n" + "\n".join(sorted(set(failures))))
    print("User distribution verified: no issuer tools, administrator UI, private keys, or activation codes.")


if __name__ == "__main__":
    main()
