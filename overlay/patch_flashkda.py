#!/usr/bin/env python3
"""No-argument entry point for the FlashKDA chunked-prefill port.

`overlay/patch_flashkda_tp3.py` is upstream's file and takes `--root` and
`--in-place`, which `GLM53_OVERLAY_ORDER` entries cannot carry: the launcher
emits every ordered overlay as a bare `python3 /opt/glm53/<name>.py`, and
`tests/test_launcher_rank_parity.py` enforces that shape. This wrapper is the
ordered entry; it applies the upstream port in place when
`HAREM_KDA_FLASHKDA=1` and does nothing otherwise, so the flag stays the only
switch and the upstream file is used unmodified.

The implementation is mounted as `/opt/glm53/flashkda_impl.py` rather than
`patch_flashkda_tp3.py` on purpose: a mounted `patch_*.py` must itself be an
ordered entry, and this one cannot be.
"""
from __future__ import annotations

import os
import subprocess
import sys

IMPL = "/opt/glm53/flashkda_impl.py"
ROOT = os.environ.get("GLM53_SITE_ROOT", "/usr/local/lib/python3.12/dist-packages")


def main() -> int:
    flag = (os.environ.get("HAREM_KDA_FLASHKDA") or "0").strip()
    if flag != "1":
        return 0
    if not os.path.isfile(IMPL):
        print(f"patch_flashkda: HAREM_KDA_FLASHKDA=1 but {IMPL} is not mounted", file=sys.stderr)
        return 2
    return subprocess.run(
        [sys.executable, IMPL, "--root", ROOT, "--in-place"], check=False
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
