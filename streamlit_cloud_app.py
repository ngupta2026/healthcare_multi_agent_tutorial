from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_import_path() -> None:
    repo_root = Path(__file__).resolve().parent
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_bootstrap_import_path()

from healthcare_support_agents.streamlit_app import main


if __name__ == "__main__":
    main()
