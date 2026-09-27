"""
pytest bootstrap.

Guarantees the project root is importable, so `import src.features`,
`import run_pipeline` and `import eval` resolve to THIS checkout no matter how
pytest is invoked (`pytest`, `python -m pytest`, from a subdirectory, via an
IDE). Previously this worked only as a side effect of `python -m pytest` putting
the current directory on `sys.path`, which meant a bare `pytest` run -- or a
run from anywhere else -- would fail to import the project at all.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
