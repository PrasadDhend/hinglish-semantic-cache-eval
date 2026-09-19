"""Repo-root conftest: put src/ on sys.path so tests can `import config`,
`from cache import decide`, etc. -- matching how scripts/*.py already import
(see scripts/verify_env.py), without turning src/ into an installed package.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
