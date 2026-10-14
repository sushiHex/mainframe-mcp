"""src-layout: make the package importable without an editable install.

Fixtures (deterministic fake models, temporary LanceDB stores) live in
`tests/v2/conftest.py`; the tests at this level need only the import path.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
