
from pathlib import Path

_REAL_PACKAGE = Path(__file__).resolve().parent.parent / "Phaseek_v2"

if not _REAL_PACKAGE.is_dir():
    raise ImportError(
        f"Phaseek v2 package directory not found: {_REAL_PACKAGE}"
    )
  
__path__ = [str(_REAL_PACKAGE)]

_real_init = _REAL_PACKAGE / "__init__.py"

if _real_init.exists():
    exec(
        compile(
            _real_init.read_text(encoding="utf-8"),
            str(_real_init),
            "exec",
        )
    )
