from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import sys
import traceback


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    module = importlib.import_module("tests.test_validation_pipeline")
    test_functions = [
        (name, function)
        for name, function in inspect.getmembers(module, inspect.isfunction)
        if name.startswith("test_")
    ]
    passed = 0
    failed = 0
    for name, function in test_functions:
        try:
            function()
            print(f"[PASS] {name}")
            passed += 1
        except Exception:
            print(f"[FAIL] {name}")
            traceback.print_exc()
            failed += 1
    print(f"\nResultat : {passed}/{passed + failed} tests passes.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
