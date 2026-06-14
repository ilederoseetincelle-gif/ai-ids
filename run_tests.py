"""
run_tests.py — Lightweight test runner for environments without pytest.

Discovers all test_*.py files in tests/, runs every function/method that
starts with test_, prints a summary. Use real pytest in production:
    pytest tests/ -v
"""
from __future__ import annotations
import importlib.util
import inspect
import sys
import traceback
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

# Ensure stub packages available
sys.path.insert(0, "/tmp/stub_pkgs")


class _Skip(Exception): pass


def get_fixture_value(fn, fixtures_module, monkeypatch_obj=None, tmp_path=None):
    """Resolve fixture parameters by name. Supports our small fixture set."""
    sig = inspect.signature(fn)
    kwargs = {}
    for param in sig.parameters:
        if param == "self":
            continue
        if param == "monkeypatch":
            kwargs[param] = monkeypatch_obj
        elif param == "tmp_path":
            kwargs[param] = tmp_path
        elif hasattr(fixtures_module, param):
            fixture_fn = getattr(fixtures_module, param)
            # Recursively resolve any nested fixtures
            nested = get_fixture_value(fixture_fn, fixtures_module, monkeypatch_obj, tmp_path)
            kwargs[param] = fixture_fn(**nested) if nested else fixture_fn()
        else:
            return None  # unknown fixture
    return kwargs


class MonkeyPatch:
    """Minimal monkeypatch implementation."""
    def __init__(self):
        self._saved = []

    def setattr(self, target, name, value):
        if isinstance(target, str):
            mod_path, attr = target.rsplit(".", 1)
            mod = importlib.import_module(mod_path)
            self._saved.append((mod, attr, getattr(mod, attr)))
            setattr(mod, attr, value)
        else:
            self._saved.append((target, name, getattr(target, name)))
            setattr(target, name, value)

    def undo(self):
        for obj, name, value in reversed(self._saved):
            setattr(obj, name, value)


def run_module(test_file: Path):
    """Run all test_* functions in a test module."""
    spec = importlib.util.spec_from_file_location(test_file.stem, test_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    passed, failed, skipped, errors = 0, 0, 0, []

    # Discover module-level test functions
    tests = []
    for name in dir(module):
        obj = getattr(module, name)
        if name.startswith("test_") and callable(obj):
            tests.append((f"{test_file.stem}::{name}", obj, None))
        elif name.startswith("Test") and inspect.isclass(obj):
            for m_name in dir(obj):
                if m_name.startswith("test_"):
                    m = getattr(obj, m_name)
                    tests.append((f"{test_file.stem}::{name}::{m_name}", m, obj))

    for full_name, test_fn, klass in tests:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            mp = MonkeyPatch()
            try:
                tmp_path = Path(td)
                kwargs = get_fixture_value(test_fn, module, mp, tmp_path)
                if kwargs is None:
                    print(f"  SKIP  {full_name}  (unresolvable fixture)")
                    skipped += 1
                    continue
                instance = klass() if klass else None
                if instance is not None:
                    test_fn(instance, **kwargs)
                else:
                    test_fn(**kwargs)
                print(f"  PASS  {full_name}")
                passed += 1
            except _Skip as e:
                print(f"  SKIP  {full_name}  ({e})")
                skipped += 1
            except AssertionError as e:
                print(f"  FAIL  {full_name}")
                print(f"        {type(e).__name__}: {e}")
                failed += 1
                errors.append((full_name, traceback.format_exc()))
            except Exception as e:
                print(f"  ERR   {full_name}")
                print(f"        {type(e).__name__}: {e}")
                failed += 1
                errors.append((full_name, traceback.format_exc()))
            finally:
                mp.undo()

    return passed, failed, skipped, errors


def main():
    test_dir = PROJECT / "tests"
    test_files = sorted(test_dir.glob("test_*.py"))

    print("=" * 72)
    print(" AI-IDS Test Suite")
    print("=" * 72)

    total_pass, total_fail, total_skip, all_errors = 0, 0, 0, []
    for tf in test_files:
        print(f"\n{tf.name}")
        print("-" * 72)
        p, f, s, e = run_module(tf)
        total_pass += p
        total_fail += f
        total_skip += s
        all_errors.extend(e)

    print("\n" + "=" * 72)
    print(f" SUMMARY: {total_pass} passed, {total_fail} failed, {total_skip} skipped")
    print("=" * 72)

    if all_errors:
        print("\n--- Failure details ---")
        for name, tb in all_errors[:5]:
            print(f"\n[{name}]")
            print(tb)
        if len(all_errors) > 5:
            print(f"\n... and {len(all_errors) - 5} more")

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
