"""Minimal pytest shim — supports raises(), mark.parametrize, approx(), and class-based tests."""

from __future__ import annotations

import contextlib
import inspect
import sys
import traceback

_builtins_abs = abs


@contextlib.contextmanager
def raises(exc_type, match=None):
    """Context manager that asserts an exception is raised.

    Usage:
        with pytest.raises(RuntimeError):
            do_something()
    """
    try:
        yield
    except exc_type as e:
        if match is not None:
            import re
            if not re.search(match, str(e)):
                raise AssertionError(
                    f"Expected exception matching '{match}', got: {e}"
                ) from e
        return
    raise AssertionError(f"Expected {exc_type.__name__} to be raised")


def approx(expected, rel=None, abs=None):
    """Approximate comparison helper."""
    return _Approx(expected, rel=rel, abs=abs)


class _Approx:
    def __init__(self, expected, rel=None, abs=None):
        self.expected = expected
        self.rel = rel if rel is not None else 1e-6
        self.abs = abs if abs is not None else 1e-12

    def __eq__(self, other):
        if isinstance(self.expected, (list, tuple)):
            if len(other) != len(self.expected):
                return False
            return all(_Approx(e, self.rel, self.abs) == o
                       for e, o in zip(self.expected, other))
        tol = max(self.abs, self.rel * _builtins_abs(self.expected))
        return _builtins_abs(self.expected - other) <= tol

    def __repr__(self):
        return f"approx({self.expected!r})"


class _Mark:
    """pytest.mark shim."""

    @staticmethod
    def parametrize(argnames, argvalues):
        """Decorator that expands parametrized tests.

        In standalone mode (no pytest runner), this creates multiple test
        functions by suffixing the original function name.
        When run by our simple test runner, parametrized tests are detected
        via the _parametrize attribute.
        """
        def decorator(fn):
            fn._parametrize = (argnames, argvalues)
            return fn
        return decorator


mark = _Mark()


# ── Test runner ─────────────────────────────────────────────────────

def _run_parametrized(func, instance=None):
    """Run a parametrized test function, returning (passed, failed, errors)."""
    passed, failed, errors = 0, 0, []
    argnames, argvalues = func._parametrize
    if isinstance(argnames, str):
        argnames = [n.strip() for n in argnames.split(",")]
    for i, vals in enumerate(argvalues):
        if not isinstance(vals, (list, tuple)):
            vals = (vals,)
        kwargs = dict(zip(argnames, vals))
        try:
            if instance is not None:
                func(instance, **kwargs)
            else:
                func(**kwargs)
            passed += 1
        except Exception as e:
            failed += 1
            errors.append(f"  {func.__name__}[{i}]: {type(e).__name__}: {e}")
    return passed, failed, errors


def _run_module_tests(module):
    """Run all test functions and test classes in a module."""
    total_passed = 0
    total_failed = 0
    all_errors = []

    # Run top-level test functions
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        if hasattr(func, "_parametrize"):
            p, f, errs = _run_parametrized(func)
            total_passed += p
            total_failed += f
            all_errors.extend(errs)
        else:
            try:
                func()
                total_passed += 1
            except Exception as e:
                total_failed += 1
                all_errors.append(f"  {name}: {type(e).__name__}: {e}")

    # Run class-based tests (Test* classes with test_* methods)
    for cls_name, cls in inspect.getmembers(module, inspect.isclass):
        if not cls_name.startswith("Test"):
            continue
        try:
            instance = cls()
        except Exception as e:
            total_failed += 1
            all_errors.append(f"  {cls_name}.__init__: {type(e).__name__}: {e}")
            continue

        for method_name in sorted(dir(cls)):
            if not method_name.startswith("test_"):
                continue
            method = getattr(instance, method_name)
            if not callable(method):
                continue

            if hasattr(method, "_parametrize"):
                p, f, errs = _run_parametrized(method.__func__, instance)
                total_passed += p
                total_failed += f
                all_errors.extend(
                    err.replace(f"  {method.__func__.__name__}", f"  {cls_name}.{method_name}")
                    for err in errs
                )
            else:
                try:
                    method()
                    total_passed += 1
                except Exception as e:
                    total_failed += 1
                    all_errors.append(
                        f"  {cls_name}.{method_name}: {type(e).__name__}: {e}"
                    )

    return total_passed, total_failed, all_errors


def main(test_paths=None):
    """Simple test runner entry point.

    Usage:
        python -c "import pytest; pytest.main()"
        python -c "import pytest; pytest.main(['tests'])"
    """
    import importlib
    import os
    import pkgutil

    if test_paths is None:
        test_paths = ["tests"]

    total_passed = 0
    total_failed = 0
    failed_modules = []

    for path in test_paths:
        if os.path.isdir(path):
            pkg = path.replace(os.sep, ".")
            try:
                package = importlib.import_module(pkg)
            except ImportError:
                sys.path.insert(0, ".")
                package = importlib.import_module(pkg)
            pkg_path = getattr(package, "__path__", [path])
            for _importer, mod_name, _is_pkg in pkgutil.walk_packages(
                pkg_path, prefix=pkg + "."
            ):
                if not mod_name.split(".")[-1].startswith("test_"):
                    continue
                try:
                    mod = importlib.import_module(mod_name)
                    p, f, errs = _run_module_tests(mod)
                    total_passed += p
                    total_failed += f
                    if f > 0:
                        failed_modules.append((mod_name, errs))
                        print(f"FAIL: {mod_name} ({p} passed, {f} failed)")
                        for e in errs:
                            print(e)
                    else:
                        print(f"OK: {mod_name} ({p} passed)")
                except Exception as e:
                    total_failed += 1
                    failed_modules.append((mod_name, [str(e)]))
                    print(f"ERROR: {mod_name} -> {type(e).__name__}: {e}")
                    traceback.print_exc()
        else:
            mod_name = path.replace(os.sep, ".").removesuffix(".py")
            try:
                mod = importlib.import_module(mod_name)
                p, f, errs = _run_module_tests(mod)
                total_passed += p
                total_failed += f
                if f > 0:
                    failed_modules.append((mod_name, errs))
                    print(f"FAIL: {mod_name} ({p} passed, {f} failed)")
                    for e in errs:
                        print(e)
                else:
                    print(f"OK: {mod_name} ({p} passed)")
            except Exception as e:
                total_failed += 1
                print(f"ERROR: {mod_name} -> {type(e).__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_passed} passed, {total_failed} failed")
    print(f"{'='*60}")

    if total_failed > 0:
        sys.exit(1)
