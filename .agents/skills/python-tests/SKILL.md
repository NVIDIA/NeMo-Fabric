---
name: python-tests
description: Python tests for Fabric; use this when writing tests
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Python Test Style

- Pytest is used to run tests.
- Do not add `@pytest.mark.asyncio` to any test. Async tests are automatically detected and run by the async runner; the decorator is unnecessary clutter.
- Do not add a `-> None` return type annotation to test functions. This is not a common convention in pytest and adds unnecessary verbosity.
- When mocking a class, do not define a new class. Use `unittest.mock.MagicMock` or `unittest.mock.AsyncMock`, with the `spec` constructor argument when necessary.
- The name of the mocked class should be prefixed with `mock`, not `fake`.
- Prefer pytest fixtures over helper methods.
- Do not repeat fixtures, if a fixture is needed in multiple test files, place it in a `conftest.py` file.
- When creating a fixture follow this pattern:
  ```python
  @pytest.fixture(name="<fixture_name>"[, scope="<scope>"])
  def <fixture_name>_fixture() -> <return_type>:
      ...
  ```
  Only specify the scope argument when the value is something other than "function".
- Prefer `pytest.mark.parametrize` over creating individual tests for
  different input types.
- If a fixture is needed for a test, but either does not return a value or the value is not used in the test, use the `@pytest.mark.usefixtures` decorator.

## Common Commands

```bash
# Focused test loop
uv run pytest -k "<pattern>"

# Run all tests
uv run pytest
```

## References

- `pyproject.toml`
- `tests/conftest.py`
