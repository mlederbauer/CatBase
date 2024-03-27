"""Test catbase."""

import catbase


def test_import() -> None:
    """Test that the package can be imported."""
    assert isinstance(catbase.__name__, str)
