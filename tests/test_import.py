"""Test cat_base."""

import cat_base


def test_import() -> None:
    """Test that the package can be imported."""
    assert isinstance(cat_base.__name__, str)
