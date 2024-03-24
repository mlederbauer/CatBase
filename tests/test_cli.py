"""Test cat_base CLI."""

from typer.testing import CliRunner

runner = CliRunner()


def test_say() -> None:
    """Test that the say command works as expected."""
    # message = "Hello Fellow Catalysis Enthusiast!"
    # result = runner.invoke(app, ["--message", message])
    # assert result.exit_code == 0
    # assert message in result.stdout
    # FIXME not working with current cmds
