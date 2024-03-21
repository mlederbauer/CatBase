"""cat_base CLI."""

import typer
from cat_base.data import create_database

app = typer.Typer()


@app.command()
def say(message: str = "") -> None:
    """Say a message."""
    typer.echo(message)

def create(database_name:str, pdf_directory: str) -> None:
    """Create a new database."""
    typer.echo(f"Creating database {database_name} in {pdf_directory}")

    create_database(database_name, pdf_directory)
