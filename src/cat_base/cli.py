"""cat_base CLI."""

import typer

from cat_base.data import (
    create_database,
    create_database_from_arxiv,
    list_databases,
)

app = typer.Typer()


@app.command()
def say(message: str = "") -> None:
    """Say a message."""
    typer.echo(message)


@app.command()
def create(
    database_name: str = typer.Option(..., "-n", "--database-name"),
    pdf_directory: str = typer.Option(..., "-d", "--pdf-directory"),
) -> None:
    """Create a new database."""
    typer.echo(f"Creating database {database_name} in {pdf_directory}")
    create_database(database_name, pdf_directory)
    typer.echo("Done! :)")


@app.command()
def list() -> None:
    """List all databases."""
    typer.echo("Listing databases:")

    list_databases()


@app.command()
def arxiv(
    database_name: str = typer.Option(..., "-n", "--database-name"),
    keyword_list: str = typer.Option(..., "-k", "--keyword-list"),
    max_docs: int = typer.Option(10, "-m", "--max-docs"),
) -> None:
    """Create a new database from arXiv."""
    typer.echo(
        f"Creating database {database_name} from arXiv with keywords {keyword_list}"
    )
    create_database_from_arxiv(database_name, keyword_list, max_docs)
    typer.echo("Done! :)")
