"""catbase CLI."""

import typer

from catbase.data import (
    create_database,
    delete_database,
    inspect_database,
    list_databases,
    parse_arxiv,
    parse_documents,
)

app = typer.Typer()


@app.command()
def hello(name: str = "Fellow Catalysis Enthusiast") -> None:
    """Say hello to NAME."""
    typer.echo(f"Hello {name}!")


@app.command()
def list() -> None:
    """List all databases."""
    typer.echo("Listing databases:")
    typer.echo("-----------------------------------")
    list_databases()
    typer.echo("-----------------------------------")
    typer.echo("That's it! :)")


@app.command()
def pluck(
    database_name: str = typer.Option(..., "-n", "--database-name")
) -> None:
    """Delete a selected database."""
    delete_database(database_name)
    typer.echo(f"Bye bye {database_name}!")


@app.command()
def inspect(
    database_name: str = typer.Option(..., "-n", "--database-name"),
    plot: bool = typer.Option(True, "-p", "--plot"),
) -> None:
    """Inspect vector database in human-readable format."""
    inspect_database(database_name, plot)


@app.command()
def create(
    database_name: str = typer.Option(..., "-n", "--database-name"),
    pdf_directory: str = typer.Option(..., "-d", "--pdf-directory"),
) -> None:
    """Create a new database."""
    typer.echo(f"Creating database {database_name} in {pdf_directory}")
    documents = parse_documents(pdf_directory)
    create_database(database_name, documents)

    typer.echo("Done! :)")


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
    documents = parse_arxiv(keyword_list, max_docs)
    create_database(database_name, documents)

    typer.echo("Done! :)")
