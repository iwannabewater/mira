"""Top-level Typer application."""

from __future__ import annotations

import typer

from mira.cli.benchmark import benchmark_command
from mira.cli.doctor import doctor_command
from mira.cli.encode import encode_command
from mira.cli.evaluate import evaluate_command
from mira.cli.export import export_command
from mira.cli.inspect import inspect_command
from mira.cli.mine import mine_command
from mira.cli.train import train_command
from mira.cli.validate_data import validate_data_command
from mira.version import __version__

app = typer.Typer(
    add_completion=False,
    help="MIRA multimodal embedding post-training infrastructure.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


def _version_callback(value: bool) -> None:  # noqa: FBT001 - Typer callback contract
    if value:
        typer.echo(__version__)
        raise typer.Exit


@app.callback()
def main(
    version: bool = typer.Option(  # noqa: FBT001 - Typer callback contract
        False,  # noqa: FBT003 - Typer flag default
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the installed MIRA version.",
    ),
) -> None:
    """Run a MIRA workflow."""
    del version


app.command("inspect")(inspect_command)
app.command("validate-data")(validate_data_command)
app.command("benchmark")(benchmark_command)
app.command("doctor")(doctor_command)
app.command("mine")(mine_command)
app.command("evaluate")(evaluate_command)
app.command("encode")(encode_command)
app.command("export")(export_command)
app.command("train")(train_command)
