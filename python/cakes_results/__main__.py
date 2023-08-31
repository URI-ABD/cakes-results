"""The plotting CLI."""

import logging
import pathlib

import typer

import cakes_results

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("cakes_results")
logger.setLevel("INFO")

app = typer.Typer()


@app.command()
def main(
    reports_dir: pathlib.Path = typer.Option(
        ...,
        "--reports-dir",
        help="The directory containing the reports.",
        exists=True,
        file_okay=False,
        readable=True,
        resolve_path=True,
    ),
    plots_dir: pathlib.Path = typer.Option(
        ...,
        "--plots-dir",
        help="The directory to save the plots.",
        exists=True,
        file_okay=False,
        writable=True,
        resolve_path=True,
    ),
) -> None:
    """The main entry point of the application."""
    logger.info(f"Loading reports from {reports_dir} ...")
    logger.info(f"Saving plots to {plots_dir} ...")

    report_paths = sorted(filter(lambda p: p.suffix == ".json", reports_dir.iterdir()))

    logger.info(f"Found {len(report_paths)} reports.")

    reports = [
        cakes_results.report.Report.from_path(report_path)
        for report_path in report_paths
    ]

    logger.info(f"Loaded {len(reports)} reports.")

    cakes_results.violins.draw(reports, plots_dir)


if __name__ == "__main__":
    app()
