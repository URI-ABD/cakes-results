"""Drawing violin plots."""

import logging
import math
import pathlib
import typing

import pandas
import seaborn
import tqdm
from matplotlib import pyplot

from . import report

logger = logging.getLogger(__file__)
logger.setLevel("INFO")


def draw(
    reports: list[report.Report],
    plots_dir: typing.Optional[pathlib.Path],
) -> None:
    """Draw violin plots from the reports.

    Args:
        reports: The reports to draw violin plots from.
        plots_dir: The directory to save the plots to. If None, the plots will
            be shown instead of saved.
    """
    first = reports[0]
    data_name = first.data_name
    metric_name = first.metric_name
    cardinality = first.cardinality
    dimensionality = first.dimensionality
    num_queries = first.num_queries

    num_shards = len(first.shard_sizes)

    df = pandas.DataFrame(columns=["k", "algorithm", "time"])
    for r in tqdm.tqdm(reports):
        report_df = pandas.DataFrame(
            {
                "k": [r.k] * num_queries,
                "algorithm": [r.algorithm] * num_queries,
                "time": list(map(math.log10, r.elapsed)),
            },
        )

        df = pandas.concat([df, report_df])

    fig: pyplot.Figure = pyplot.figure(figsize=(14, 7), dpi=128)

    ax: pyplot.Axes = seaborn.violinplot(
        data=df,
        x="algorithm",
        y="time",
        hue="k",
        linewidth=0.1,
        cut=0,
    )

    ax.set_ylabel("Time per Query (10 ^ seconds)")
    ax.set_title(
        f"{data_name} - {metric_name} - {num_shards} shard(s) - "
        f"({cardinality} x {dimensionality}) shape",
    )

    fig.tight_layout()

    if plots_dir is None:
        pyplot.show()
    else:
        plot_name = f"{data_name}_{num_shards}.png"
        fig.savefig(plots_dir / plot_name, dpi=128)
