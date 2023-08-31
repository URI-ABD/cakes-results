"""Pydantic models for benchmarks and reports from Rust."""

import json
import pathlib

import numpy
import pydantic


class Report(pydantic.BaseModel):
    """Report from Rust."""

    data_name: str
    metric_name: str
    cardinality: int
    dimensionality: int
    shard_sizes: tuple[int, ...]
    num_queries: int
    k: int
    algorithm: str
    elapsed: list[float]

    elapsed_mean: float = 0.0
    elapsed_std: float = 0.0

    throughput_mean: float = 0.0

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        """Calculate additional fields and initialize the Report."""
        elapsed = kwargs.get("elapsed")

        kwargs["elapsed_mean"] = float(numpy.mean(elapsed))
        kwargs["elapsed_std"] = float(numpy.std(elapsed))

        kwargs["throughput_mean"] = float(
            kwargs["num_queries"] / kwargs["elapsed_mean"],
        )

        super().__init__(**kwargs)

    def __str__(self) -> str:
        """A summary of the report."""
        return (
            f"Report(\n"
            f"  data_name={self.data_name},\n"
            f"  metric_name={self.metric_name},\n"
            f"  cardinality={self.cardinality},\n"
            f"  dimensionality={self.dimensionality},\n"
            f"  shard_sizes={self.shard_sizes},\n"
            f"  num_queries={self.num_queries},\n"
            f"  k={self.k},\n"
            f"  algorithm={self.algorithm},\n"
            f"  elapsed_mean={self.elapsed_mean:.3e} seconds,\n"
            f"  elapsed_std={self.elapsed_std:.3e} seconds,\n"
            f"  throughput_mean={self.throughput_mean:.3e} QPS,\n"
            f")"
        )

    @classmethod
    def from_path(cls, path: pathlib.Path) -> "Report":
        """Load a report from a path."""
        with path.open("r") as f:
            return cls(**json.load(f))
