"""Pydantic models for benchmarks and reports from Rust."""

import numpy
import pydantic


class Report(pydantic.BaseModel):
    """Report from Rust."""

    data_name: str
    metric_name: str
    cardinality: int
    dimensionality: int
    shard_sizes: list[int]
    num_queries: int
    k: int
    algorithm: str
    elapsed: list[float]

    elapsed_mean: float = 0.0
    elapsed_std: float = 0.0

    throughput_mean: float = 0.0
    throughput_std: float = 0.0

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        """Calculate additional fields and initialize the Report."""
        elapsed = kwargs.get("elapsed")

        kwargs["elapsed_mean"] = float(numpy.mean(elapsed))
        kwargs["elapsed_std"] = float(numpy.std(elapsed))

        kwargs["throughput_mean"] = float(1.0 / kwargs["elapsed_mean"])
        kwargs["throughput_std"] = float(
            kwargs["elapsed_std"] / (kwargs["elapsed_mean"] ** 2),
        )

        super().__init__(**kwargs)
