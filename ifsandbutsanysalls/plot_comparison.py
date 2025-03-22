#!/usr/bin/env python3
import click
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pathlib import Path


@click.command()
@click.argument("first_profile", type=Path)
@click.argument("second_profile", type=Path)
@click.argument("output", type=Path)
def main(first_profile: Path, second_profile: Path, output: Path):
    first_df, first_name = pd.read_json(first_profile), first_profile.name.split(".", 1)[0]
    second_df, second_name = pd.read_json(second_profile), second_profile.name.split(".", 1)[0]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(21, 5))
    for ax, prefix in zip(axes, ["model-forward", "encoder-layer", "clamp"]):
        axmax, axmin = 0, (1 << 32)
        for df, name in zip([first_df, second_df], [first_name, second_name]):
            view = df[df.Name.str.startswith(prefix)]
            if view.empty:
                continue
            time_ms = view["Projected Duration (ns)"] / 1E6
            median = time_ms.median()
            ax.scatter(np.arange(len(time_ms)), time_ms, label=f"{name} (median: {median:.1f} ms)")
            axmax = max(time_ms.max(), axmax)
            axmin = min(time_ms.min(), axmin)
        ax.set_ylim(0.95 * axmin, 1.05 * axmax)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(f"{prefix} time (ms)")
        ax.legend()
    plt.tight_layout()
    plt.savefig(output)


if __name__ == "__main__":
    main()

