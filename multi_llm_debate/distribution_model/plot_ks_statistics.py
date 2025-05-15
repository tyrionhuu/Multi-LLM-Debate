import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt


def plot_ks_statistics(
    ks_stats: Dict[str, List[Optional[float]]],
    round_number: int,
    title: str = "KS Statistics per Model",
    xlabel: str = "Step",
    ylabel: str = "KS Statistic",
    ks_threshold: Optional[float] = None,
    output_dir: Union[str, Path] = None,
) -> None:
    """Plot KS statistics for multiple models.

    Args:
        ks_stats (Dict[str, List[Optional[float]]]): Dictionary mapping model
            names to lists of KS statistics (may include None).
        round_number (int): Total number of rounds. Number of KS values is
            round_number - 1.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        ks_threshold (Optional[float]): If provided, plot a horizontal dotted
            line at this KS statistic value.
        output_dir (Optional[str]): If provided, save the plot to this directory
            as 'ks_statistics.png'.
    """
    plt.figure(figsize=(10, 6))
    ks_length = round_number - 1
    for model, stats in ks_stats.items():
        # Drop None values and truncate to ks_length
        processed_stats = [v for v in stats[:ks_length] if v is not None]
        if len(processed_stats) < ks_length:
            # Pad with None values to maintain length
            processed_stats += [0] * (ks_length - len(processed_stats))
        steps = list(range(1, len(processed_stats) + 1))
        plt.plot(
            steps,
            processed_stats,
            marker="o",
            markersize=16,
            markerfacecolor="white",
            markeredgewidth=3,
            label=model,
        )
    if ks_threshold is not None:
        plt.axhline(
            y=ks_threshold,
            color="red",
            linestyle=":",
            linewidth=3,
            label=f"KS Threshold ({ks_threshold})",
        )
    # plt.title(title)
    plt.xlabel(xlabel, fontsize=28)
    plt.ylabel(ylabel, fontsize=28)
    plt.xticks(steps, fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=28, loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"{title.replace(' ', '_')}_ks_statistics.png"
        )
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    plt.show()


if __name__ == "__main__":

    ks_statistics = {
        "Gemini-3-4B": [
            None,
            0.22978441326446353,
            0.11579178399431678,
            0.029569579965277093,
            0.011493089039854687,
            0.01161738764989012,
            0.010739827910453859,
            0.002339416317065912,
            0.004208931014599349,
        ],
        "Llama-3.1-8B": [
            None,
            0.3181776925443285,
            0.06914372795603801,
            0.07086884856631592,
            0.019606787139315074,
            0.004945623976721769,
            0.011342752108941256,
            0.004687371292200371,
            0.0011140937991934163,
            0.0025667777881736575,
        ],
        "Qwen-2.5-7B": [
            None,
            0.2708753979600419,
            0.08288505667739576,
            0.09426162674299127,
            0.1037408001591767,
            0.02093793300903518,
            0.004784418519818756,
            0.0021170985854290225,
            0.00598787202403106,
        ],
        "Gemini-2.0-Flash": [
            None,
            0.44385260804387594,
            0.15176702053522456,
            0.019825680749759017,
            0.01277788586126316,
            0.0021719363575622153,
            0.017549209995945864,
        ],
    }
    plot_ks_statistics(
        ks_statistics, round_number=10, ks_threshold=0.05, output_dir="plots"
    )
