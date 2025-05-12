from typing import Dict, List, Optional
import matplotlib.pyplot as plt

def plot_ks_statistics(
    ks_stats: Dict[str, List[Optional[float]]],
    title: str = "KS Statistics per Model",
    xlabel: str = "Step",
    ylabel: str = "KS Statistic"
) -> None:
    """Plot KS statistics for multiple models.

    Args:
        ks_stats (Dict[str, List[Optional[float]]]): Dictionary mapping model
            names to lists of KS statistics (may include None).
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
    """
    for model, stats in ks_stats.items():
        steps = list(range(len(stats)))
        # Filter out None values for plotting
        filtered_steps = [s for s, v in zip(steps, stats) if v is not None]
        filtered_stats = [v for v in stats if v is not None]
        plt.plot(
            filtered_steps,
            filtered_stats,
            marker='o',
            markersize=8,
            markerfacecolor='white',
            markeredgewidth=2,
            label=model
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

ks_statistics = {'Gemini-3-4B': [None, 0.22978441326446353, 0.11579178399431678, 0.029569579965277093, 0.011493089039854687, 0.01161738764989012, 0.010739827910453859, 0.002339416317065912, 0.004208931014599349], 'Llama-3.1-8B': [None, 0.3181776925443285, 0.06914372795603801, 0.07086884856631592, 0.019606787139315074, 0.004945623976721769, 0.011342752108941256, 0.004687371292200371, 0.0011140937991934163, 0.0025667777881736575], 'Qwen-2.5-7B': [None, 0.2708753979600419, 0.08288505667739576, 0.09426162674299127, 0.1037408001591767, 0.02093793300903518, 0.004784418519818756, 0.0021170985854290225, 0.00598787202403106], 'Gemini-2.0-Flash': [None, 0.44385260804387594, 0.15176702053522456, 0.019825680749759017, 0.01277788586126316, 0.0021719363575622153, 0.017549209995945864]}
plot_ks_statistics(ks_statistics)