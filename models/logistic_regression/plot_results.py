"""Rebuild logistic-regression figures from saved fold summaries, without fitting."""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from afib.preprocessing import SCENARIOS


def plot_results(output, destination=None):
    """Plot the primary threshold using the recorded outer-fold mean and SD."""
    output = Path(output)
    destination = Path(destination) if destination is not None else PROJECT / 'results/figures/logistic_regression'
    summary = pd.read_csv(output / 'output_data/summary.csv')
    selected = summary[summary.threshold_policy.eq('fixed_0.50') & summary.variant.isin(list('ABCD'))]
    if selected.empty:
        raise ValueError('The result table contains no primary-threshold A-D results.')
    destination.mkdir(parents=True, exist_ok=True)
    scenarios = list(SCENARIOS)
    for metric in ('macro_f1', 'sensitivity', 'brier'):
        figure, axis = plt.subplots(figsize=(12, 5))
        for offset, arm in enumerate('ABCD'):
            rows = selected[selected.metric.eq(metric) & selected.variant.eq(arm)]
            if rows.empty:
                continue
            rows = rows.set_index('scenario').reindex(scenarios)
            # Offsets separate error bars; every point still refers to the
            # same corruption condition. SD is across outer-fold means.
            axis.errorbar(np.arange(len(scenarios)) + (offset - 1.5) * .06,
                          rows['mean'], yerr=rows['sd'], marker='o', capsize=3, label=arm)
        axis.set_xticks(np.arange(len(scenarios)), scenarios, rotation=35, ha='right')
        axis.set_ylabel(metric)
        axis.set_title('Logistic regression: outer-fold mean +/- sample SD')
        axis.legend(title='Arm')
        axis.grid(alpha=.2)
        figure.tight_layout()
        target = destination / f'{metric}.png'
        figure.savefig(target, dpi=160)
        plt.close(figure)
        print(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT / 'results/logistic_regression',
                        help='Saved experiment directory containing output_data/summary.csv.')
    parser.add_argument('--destination', type=Path,
                        help='Figure directory; defaults to results/figures/logistic_regression.')
    args = parser.parse_args()
    plot_results(args.output, args.destination)


if __name__ == '__main__':
    main()
