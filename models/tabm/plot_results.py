"""Rebuild TabM figures from saved result tables without data or retraining."""

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'src'))
from models.tabm.reporting import _figures, completed_locations, read_json


def plot_results(output, destination=None):
    """Reuse the experiment's figure code while preserving recorded artifacts."""
    output = Path(output)
    destination = Path(destination) if destination is not None else PROJECT / 'results/figures/tabm'
    config = read_json(output / 'run_config.json')
    data = output / 'output_data'
    summary = pd.read_csv(data / 'summary.csv')
    reliability = pd.read_csv(data / 'reliability.csv')
    progress = pd.read_csv(data / 'tuning_progress.csv')
    histories = pd.read_csv(data / 'training_history.csv')
    # Completion markers determine the progress chart; models are never loaded.
    locations, _ = completed_locations(output, config)
    destination.mkdir(parents=True, exist_ok=True)
    _figures(destination, config, summary, reliability, progress, histories, locations)
    for path in sorted(destination.glob('*.png')):
        print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT / 'results/tabm',
                        help='Saved TabM experiment directory.')
    parser.add_argument('--destination', type=Path,
                        help='Figure directory; defaults to results/figures/tabm.')
    args = parser.parse_args()
    plot_results(args.output, args.destination)


if __name__ == '__main__':
    main()
