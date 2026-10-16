"""Prepare external PTB-XL inputs without adding datasets to the submission."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))

from afib.data_preparation import prepare_dataset
from afib.paths import data_directory, processed_directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=data_directory(PROJECT),
                        help='Raw source folder; defaults to AFIB_DATA_DIR or ../afib_data/raw.')
    parser.add_argument('--processed-dir', type=Path, default=processed_directory(PROJECT),
                        help='External output folder; defaults to AFIB_PROCESSED_DIR or ../afib_data/processed.')
    args = parser.parse_args(argv)
    manifest = prepare_dataset(PROJECT, data_dir=args.data_dir, processed_dir=args.processed_dir)
    print(f"Prepared {manifest['rows']:,} ECGs from {manifest['patients']:,} patients.")
    print(f'Prepared data: {args.processed_dir.resolve()}')
    print('Training-fold imputation, encoding and scaling are fitted later by each model runner.')


if __name__ == '__main__':
    main()
