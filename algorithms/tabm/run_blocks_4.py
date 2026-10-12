"""Cache the initial four-block setting in every requested inner split."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from algorithms.tabm.run_experiment import main

if __name__ == '__main__':
    main(['--phase', 'tune', '--blocks', '4', *sys.argv[1:]])
