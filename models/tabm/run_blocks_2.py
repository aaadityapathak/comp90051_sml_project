"""Cache the initial two-block setting in every requested inner split."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.tabm.run_experiment import main

if __name__ == '__main__':
    main(['--phase', 'tune', '--blocks', '2', *sys.argv[1:]])
