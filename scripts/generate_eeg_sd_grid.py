import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.evaluation.generate_eeg_sd_grid import main


if __name__ == "__main__":
    main()
