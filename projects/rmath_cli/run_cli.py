import sys
import os

# Get the directory of this script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Add the rmath toolkit root to path
sys.path.append(base_dir)

from projects.rmath_cli.main import main

if __name__ == "__main__":
    main()
