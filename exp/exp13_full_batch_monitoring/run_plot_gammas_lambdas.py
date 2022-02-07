"""This script coordinates the execution of the plotting script 
`plot_gammas_lambdas.py` for all configurations. 
"""

import os
import subprocess

from config import CONFIGURATIONS, PLOT_OUTPUT, config_to_config_str

VERBOSE = True
PLOT_SUBDIR = os.path.join(PLOT_OUTPUT, "gammas_lambdas")


def get_plot_savedir():
    """Return subdirectory `PLOT_SUBDIR` for this plotting script"""
    savedir = PLOT_SUBDIR
    os.makedirs(savedir, exist_ok=True)
    return savedir


if __name__ == "__main__":

    for config in CONFIGURATIONS:

        # Construct command line
        config_str = config_to_config_str(config)
        cmd_str = rf"python ./plot_gammas_lambdas.py --config_str {config_str}"

        print(f"config_str = {config_str}")
        if VERBOSE:
            print(f"cmd_str = {cmd_str}")

        subprocess.check_call(cmd_str)
