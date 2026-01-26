###############################################################################
# Training reconstructions CLI (2025)
#
# Author: Vittoria De Pellegrini
# Affiliation: PhD Student, King Abdullah University of Science and Technology
#
# Description:
#   Command-line entry point that delegates to `dataset_reconstruction.run_cli`
#   with the training dataset split selected by default.
###############################################################################

from pressure_buildup_vae.dataset_reconstruction import run_cli


if __name__ == "__main__":
    run_cli(default_split="train")