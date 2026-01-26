###############################################################################
# Utility helpers (2025)
#
# Author: Vittoria De Pellegrini
# Affiliation: PhD Student, King Abdullah University of Science and Technology
###############################################################################

import torch
import random
import numpy as np


# --------------------------------------------------
# Reproducibility: set global random seeds
# --------------------------------------------------
def set_seed(seed):
    """Sets all random seeds to a fixed value and takes out any
    randomness from cuda kernels

    Parameters
    ----------
    seed : :obj:`int`
        Seed number

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True


# --------------------------------------------------
# Metrics: compute R² excluding zero-valued targets
# --------------------------------------------------
def custom_r2_score(y_true, y_pred):
    """
    Custom implementation of R² score excluding zero values in the ground truth.
    """
    nonzero_mask = y_true != 0
    y_true_nonzero = y_true[nonzero_mask]
    y_pred_nonzero = y_pred[nonzero_mask]

    y_mean = np.mean(y_true_nonzero)
    ss_res = np.sum((y_true_nonzero - y_pred_nonzero) ** 2)  # Residual sum of squares
    ss_tot = np.sum((y_true_nonzero - y_mean) ** 2)  # Total sum of squares

    if ss_tot == 0:  # Handle edge case where all y_true_nonzero values are the same
        return 1.0 if ss_res == 0 else 0.0

    return 1 - (ss_res / ss_tot)