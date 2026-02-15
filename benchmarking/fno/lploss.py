"""
###############################################################################
# Benchmarking Python Module (2026)
#
# Author: Vittoria De Pellegrini
# Affiliation: PhD Student, King Abdullah University of Science and Technology
#
# Description: fno/lploss.py
#              Baseline benchmarking source file for training, modeling,
#              evaluation, or utilities in the benchmarking workflow.
###############################################################################
"""

import torch


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super().__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
