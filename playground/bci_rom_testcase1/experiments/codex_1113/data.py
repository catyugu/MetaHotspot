"""Bind the current asymmetric model and independent evaluation parameters."""

import sys
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp

CASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CASE))
from model_case1 import Case1Config, Case1Model  # noqa: E402

NOMINAL = np.array([100.0, 1000.0, 30.0])
POWER = np.full(4, 5.0)


def load_model():
    return Case1Model(Case1Config(max_cell_mm=0.5))


def matrix(model, physical_h):
    A = model.core.K.copy()
    for p, H in zip(
        model.physical_to_effective(physical_h), model.boundary_terms, strict=True
    ):
        A = A + p * H
    return A.tocsr()


def evaluation_points():
    from itertools import product

    rng = np.random.default_rng(1113)
    return np.vstack(
        [
            NOMINAL,
            list(product((1.0, 1.0e4), repeat=3)),
            10.0 ** rng.uniform(0.0, 4.0, size=(8, 3)),
        ]
    )


def flotherm():
    def read(name):
        return sp.csc_matrix(scipy.io.mmread(CASE / "MATRICES" / name))

    return (
        read("K_bci_hat.mtx"),
        read("M_bci_hat.mtx"),
        read("g_bci_hat.mtx").toarray(),
        [read(f"delta_H_bci_hat[{i}].mtx") for i in range(3)],
    )
