"""Full 3-D TIM/lid FOM attached to the silicon top after ROM extraction."""

import numpy as np
import scipy.sparse as sp
from assembly import COMPONENT_TOP, FIXED_HTC, MM, build_mesh

EXTERNAL_SPECS = (
    ("tim", 0.05 * MM, 4.0, 4.0, 1.80e6),
    ("copper_lid", 0.80 * MM, 390.0, 390.0, 3.45e6),
)


def assemble_cooler(component, cooling_scale=1.0):
    """TIM only over each die; continuous lid bridges the empty cavity.

    The lid retains lateral/through-thickness conduction and heat storage.
    Cavity surfaces are adiabatic. Only the lid top has coolant Robin terms.
    """
    if not np.isfinite(cooling_scale) or cooling_scale <= 0:
        raise ValueError("cooling scale must be finite and positive")
    cooler = build_mesh(
        component.max_cell_size,
        "external_TIM_lid",
        EXTERNAL_SPECS,
        {0},
        "bottom",
        z0=COMPONENT_TOP,
        top_htc=tuple(cooling_scale * h for h in FIXED_HTC),
    )
    if cooler.m != component.m or not np.allclose(
        cooler.points[: cooler.m], component.points[: component.m], rtol=0, atol=1e-14
    ):
        raise ValueError("silicon/TIM interfaces do not match")
    return cooler


def attachment(component_order, cooler):
    """Scatter external states into [component, external interior], sharing ports."""
    ids = np.r_[np.arange(cooler.m), component_order + np.arange(cooler.n)]
    return sp.csc_matrix(
        (np.ones(cooler.N), (np.arange(cooler.N), ids)),
        shape=(cooler.N, component_order + cooler.n),
    )


def connect(component_operator, cooler, component_ports):
    """Conforming assembly: temperature continuity and equal/opposite heat flow.

    Each side contributes its own capacity and conduction. No terminal chain,
    prescribed FOM interface trajectory, penalty/contact resistance or duplicate
    interface state is introduced.
    """
    if component_ports != cooler.m:
        raise ValueError("interface sizes differ")
    n = component_operator.shape[0]
    scatter = attachment(n, cooler)
    return (
        sp.block_diag(
            (component_operator, sp.csc_matrix((cooler.n, cooler.n))), format="csc"
        )
        + scatter.T @ cooler.K @ scatter
    ).tocsc()
