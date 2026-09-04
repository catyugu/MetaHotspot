"""Model-agnostic macromodel (MOR) algorithms for MetaHotspot.

This subpackage contains only model-independent reduced-order thermal-model
algorithms and plain-data contracts.  It never names a concrete case geometry,
material, layer stack, or model configuration — concrete models and case
workflows live in the playground/ adapters that consume these algorithms.

Modules:

* :mod:`utils` — FANTASTIC–BCI operator-level machinery: per-port spectral
  bounds, elliptic shift planning, residual-driven Krylov enrichment,
  ``build_parametric_basis``, ``project_bci``, reduced solves, accuracy
  metrics.  Purely numeric; consumes the generic ``Operators`` interface.
* :mod:`affine` — the affine-parametric model base contract
  (:class:`~metahotspot.macromodel.affine.AffineParametricModel`), its
  plain-data satellites (:class:`~metahotspot.macromodel.affine.BoundaryGroup`,
  :class:`~metahotspot.macromodel.affine.SourcePort`,
  :class:`~metahotspot.macromodel.affine.CellLayout`,
  :class:`~metahotspot.macromodel.affine.AffineSolveResult`), the
  :class:`~metahotspot.macromodel.geometry.CellGeometry` geometry view,
  and the model registry/factory (:func:`~metahotspot.macromodel.affine.create`,
  :func:`~metahotspot.macromodel.affine.register`).  Concrete models register
  from playground adapters; the library itself registers nothing.
* :mod:`embeddable` — the embeddable ROM extractor: boundary-face port
  enumeration (only explicitly declared ambient faces are excluded; every other
  boundary face becomes a connectable :class:`~metahotspot.macromodel.embeddable.FacePort`),
  subdomain assembly and whole-subdomain extraction
  (:class:`~metahotspot.macromodel.embeddable.EmbeddableRom`),
  non-conforming common-patch area weighting, and
  independent-interface-node coupling
  (:func:`~metahotspot.macromodel.embeddable.connect`).

All algorithms consume plain numpy/scipy data or the ``metahotspot`` bindings'
generic ``Operators`` / ``CellFields`` contracts — never a named case.
"""

from metahotspot.macromodel import utils
from metahotspot.macromodel import affine
from metahotspot.macromodel import embeddable

# Re-export the stable public surface.
from metahotspot.macromodel.affine import (
    AffineParametricModel,
    AffineSolveResult,
    BoundaryGroup,
    CellLayout,
    SourcePort,
    create,
    register,
    registered_names,
)
from metahotspot.macromodel.geometry import BoundarySurface, CellGeometry
from metahotspot.macromodel.embeddable import (
    EmbeddableRom,
    FacePort,
    Subdomain,

    build_subdomain,
    common_patches,
    connect,
    enumerate_interface_ports,
    extract_rom,

    interface_trace,
    side_junction_rise,
    solve_system,
)

__all__ = [
    # modules
    "utils",
    "affine",
    "embeddable",
    # affine contract
    "AffineParametricModel",
    "AffineSolveResult",
    "BoundaryGroup",
    "CellLayout",
    "SourcePort",
    "create",
    "register",
    "registered_names",
    "BoundarySurface",
    "CellGeometry",
    # embeddable ROM
    "EmbeddableRom",
    "FacePort",
    "Subdomain",

    "build_subdomain",
    "common_patches",
    "connect",
    "enumerate_interface_ports",
    "extract_rom",

    "interface_trace",
    "side_junction_rise",
    "solve_system",
]
