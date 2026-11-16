# Data contract

- Question: end-to-end extraction speed from one AMG hierarchy per shared frequency, reused across HTC samples and heat-source ports.
- Route: reused-public benchmark defined by the user's repository, not a substituted mesh.
- Source: https://github.com/catyugu/MetaHotspot/tree/deb981297d1353d9005e0a042657e1d8eef0e595/playground/bci_rom_testcase1
- Geometry: `model_case1.py`, unmodified `Case1Config(max_cell_mm=0.5)`, 146704 solid cells, four source ports, three HTC groups.
- Operators: assembled by the matching Linux CI wheel, artifact 14121435121, workflow 43612616388, same revision as Python sources. No stored matrix from an older experiment is reused.
- Licence: repository MIT. The code-generated layered heat-sink benchmark is a computational model, not a physical measurement dataset.
- Training: physical HTC box [1,10000]^3 mapped using `physical_to_effective`; adaptive log-uniform draws in the effective coefficient box, exactly as the current extraction code.
- Validation: all eight physical HTC corners, nominal (100,1000,30), and four independent log-uniform points (seed 74021). These points do not change extraction or the cache policy.
- Transients: nominal point and both uniform HTC extremes, four 5 W sources stepped from zero, BDF1 dt=20 s through 1000 s, identical discretization for FOM and ROM.
- Accuracy reference: independently solved affine FOM, tolerance 1e-10; field and junction errors are measured relative to this same discrete model.
- Preprocessing: none beyond native assembly and documented physical-to-effective boundary conversion. No mesh scaling, port aggregation, or source-space modification.
- Reuse boundary: AMG may be reused only at exactly identical float-valued frequency; never across frequencies. The cache has a single slot, released and overwritten when advancing frequency; all source ports finish the current frequency first. All cached hierarchies are discarded between complete extractions.
- Provenance: real CPU execution; timing of the numerical experiment is measured. No numerical placeholder is permitted in reports or plots.
