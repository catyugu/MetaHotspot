// Physical model of the chiplet-based 2.5D stack, mirroring the paper.
//
// Provenance tags: PAPER (stated in the ASP-DAC 2026 DTDM paper), REF-28
// (TAP-2.5D, DATE 2021), CODE (TAP-2.5D authors' released input), ASSUMED
// (not recoverable from the paper, chosen explicitly).
//
// Read the tags strictly. Fig. 3 of the paper is a cross-section with layer
// names and no numbers, and the text points at REF-26 / REF-27 for the material
// data, so no thickness or material here is PAPER. What the paper does state is
// the two convection coefficients, the ambient, the 251.6 W total and its split
// over three categories, and the module assignment of every layer.
#pragma once

#include <array>
#include <string>
#include <vector>

namespace mhs::dtdm {

    inline constexpr double kAmbient = 300.0; // PAPER: ambient temperature
    inline constexpr double kHeatSinkH = 400.0; // PAPER: heat-sink convection
    inline constexpr double kPcbH = 10.0; // PAPER: PCB bottom convection

    // Footprints of the packaging stack. REF-28 Sec. III-A states that the edge
    // of the heat spreader is twice the interposer's and the heat sink's twice
    // the spreader's; the paper does not restate it, and Fig. 3 draws the
    // substrate and the PCB wider than the interposer, which is not modelled.
    inline constexpr double kInterposer = 45e-3; // REF-28 Sec. IV, case study 1
    inline constexpr double kSpreader = 90e-3;
    inline constexpr double kSink = 180e-3;

    struct Material {
        const char* name;
        double k; // W/(m K)
        double rho_c; // J/(m3 K)
    };

    // CODE: TAP-2.5D layers.lcf volumetric heat capacities.
    inline constexpr Material kSilicon { "silicon", 100.0, 2330.0 * 751.0 };
    inline constexpr Material kCopper { "copper", 400.0, 8960.0 * 385.0 };
    inline constexpr Material kBump { "bump", 1.6, 1600.0 * 1450.0 };
    inline constexpr Material kTim { "tim", 4.0, 4000.0 * 1000.0 };
    inline constexpr Material kFr4 { "fr4", 0.30, 1250.0 * 2656.0 };
    inline constexpr Material kAluminum { "aluminum", 400.0, 2700.0 * 897.0 };

    // In-plane footprint of a layer. The DoF grouping follows the footprint, not
    // the module: the TIM belongs to CM I but sits on the chip.
    enum Footprint : int { kFootprintSink = 0, kFootprintSpreader = 1, kFootprintChiplets = 2, kFootprintInterposer = 3 };

    // PAPER Sec. III / Fig. 3 lists the layers and their order top to bottom, and
    // Sec. III states the module of each. The thicknesses and materials are CODE
    // (the TAP-2.5D released input) or ASSUMED, not PAPER.
    struct Layer {
        const char* name;
        double thickness;
        Material material;
        int module; // 0 = CM I, 1 = KM, 2 = CM II
        int footprint;
    };

    inline constexpr std::array<Layer, 11> kStack { {
        { "heat_sink", 6.9e-3, kAluminum, 0, kFootprintSink }, // ASSUMED HotSpot default
        { "heat_spreader", 1.0e-3, kCopper, 0, kFootprintSpreader }, // CODE 1 mm, k = 400
        { "tim", 20e-6, kTim, 0, kFootprintChiplets }, // CODE 20 um, k = 4.0
        { "chip", 150e-6, kSilicon, 1, kFootprintChiplets }, // CODE 150 um, k = 100
        { "micro_bump", 10e-6, kBump, 1, kFootprintChiplets }, // CODE 10 um
        { "rdl", 5e-6, kCopper, 1, kFootprintChiplets }, // ASSUMED 5 um
        { "interposer", 110e-6, kSilicon, 2, kFootprintInterposer }, // CODE 110 um
        { "c4_bump", 70e-6, kBump, 2, kFootprintInterposer }, // CODE 70 um
        { "substrate", 200e-6, kFr4, 2, kFootprintInterposer }, // CODE 200 um, k = 0.30
        { "bga", 500e-6, kBump, 2, kFootprintInterposer }, // ASSUMED 500 um
        { "pcb", 1.6e-3, kFr4, 2, kFootprintInterposer }, // ASSUMED 1.6 mm
    } };

    inline constexpr int kModuleCount = 3;
    inline constexpr std::array<const char*, kModuleCount> kModuleNames { "CM_I", "KM", "CM_II" };

    struct Rect {
        double x, y, w, h;
    };

    struct Chiplet {
        const char* name;
        double width, height, power;
        double x, y;
    };

    struct SystemSpec {
        const char* name;
        double interposer;
        std::vector<Chiplet> chiplets;
    };

    // REF-28 Table II chiplet sizes. The paper's Fig. 4 fixes which chiplet sits
    // where and how the HBM parts are turned, but prints no coordinates, so the
    // placements below are ASSUMED reconstructions on a 0.5 mm gap grid inside
    // the interposer frame [-22.5, 22.5] mm and only their arrangement follows
    // the figure.
    // PAPER Fig. 5 states the totals and the per-category shares:
    //   multi-GPU: 251.6 W, GPU 78.1 %, CPU 13.9 %, HBM 8.0 %,
    //   CPU-DRAM : 272 W,   CPU 88.2 %, DRAM 11.8 %.
    // The figures are pie charts, so no per-chiplet wattage is printed; the values
    // below are that total split evenly inside each category (ASSUMED). REF-28
    // Table II gives 105 / 295 / 20 W per CPU / GPU / HBM for the upstream
    // TAP-2.5D configuration, which is not this paper's input.
    inline SystemSpec multi_gpu_compact()
    {
        return { "multi_gpu_compact",
            kInterposer,
            {
                { "cpu", 12.0e-3, 12.0e-3, 35.0, -22.0e-3, 10.0e-3 },
                { "gpu0", 18.2e-3, 18.2e-3, 98.25, -9.5e-3, -22.0e-3 },
                { "gpu1", 18.2e-3, 18.2e-3, 98.25, -9.5e-3, -3.3e-3 },
                { "hbm0", 7.75e-3, 11.87e-3, 6.7, 9.2e-3, -22.0e-3 },
                { "hbm1", 7.75e-3, 11.87e-3, 6.7, 9.2e-3, -9.63e-3 },
                { "hbm2", 7.75e-3, 11.87e-3, 6.7, 9.2e-3, 2.74e-3 },
            } };
    }

    // GPUs on the left edge, CPU top-right, the three HBM chiplets between them.
    inline SystemSpec multi_gpu_thermal_aware()
    {
        return { "multi_gpu_thermal_aware",
            kInterposer,
            {
                { "cpu", 12.0e-3, 12.0e-3, 35.0, 4.95e-3, 10.0e-3 },
                { "gpu0", 18.2e-3, 18.2e-3, 98.25, -22.0e-3, -22.0e-3 },
                { "gpu1", 18.2e-3, 18.2e-3, 98.25, -22.0e-3, 3.3e-3 },
                { "hbm0", 7.75e-3, 11.87e-3, 6.7, -3.3e-3, -22.0e-3 },
                { "hbm1", 7.75e-3, 11.87e-3, 6.7, -3.3e-3, -9.63e-3 },
                { "hbm2", 7.75e-3, 11.87e-3, 6.7, -3.3e-3, 2.74e-3 },
            } };
    }

    // REF-28 Table II "CPU-DRAM System" sizes, PAPER Fig. 8(b) powers: 4 x 60 W
    // CPU + 4 x 8 W DRAM = 272 W. REF-30 gives a 24x36 mm interposer and 0.5 mm
    // chip-to-chip spacing; coordinates ASSUMED to match, in the 45 mm frame.
    inline SystemSpec cpu_dram()
    {
        return { "cpu_dram",
            kInterposer,
            {
                { "cpu0", 8.25e-3, 9.0e-3, 60.0, -8.5e-3, 0.25e-3 },
                { "cpu1", 8.25e-3, 9.0e-3, 60.0, 0.25e-3, 0.25e-3 },
                { "cpu2", 8.25e-3, 9.0e-3, 60.0, -8.5e-3, -9.25e-3 },
                { "cpu3", 8.25e-3, 9.0e-3, 60.0, 0.25e-3, -9.25e-3 },
                { "dram0", 8.75e-3, 8.75e-3, 8.0, -18.5e-3, 0.25e-3 },
                { "dram1", 8.75e-3, 8.75e-3, 8.0, -18.5e-3, -9.0e-3 },
                { "dram2", 8.75e-3, 8.75e-3, 8.0, 9.75e-3, 0.25e-3 },
                { "dram3", 8.75e-3, 8.75e-3, 8.0, 9.75e-3, -9.0e-3 },
            } };
    }

    // In-plane discretisation. One number per level sets the finest cell: the
    // cells across the largest chiplet edge. Every other interval of the axis -
    // chiplet interiors, the gaps between them, the interposer margin, the
    // spreader and the heat sink - follows from it, uniform where the chiplets are
    // and geometrically coarsening outwards. The unknowns therefore sit where the
    // heat is, and the paper's four counts come out of the geometry rather than
    // from padding the periphery. The paper says only that four meshing schemes
    // were used and gives their DoF; how they refine, and with what element, is
    // not published, so this rule is ASSUMED and the DoF match is a scale check,
    // not evidence of an equally accurate mesh.
    struct Layout {
        const char* name;
        int n_fine; // cells across the widest chiplet
        long target; // the paper's Table I DoF count
    };

    // The z distribution is solved for at build time (see mesh.cpp), inside the
    // z counts a stack this thin can justify.
    // Cells across the widest chiplet, chosen so the geometry lands on the paper's
    // count: the four levels miss it by +0.25, +1.14, -0.53 and +0.31 percent.
    inline constexpr std::array<Layout, 4> kLevels { {
        { "Mesh1", 19, 26513L },
        { "Mesh2", 59, 210848L },
        { "Mesh3", 151, 1208923L },
        { "Mesh4", 198, 2037523L },
    } };

} // namespace mhs::dtdm
