// The chiplet-based 2.5D stack of the HDI paper (ISEDA 2025), Table I.
//
// Everything here is PAPER: Table I gives the layer sizes, the thermal
// conductivity, the heat capacity and the density of every layer; Sec. III gives
// the convection coefficients, the ambient, the region split and the two chip
// layouts (Fig. 4 for chip1, Fig. 5 for chip2). The only things this file has to
// assume are the chiplet coordinates inside the 45 x 35 mm chip outline, which
// Figs. 4 and 5 draw with dimension arrows but no origin, and the gaps between
// neighbouring chiplets.
#pragma once

#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace mhs::hdim {

    inline constexpr double kAmbient = 300.0; // PAPER Sec. III
    inline constexpr double kSinkH = 200.0; // PAPER: heat sink top
    inline constexpr double kPcbH = 10.0; // PAPER: PCB bottom

    /// Region of Sec. II-A: the nonlinear one carries the temperature-dependent
    /// conductivity of the dies, the two linear ones do not.
    enum Region : int { kNonlinear = 0, kLinearTop = 1, kLinearBottom = 2 };
    inline constexpr int kRegionCount = 3;
    inline constexpr std::array<const char*, kRegionCount> kRegionNames {"nonlinear", "linear_top", "linear_bottom"};

    /// Table I. `k_parallel` is in-plane, `k_perpendicular` is through-plane; the
    /// table gives one number when the layer is isotropic.
    struct Layer {
        const char* name;
        double x; // mm
        double y; // mm
        double z; // mm
        double k_parallel;
        double k_perpendicular;
        double cp; // J/(kg K)
        double rho; // kg/m3
        Region region;
        /// Table I writes the die conductivity as 148 (T/300)^-1.3.
        bool temperature_dependent;
    };

    // Top to bottom, Fig. 3. The die conductivity is the only nonlinear one; BEOL
    // sits between the two dies and belongs to the nonlinear region.
    inline constexpr std::array<Layer, 12> kStack {{
        {"heat_sink", 100.0, 100.0, 6.9, 238.0, 238.0, 900.0, 2700.0, kLinearTop, false},
        {"pedestal", 60.0, 60.0, 1.0, 400.0, 400.0, 385.0, 8960.0, kLinearTop, false},
        {"tim", 45.0, 35.0, 0.02, 3.0, 3.0, 1200.0, 2600.0, kLinearTop, false},
        {"chip2", 45.0, 35.0, 0.05, 148.0, 148.0, 700.0, 2329.0, kNonlinear, true},
        {"beol", 45.0, 35.0, 0.015, 2.25, 2.25, 517.0, 4208.0, kNonlinear, false},
        {"chip1", 45.0, 35.0, 0.05, 148.0, 148.0, 700.0, 2329.0, kNonlinear, true},
        {"micro_bumps", 45.0, 35.0, 0.01, 0.9, 2.5, 490.0, 3765.0, kLinearBottom, false},
        {"interposer", 45.0, 35.0, 0.11, 128.0, 148.0, 644.0, 2642.0, kLinearBottom, false},
        {"c4_underfill", 45.0, 35.0, 0.07, 0.66, 25.01, 490.0, 3765.0, kLinearBottom, false},
        {"substrate", 60.0, 60.0, 0.2, 1.4, 1.4, 730.0, 2210.0, kLinearBottom, false},
        {"bga", 60.0, 60.0, 0.94, 0.71, 28.1, 490.0, 3765.0, kLinearBottom, false},
        {"pcb", 100.0, 100.0, 2.0, 0.3, 0.3, 1369.0, 1900.0, kLinearBottom, false},
    }};

    /// The two planes where the nonlinear region meets a linear one: chip2 / TIM
    /// above, chip1 / micro bumps below. They are the interfaces of Sec. II-C.
    inline constexpr int kInterfaceCount = 2;
    inline constexpr std::array<int, kInterfaceCount> kInterfaceLayer {{3, 5}}; // the nonlinear side
    inline constexpr std::array<Region, kInterfaceCount> kInterfaceLinear {{kLinearTop, kLinearBottom}};

    /// Dissipation follows Table I: kappa(T) = 148 (T / 300)^-1.3.
    inline double die_conductivity(double temperature) { return 148.0 * std::pow(temperature / 300.0, -1.3); }

    struct Rect {
        double x0, y0, x1, y1;
    };

    inline bool contains(const Rect& r, double x, double y) { return x > r.x0 && x < r.x1 && y > r.y0 && y < r.y1; }

    /// One chiplet: its footprint in the chip outline and its power.
    struct Chiplet {
        const char* name;
        Rect rect; // mm, origin at the chip outline's lower-left corner
        double power;
    };

    struct ChipSpec {
        const char* name;
        double total_power;
        std::vector<Chiplet> chiplets;
    };

    /// Fig. 4: multi-GPU chip, 125.8 W, GPU 78.1 % / CPU 13.9 % / HBM 8.0 %.
    inline ChipSpec multi_gpu_chip()
    {
        return {"multi_gpu", 125.8,
            {
                {"gpu0", {0.6, 16.2, 18.8, 34.4}, 49.125},
                {"gpu1", {26.2, 0.6, 44.4, 18.8}, 49.125},
                {"hbm0", {20.2, 22.5, 27.95, 34.4}, 3.353},
                {"hbm1", {28.8, 22.5, 36.55, 34.4}, 3.353},
                {"hbm2", {0.6, 0.6, 8.35, 12.47}, 3.353},
                {"cpu", {12.6, 0.6, 24.6, 12.6}, 17.49},
            }};
    }

    /// Fig. 5: AI training chip, 58.3 W, AI core 73.2 % / HBM 22.9 % / IO 3.9 %.
    inline ChipSpec ai_chip()
    {
        return {"ai_training", 58.3,
            {
                {"ai_core", {0.6, 10.1, 32.0, 24.6}, 42.68},
                {"io", {33.0, 9.6, 43.5, 25.6}, 2.27},
                {"hbm0", {0.6, 25.7, 12.47, 34.4}, 3.338},
                {"hbm1", {13.5, 25.7, 25.37, 34.4}, 3.338},
                {"hbm2", {0.6, 0.6, 8.35, 9.0}, 3.338},
                {"hbm3", {9.4, 0.6, 17.15, 9.0}, 3.338},
            }};
    }

    /// The chip outline the two dies share (Table I: 45 x 35 mm).
    inline constexpr double kChipX = 45.0;
    inline constexpr double kChipY = 35.0;

    /// Meshing levels. One number is enough: the in-plane cell over the chip
    /// outline, everything else follows from it. Fig. 9 reports five DoF counts
    /// and no mesh rule, so each level's in-plane size was calibrated by bisection
    /// (`--probe`) to reproduce the reported unknowns; the cells through a die and
    /// through a thin layer are refined alongside, as the L-shaped mesh of Fig. 6.
    struct Level {
        const char* name;
        double inplane; // mm, cell size over the chip outline
        int z_die; // cells through each die
        int z_thin; // cells through the thin layers (TIM, BEOL, bumps, ...)
        long target; // Fig. 9
    };

    inline constexpr std::array<Level, 5> kLevels {{
        {"Mesh1", 0.5983, 1, 1, 65760L},
        {"Mesh2", 0.2839, 2, 1, 304640L},
        {"Mesh3", 0.2154, 2, 2, 912640L},
        {"Mesh4", 0.1560, 3, 2, 1815040L},
        {"Mesh5", 0.1397, 4, 3, 3296640L},
    }};

} // namespace mhs::hdim
