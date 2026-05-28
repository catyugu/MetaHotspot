# MetaHotspot Interface Design

## Overview

This document describes the core data structures and module interfaces for the MetaHotspot thermal simulation framework, following the architectural decisions captured in `docs/adr/`.

---

## 1. Namespace Conventions

All code lives in the `mhs` namespace. Each module gets a sub-namespace:

```text
mhs::general   — types, constants, tolerances
mhs::model     — data structures (IO and internal)
mhs::io        — XML serialization/deserialization
mhs::preprocessor  — mesh generation, BC resolution
mhs::assembler     — system assembly
mhs::solver        — linear solver factory
mhs::scheduler     — simulation loop orchestration
mhs::postprocessor — VTU/XML output
mhs::expr          — expression parsing and evaluation
mhs::xmlparser     — XML parsing
mhs::logger        — spdlog wrapper
mhs::utils         — utilities
```

---

## 2. IO Model Structures

IO structures mirror the XML schema directly. They are used only for deserialization/serialization.

### 2.1 Top-Level Structure

```cpp
namespace mhs::model::io {

struct Variable { std::string name; double value; };

struct Rect {
    bool add_sub;
    std::string width_expr;    // geometry expression (string)
    std::string height_expr;
    std::string x_expr;
    std::string y_expr;
    std::string x_size_expr;
    std::string y_size_expr;
    std::string x_interval_expr;
    std::string y_interval_expr;
    std::string name;
};

struct Block {
    std::vector<Rect> all_rects;
    std::string material_name;
    std::string thickness_expr;
    std::string mesh_size_x_expr;
    std::string mesh_size_y_expr;
    std::string mesh_size_z_expr;
    std::string x_offset_expr;
    std::string y_offset_expr;
    std::string z_offset_expr;
    double ti_reyuan = 0.0;
    std::string name;
    bool is_normal_material = true;
};

struct Layer {
    std::vector<Block> blocks;
    std::string name;
    std::string thickness_expr;
    std::string mesh_size_x_expr;
    std::string mesh_size_y_expr;
    std::string mesh_size_z_expr;
    std::string x_offset_expr;
    std::string y_offset_expr;
    std::string period_width_expr;
    int period_width = 10;
    bool is_top_layer = false;
};

enum class BoundaryCategory { Electrical };
enum class ThermalBCType { FirstType, SecondType, ThirdType };

struct FirstTypeThermalBC  { double temperature = 300.0; };
struct SecondTypeThermalBC { double heat_flux = 0.0; };
struct ThirdTypeThermalBC  { double convection_coeff = 0.0; double environment_temp = 300.0; };

struct Boundary {
    BoundaryCategory category;
    std::string name;
    std::vector<std::string> face_keys; // raw face key strings
    ThermalBCType bc_type;
    FirstTypeThermalBC  first;
    SecondTypeThermalBC second;
    ThirdTypeThermalBC  third;
};

struct Material {
    std::string name;
    double daore_xishu = 0.0;       // thermal conductivity k
    double midu = 0.0;              // density rho (optional)
    double bi_rerong = 0.0;         // specific heat c (optional)
};

enum class StudyType { Steady, Transient };
enum class LengthUnit { Mm, Cm, M };
enum class Dimension { Dimension2D, Dimension3D };

struct Structure {
    // Metadata
    std::string software_mode;
    StudyType study_type;
    Dimension dimension;
    LengthUnit length_unit;
    double initial_temperature = 300.0;
    double ambient_temperature = 300.0;
    int die_layer_num = 0;

    // Geometry variables
    std::vector<Variable> variables;

    // Layers and materials
    std::vector<Layer> layers;
    std::unordered_map<std::string, Material> materials;

    // Boundaries
    std::vector<Boundary> boundaries;

    // Transient settings (used if study_type == Transient)
    double transient_duration = 0.0;
    double transient_time_step = 1.0;
    std::string transient_time_unit = "s";

    // Results (for reading reference values from XML)
    std::vector<double> result_values;  // flat array of temperature values
    std::vector<double> result_x;
    std::vector<double> result_y;
    std::vector<double> result_z;
};

} // namespace mhs::model::io
```

### 2.2 Expression Function Types

```cpp
namespace mhs::model::io {

enum class FunctionType { Expression, DoubleExponential, Gauss, Sine, PieceWise };

struct ExpressionFunction {
    std::string expression;  // e.g. "20*(x+1)-exp(x)"
    double draw_min_x = 0.0;
    double draw_max_x = 100.0;
};

struct DoubleExponentialFunction {
    double a = 0.0, alpha = 0.0, beta = 0.0;
    double draw_min_x = 0.0, draw_max_x = 100.0;
};

struct GaussFunction {
    double a = 0.0, tau = 0.0, x0 = 0.0;
    double draw_min_x = 0.0, draw_max_x = 100.0;
};

struct SineFunction {
    double a = 0.0, omega = 0.0, phi = 0.0;
    double draw_min_x = 0.0, draw_max_x = 100.0;
};

struct PieceWiseFunction {
    struct Point { double x = 0.0, y = 0.0; };
    std::vector<Point> points;
    double draw_min_x = 0.0, draw_max_x = 100.0;
};

struct Function {
    std::string key;
    FunctionType type;
    ExpressionFunction expression;
    DoubleExponentialFunction double_exp;
    GaussFunction gauss;
    SineFunction sine;
    PieceWiseFunction piecewise;
};

} // namespace mhs::model::io
```

---

## 3. Internal Model Structures (SoA)

Internal structures are flat SoA arrays, optimized for cache locality and vectorization.

### 3.1 Mesh Geometry

```cpp
namespace mhs::model::internal {

struct MeshGeometry {
    int nx = 0, ny = 0, nz = 0;      // number of cells in each direction
    int cell_count = 0;              // nx * ny * nz

    // Vertex coordinates (one more than cell count per axis)
    std::vector<double> vertex_x;    // size nx+1
    std::vector<double> vertex_y;    // size ny+1
    std::vector<double> vertex_z;    // size nz+1

    // Cell dimensions (for flux calculations)
    std::vector<double> dx;          // size nx, distance between x-vertices
    std::vector<double> dy;          // size ny
    std::vector<double> dz;          // size nz

    // Cell center coordinates (for BC expressions)
    std::vector<double> cx;          // size nx, x-coordinate of each cell center
    std::vector<double> cy;          // size ny
    std::vector<double> cz;          // size nz
};

} // namespace mhs::model::internal
```

### 3.2 Cell Fields (SoA)

```cpp
namespace mhs::model::internal {

enum class MaterialID : uint8_t { Void = 0, Copper = 1, Silicon = 2, TIM = 3 };
enum class LayerID : uint8_t { None = 0, Layer1 = 1, Layer2 = 2, Layer3 = 3 };

// Material property slots — all precompiled into FieldExpression by preprocessor.
// is_constant=true → use constant_value directly (no eval overhead).
// is_constant=false → call expr.eval(ctx) at assembly time.
struct MaterialProps {
    expr::FieldExpression k;   // thermal conductivity k(x,y,z,T,t)
    expr::FieldExpression rho; // density rho(x,y,z,T,t)
    expr::FieldExpression c;   // specific heat c(x,y,z,T,t)
};

struct CellFields {
    int cell_count = 0;

    std::vector<MaterialID> material_id;   // size cell_count
    std::vector<LayerID> layer_id;         // size cell_count

    // BC-applied flags (bitmask for which faces have BCs applied)
    std::vector<uint8_t> bc_flags;         // size cell_count, bitmask
};

} // namespace mhs::model::internal
```

### 3.3 Face BC Arrays (SoA)

```cpp
namespace mhs::model::internal {

enum class BcType : uint8_t { None = 0, FirstType = 1, SecondType = 2, ThirdType = 3 };

// BC parameter table — all precompiled into FieldExpression by preprocessor.
// Each entry is a function: eval(ctx) → value.
// bc_type still determines which parameter to use (e.g. FirstType uses dirichlet_T.eval(ctx)).
struct BCParamTable {
    std::vector<expr::FieldExpression> dirichlet_T;          // size N_dirichlet
    std::vector<expr::FieldExpression> neumann_q;           // size N_neumann
    std::vector<expr::FieldExpression> cauchy_h;            // size N_cauchy
    std::vector<expr::FieldExpression> cauchy_T_inf;         // size N_cauchy
};

struct FaceBCFields {
    // 6 faces, each with N_xy or N_xz or N_yz cells
    // bc_type[i] determines BC type; bc_param_idx[i] indexes into the appropriate
    // BCParamTable vector (dirichlet_T, neumann_q, cauchy_h/cauchy_T_inf).
    // Z- face: size nx * ny
    std::vector<BcType> bc_type_zm;
    std::vector<uint16_t> bc_param_idx_zm;   // index into dirichlet_T / neumann_q / cauchy_*
    // Z+ face: size nx * ny
    std::vector<BcType> bc_type_zp;
    std::vector<uint16_t> bc_param_idx_zp;
    // Y- face: size nx * nz
    std::vector<BcType> bc_type_ym;
    std::vector<uint16_t> bc_param_idx_ym;
    // Y+ face: size nx * nz
    std::vector<BcType> bc_type_yp;
    std::vector<uint16_t> bc_param_idx_yp;
    // X- face: size ny * nz
    std::vector<BcType> bc_type_xm;
    std::vector<uint16_t> bc_param_idx_xm;
    // X+ face: size ny * nz
    std::vector<BcType> bc_type_xp;
    std::vector<uint16_t> bc_param_idx_xp;
};

} // namespace mhs::model::internal
```

### 3.4 Global State Buffer

```cpp
namespace mhs::model::internal {

struct GlobalState {
    int cell_count = 0;
    double current_time = 0.0;
    int time_step = 0;

    // Primary solution vector
    std::vector<double> T;           // temperature, size cell_count

    // For transient: previous time step temperatures
    std::vector<double> T_prev;      // size cell_count

    // For Newton iteration: residual vector
    std::vector<double> residual;   // size cell_count
};

} // namespace mhs::model::internal
```

### 3.5 Fully Assembled Internal Model

```cpp
namespace mhs::model::internal {

struct InternalModel {
    MeshGeometry mesh;

    CellFields cells;

    FaceBCFields face_bcs;
    BCParamTable bc_params;

    // Material properties per material ID
    // MaterialID enum index → MaterialProps (each prop is a precompiled FieldExpression)
    std::array<MaterialProps, 256> material_table;  // indexed by uint8_t MaterialID

    // Simulation metadata
    double initial_temperature = 300.0;
    double ambient_temperature = 300.0;
    StudyType study_type = StudyType::Steady;
    double transient_duration = 0.0;
    double transient_time_step = 1.0;
};

} // namespace mhs::model::internal
```

---

## 4. Module Interfaces

### 4.1 `xmlparser`

```cpp
namespace mhs::xmlparser {

// Parse XML file into a generic DOM tree
class XmlDocument {
public:
    static XmlDocument parse_file(const std::string& path);
    static XmlDocument parse_string(const std::string& xml);

    // Navigate the DOM tree
    class Node {
    public:
        std::string name() const;
        std::string text() const;
        std::string attr(const std::string& name) const;
        std::vector<Node> children() const;
        std::vector<Node> children(const std::string& name);
        Node first_child(const std::string& name) const;
        bool has_child(const std::string& name) const;
    };

    Node root() const;
};

} // namespace mhs::xmlparser
```

### 4.2 `io`

```cpp
namespace mhs::io {

// XML → IO model
class Reader {
public:
    explicit Reader(const std::string& xml_path);

    model::io::Structure read_structure();

private:
    xmlparser::XmlDocument doc_;
};

// IO model → XML (for results)
class Writer {
public:
    explicit Writer(const std::string& output_path);

    void write_result(const model::internal::InternalModel& model,
                      const std::vector<double>& temperature);

    void write_vtu(const model::internal::InternalModel& model,
                   const std::vector<double>& temperature,
                   const std::string& vtu_path);

private:
    std::string output_path_;
};

} // namespace mhs::io
```

### 4.3 `expr`

```cpp
namespace mhs::expr {

// Field expression context
struct FieldContext {
    double x = 0.0, y = 0.0, z = 0.0;  // spatial position
    double T = 0.0;                      // temperature at this location
    double t = 0.0;                      // current simulation time
};

// Compiled field expression (using exprtk)
// All expressions are FieldExpression objects — no raw strings escape preprocessor.
class FieldExpression {
public:
    // Construct a constant expression (no eval needed)
    static FieldExpression make_constant(double value);

    // Construct from string expression like "1.5 + 0.002*T"
    // Registers x, y, z, T, t symbols; looks up function names in expr module's pool
    static FieldExpression from_string(const std::string& expr);

    // Evaluate this expression at the given context
    double eval(const FieldContext& ctx) const;

    bool is_constant() const { return is_constant_; }
    double constant_value() const { return constant_value_; }

private:
    void* exprtk_expr_ = nullptr;  // opaque exprtk handle
    bool is_constant_ = false;
    double constant_value_ = 0.0;
};

// Native function type — takes full context, returns a double.
// Used for piecewise functions, spatialstep functions, etc. that are
// easier to express in C++ than as strings.
using NativeFunc = std::function<double(const FieldContext&)>;

// Register a native C++ function into the global function pool.
// The function receives the full FieldContext and returns a double.
// Example: register_native("my_piecewise", [](const FieldContext& ctx) {
//     if (ctx.x < 1.0) return 0.0;
//     if (ctx.x < 2.0) return 1.0;
//     return 2.0;
// });
void register_native(const std::string& name, NativeFunc func);

// Register a user-defined function into the global function pool
void register_function(const std::string& name,
                       const std::string& expression,
                       const std::vector<std::string>& arg_names);

} // namespace mhs::expr
```

### 4.4 `preprocessor`

```cpp
namespace mhs::preprocessor {

// Convert IO model → internal model
class ModelBuilder {
public:
    explicit ModelBuilder(const model::io::Structure& io_model);

    // Returns the fully assembled internal model
    model::internal::InternalModel build();

private:
    const model::io::Structure& io_model_;
};

// Process layer geometry → fill material_id/layer_id per cell
class LayerProcessor {
public:
    static void resolve_layer_geometry(
        const model::io::Structure& io_model,
        const std::vector<double>& vertex_x,
        const std::vector<double>& vertex_y,
        const std::vector<double>& vertex_z,
        model::internal::CellFields& cells);
};

// Parse face-key strings → per-face BC arrays
class FaceKeyProcessor {
public:
    static void resolve_face_keys(
        const std::vector<model::io::Boundary>& boundaries,
        const model::internal::MeshGeometry& mesh,
        model::internal::FaceBCFields& face_bcs,
        model::internal::BCParamTable& bc_params);
};

} // namespace mhs::preprocessor
```

### 4.5 `assembler`

```cpp
namespace mhs::assembler {

// Assembly context — what the assembler receives each call
struct AssemblyContext {
    const model::internal::InternalModel& model;
    const model::internal::GlobalState& state;

    // Full context for expression evaluation (passed to FieldExpression::eval)
    expr::FieldContext expr_ctx;
};

// Result of assembly: linear system A * T = b
struct LinearSystem {
    Eigen::SparseMatrix<double> A;  // sparse Jacobian
    Eigen::VectorXd b;               // RHS (including BC contributions)
    Eigen::VectorXd residual;       // for convergence check
};

class Assembler {
public:
    explicit Assembler(const model::internal::InternalModel& model);

    // Assemble Jacobian A and RHS b for current state
    // Called each Newton iteration
    LinearSystem assemble(const model::internal::GlobalState& state, double t);

    // Assemble only RHS (for fixed-point iteration or Picard)
    LinearSystem assemble_rhs_only(const model::internal::GlobalState& state, double t);

private:
    const model::internal::InternalModel& model_;
};

} // namespace mhs::assembler
```

### 4.6 `solver`

```cpp
namespace mhs::solver {

enum class SolverType { SparseLU, BiCGSTAB };

struct SolverConfig {
    SolverType type = SolverType::BiCGSTAB;
    double tolerance = 1e-8;
    int max_iterations = 1000;
};

class SolverFactory {
public:
    static std::unique_ptr<SolverBase> create(const SolverConfig& config);
};

class SolverBase {
public:
    virtual ~SolverBase() = default;

    // Solve A * x = b for x
    virtual Eigen::VectorXd solve(const Eigen::SparseMatrix<double>& A,
                                  const Eigen::VectorXd& b) = 0;

    // Compute residual norm after solve
    virtual double residual_norm() const = 0;
};

} // namespace mhs::solver
```

### 4.7 `scheduler`

```cpp
namespace mhs::scheduler {

struct SchedulerConfig {
    double transient_duration = 0.0;
    double time_step = 1.0;
    int max_newton_iterations = 50;
    double newton_tolerance = 1e-6;
    double underrelaxation = 1.0;  // 1.0 = no underrelaxation
    bool is_steady = false;
};

class Scheduler {
public:
    explicit Scheduler(const model::internal::InternalModel& model,
                       const SchedulerConfig& config);

    // Run the full simulation, write results via postprocessor
    // - For steady-state (is_steady=true): single Newton loop at t=0, no time stepping
    // - For transient: time steps from t=0 to transient_duration
    void run(postprocessor::PostProcessor& pp);

    // Step-by-step API (for testing)
    void initialize();
    bool advance_time_step();       // returns true if converged
    bool is_finished() const;

    const model::internal::GlobalState& state() const { return state_; }

private:
    bool solve_nonlinear_step();
    bool check_convergence();

    const model::internal::InternalModel& model_;
    SchedulerConfig config_;
    model::internal::GlobalState state_;
    double current_time_ = 0.0;     // t=0 for steady-state, advances for transient
    int current_step_ = 0;
    std::unique_ptr<assembler::Assembler> assembler_;
    std::unique_ptr<solver::SolverBase> linear_solver_;
};

} // namespace mhs::scheduler
```

### 4.8 `postprocessor`

```cpp
namespace mhs::postprocessor {

class PostProcessor {
public:
    explicit PostProcessor(const std::string& output_dir);

    // Write VTU file for visualization
    void write_vtu(const model::internal::InternalModel& model,
                   const std::vector<double>& temperature,
                   const std::string& filename);

    // Write result XML (compatible with original GUI format)
    void write_xml_result(const model::internal::InternalModel& model,
                          const std::vector<double>& temperature,
                          const std::string& filename);

    // Compute derived quantities
    double max_temperature(const std::vector<double>& T) const;
    double min_temperature(const std::vector<double>& T) const;

private:
    std::string output_dir_;
};

} // namespace mhs::postprocessor
```

---

## 5. Data Flow Summary

```text
XML file
  └─> xmlparser::XmlDocument
        └─> io::Reader
              └─> model::io::Structure (IO model — strings only, mirrors XML schema)
                    └─> preprocessor::ModelBuilder
                          ├─> LayerProcessor::resolve_layer_geometry()
                          │     └─> model::internal::CellFields (SoA)
                          ├─> FaceKeyProcessor::resolve_face_keys()
                          │     └─> model::internal::FaceBCFields + BCParamTable (SoA, strings resolved)
                          ├─> Compile all expressions → expr::FieldExpression
                          │     ├─> MaterialProps.k/rho/c per material
                          │     ├─> BCParamTable entries per boundary
                          │     └─> User-defined function pool (exprtk + native)
                          └─> model::internal::InternalModel (SoA, ALL expressions compiled)
                                └─> scheduler::Scheduler
                                      ├─> assembler::Assembler
                                      │     ├─ eval material props: mat.props.k.eval(ctx)
                                      │     ├─ eval BC params: bc_params.dirichlet_T[idx].eval(ctx)
                                      │     └─> solver::SolverBase
                                      │           └─> Eigen solution
                                      └─> postprocessor::PostProcessor
                                            ├─> VTU file
                                            └─> XML result file
```

---

## 6. Key Design Principles

1. **No raw strings in internal model** — preprocessor compiles ALL expressions (material props, BC params, heat sources) into `expr::FieldExpression` before passing to scheduler/assembler. The internal model contains only evaluable functions, no expression strings.
2. **No virtual functions** — use static polymorphism via templates where needed
3. **No exceptions** — errors logged via `mhs::logger` and program exits with error code
4. **POD types preferred** — all internal model structs are POD-compatible for safety
5. **Pure functions where possible** — `assembler::assemble()` is stateless given model + state
6. **SoA throughout internal model** — all hot-loop arrays are contiguous per-field
7. **Compiled expressions** — exprtk expressions precompiled once, evaluated many times
8. **No shared mutable state** — modules communicate via const references and return values
9. **Native functions for complex forms** — piecewise, tabulated, or spatially complex functions registered via `register_native()` and stored in the expr module's function pool
