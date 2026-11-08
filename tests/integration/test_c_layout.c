#include "metahotspot_c_api.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(expr)                                                                                                    \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            fprintf(stderr, "%s:%d: %s (%s)\n", __FILE__, __LINE__, #expr, mhs_last_error());                          \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)
#define OK(expr) CHECK((expr) == MHS_OK)

static int volume(mhs_model_t* m, mhs_index_t material, mhs_index_t source, const double lo[3], const double hi[3])
{
    mhs_index_t id;
    OK(mhs_model_add_volume(m, 0, material, source, &id));
    OK(mhs_model_volume_add_box(m, 0, id, MHS_GEOM_ADD, lo, hi));
    return 0;
}

/* GL-09 multi-domain path, compiled as C with no internal C++ access. */
static int connection_client(void)
{
    mhs_model_t* m = NULL;
    mhs_compiled_t* c = NULL;
    mhs_solution_t* solution = NULL;
    mhs_index_t upper, material, volume_id, a, b, connection;
    const double xy[] = {0, 1}, lower_z[] = {0, 0.5}, upper_z[] = {0.5, 1};
    const double lo[] = {0, 0, 0}, hi[] = {1, 1, 1};
    const mhs_surface_region_t region = {MHS_AXIS_Z, 0.5, {0, 1, 0, 1}};
    OK(mhs_model_create(&m));
    OK(mhs_model_add_fom_domain(m, &upper));
    CHECK(upper == 1);
    OK(mhs_model_add_material(m, "solid", "1", "1", "1", "1", "1", NULL, &material));
    for (mhs_index_t d = 0; d < 2; ++d) {
        OK(mhs_model_set_cartesian_mesh(m, d, 2, xy, 2, xy, 2, d ? upper_z : lower_z));
        OK(mhs_model_add_volume(m, d, material, MHS_INVALID_INDEX, &volume_id));
        OK(mhs_model_volume_add_box(m, d, volume_id, MHS_GEOM_ADD, lo, hi));
        OK(mhs_model_set_default_dirichlet(m, d, "300+10*z"));
    }
    OK(mhs_model_add_port(m, 0, "top", &region, &a));
    OK(mhs_model_add_port(m, upper, "bottom", &region, &b));
    CHECK(mhs_model_compile(m, &c) == MHS_ERROR && c == NULL);
    OK(mhs_model_connect(m, 0, a, upper, b, &connection));
    OK(mhs_model_compile(m, &c));
    mhs_model_destroy(m);
    mhs_compiled_info_t info;
    OK(mhs_compiled_get_info(c, &info));
    CHECK(info.domain_count == 2 && info.field_count == 2 && info.state_count == 2 && info.cell_count == 2);
    OK(mhs_compiled_solve(c, NULL, NULL, 0, NULL, &solution));
    double state[2], flux[1];
    OK(mhs_solution_copy_state(solution, state, 2));
    CHECK(fabs(state[0] - 302.5) < 1e-6 && fabs(state[1] - 307.5) < 1e-6);
    size_t count;
    OK(mhs_compiled_connection_patch_count(c, connection, &count));
    CHECK(count == 1);
    OK(mhs_compiled_eval_connection_flux(c, connection, state, 2, 0, flux, 1));
    CHECK(fabs(flux[0] + 10) < 1e-6);
    CHECK(mhs_compiled_eval_connection_flux(c, connection, state, 1, 0, flux, 1) == MHS_ERROR);
    CHECK(mhs_compiled_connection_patch_count(c, 99, &count) == MHS_ERROR);
    mhs_solution_destroy(solution);
    mhs_compiled_destroy(c);
    return 0;
}

/* GL-03/GL-09: staggered pressure/face-velocity state over the public C ABI. */
static int stokes_client(void)
{
    mhs_model_t* m = NULL;
    mhs_compiled_t* c = NULL;
    mhs_solution_t* solution = NULL;
    mhs_index_t material, volume_id;
    const double x[] = {0, 1, 2}, yz[] = {0, 1};
    const double lo[] = {0, 0, 0}, hi[] = {2, 1, 1};
    const double temperature = 300;
    const mhs_surface_region_t inlet = {MHS_AXIS_X, 0, {0, 1, 0, 1}};
    const mhs_surface_region_t outlet = {MHS_AXIS_X, 2, {0, 1, 0, 1}};
    OK(mhs_model_create(&m));
    OK(mhs_model_set_cartesian_mesh(m, 0, 3, x, 2, yz, 2, yz));
    OK(mhs_model_add_material(m, "fluid", "1", "1", "1", "1", "1", "T/300", &material));
    OK(mhs_model_add_volume(m, 0, material, MHS_INVALID_INDEX, &volume_id));
    OK(mhs_model_volume_add_box(m, 0, volume_id, MHS_GEOM_ADD, lo, hi));
    OK(mhs_model_set_flow_mode(m, 0, MHS_FLOW_STOKES));
    OK(mhs_model_add_fluid_boundary(m, 0, &inlet, 1, MHS_FLUID_PRESSURE_INLET, 1, &temperature));
    OK(mhs_model_add_fluid_boundary(m, 0, &outlet, 1, MHS_FLUID_OUTLET, 0, &temperature));
    OK(mhs_model_compile(m, &c));
    mhs_model_destroy(m);
    mhs_compiled_info_t info;
    OK(mhs_compiled_get_info(c, &info));
    CHECK(info.state_count == 7 && info.field_count == 5 && info.cell_count == 2);
    mhs_field_info_t velocity;
    OK(mhs_compiled_get_field_info(c, 2, &velocity));
    CHECK(velocity.quantity == MHS_FIELD_VELOCITY);
    CHECK(velocity.support == MHS_SUPPORT_FACE);
    CHECK(velocity.dof_count == 3);
    OK(mhs_compiled_solve(c, NULL, NULL, 0, NULL, &solution));
    mhs_solution_info_t result;
    OK(mhs_solution_get_info(solution, &result));
    CHECK(result.converged && result.state_count == 7);
    double state[7];
    OK(mhs_solution_copy_state(solution, state, 7));
    for (size_t i = 0; i < 7; ++i)
        CHECK(isfinite(state[i]));
    CHECK(state[4] > 0 && state[5] > 0 && state[6] > 0);
    mhs_solution_destroy(solution);
    mhs_compiled_destroy(c);
    return 0;
}

/* GL-01 A and GL-09: a pure C client, without private C++ access. */
int main(void)
{
    CHECK(connection_client() == 0);
    CHECK(stokes_client() == 0);
    mhs_model_t* m = NULL;
    mhs_compiled_t* c = NULL;
    mhs_solution_t *steady = NULL, *transient = NULL;
    mhs_index_t silicon, tim, copper, liquid, source;
    double x[13], y[9], z[11];
    for (size_t i = 0; i < 13; ++i)
        x[i] = (double)i * 0.001;
    for (size_t i = 0; i < 9; ++i)
        y[i] = (double)i * 0.001;
    for (size_t i = 0; i < 11; ++i)
        z[i] = (double)i * 0.0005;
    OK(mhs_model_create(&m));
    OK(mhs_model_set_cartesian_mesh(m, 0, 13, x, 9, y, 11, z));
    OK(mhs_model_add_material(m, "silicon", "130", "130", "130", "2330", "700", NULL, &silicon));
    OK(mhs_model_add_material(m, "TIM", "4", "4", "4", "2500", "1000", NULL, &tim));
    OK(mhs_model_add_material(m, "copper", "390", "390", "390", "8960", "385", NULL, &copper));
    OK(mhs_model_add_material(m, "liquid", "0.6", "0.6", "0.6", "997", "4180", "8.9e-4", &liquid));
    OK(mhs_model_add_source(m, "1.25e8", &source));
    CHECK(volume(m, silicon, MHS_INVALID_INDEX, (double[]) {0, 0, 0}, (double[]) {0.012, 0.008, 0.001}) == 0);
    CHECK(volume(m, tim, MHS_INVALID_INDEX, (double[]) {0, 0, 0.001}, (double[]) {0.012, 0.008, 0.0015}) == 0);
    CHECK(volume(m, copper, MHS_INVALID_INDEX, (double[]) {0, 0, 0.0015}, (double[]) {0.012, 0.008, 0.005}) == 0);
    CHECK(volume(m, liquid, MHS_INVALID_INDEX, (double[]) {0, 0.003, 0.0025}, (double[]) {0.012, 0.005, 0.0035}) == 0);
    CHECK(volume(m, silicon, source, (double[]) {0.004, 0.002, 0}, (double[]) {0.008, 0.006, 0.001}) == 0);
    const mhs_surface_region_t top = {MHS_AXIS_Z, 0.005, {0, 0.012, 0, 0.008}};
    const mhs_surface_region_t inlet = {MHS_AXIS_X, 0, {0.003, 0.005, 0.0025, 0.0035}};
    const mhs_surface_region_t outlet = {MHS_AXIS_X, 0.012, {0.003, 0.005, 0.0025, 0.0035}};
    const double temperature = 300;
    OK(mhs_model_add_convection(m, 0, &top, 1, "400", "300"));
    OK(mhs_model_add_fluid_boundary(m, 0, &inlet, 1, MHS_FLUID_MASS_FLOW_INLET, 2e-4, &temperature));
    OK(mhs_model_add_fluid_boundary(m, 0, &outlet, 1, MHS_FLUID_OUTLET, 0, NULL));
    OK(mhs_model_add_point_probe(m, "hotspot", (double[]) {0.006, 0.004, 0.00025}));
    OK(mhs_model_add_point_probe(m, "upstream", (double[]) {0.0015, 0.0035, 0.00275}));
    OK(mhs_model_add_point_probe(m, "downstream", (double[]) {0.0105, 0.0035, 0.00275}));
    OK(mhs_model_add_point_probe(m, "copper", (double[]) {0.006, 0.004, 0.00475}));
    OK(mhs_model_compile(m, &c));
    mhs_model_destroy(m);
    mhs_compiled_info_t info;
    mhs_field_info_t field;
    mhs_domain_info_t domain;
    OK(mhs_compiled_get_info(c, &info));
    CHECK(info.cell_count == 960 && info.state_count == 960 && info.domain_count == 1 && info.field_count == 1);
    OK(mhs_compiled_get_domain_info(c, 0, &domain));
    CHECK(domain.cell_count == 960 && domain.field_count == 1);
    OK(mhs_compiled_get_field_info(c, 0, &field));
    CHECK(field.domain == 0 && field.quantity == MHS_FIELD_TEMPERATURE && field.support == MHS_SUPPORT_CELL
        && field.dof_count == 960);
    mhs_index_t entities[960], dofs[960];
    OK(mhs_compiled_copy_field_mapping(c, 0, entities, dofs, 960));
    for (size_t i = 0; i < 960; ++i)
        CHECK(entities[i] == i && dofs[i] == i);
    CHECK(mhs_compiled_get_field_info(c, 99, &field) == MHS_ERROR);
    CHECK(mhs_compiled_copy_field_mapping(c, 0, NULL, NULL, 960) == MHS_ERROR);
    OK(mhs_compiled_solve(c, NULL, NULL, 0, NULL, &steady));
    mhs_study_t study;
    mhs_study_default(&study);
    study.type = MHS_STUDY_TRANSIENT;
    study.duration = 2;
    study.output_interval = 0.1;
    OK(mhs_compiled_solve(c, &study, NULL, 0, NULL, &transient));
    mhs_compiled_destroy(c);
    mhs_solution_info_t result;
    OK(mhs_solution_get_info(transient, &result));
    CHECK(result.converged && result.state_count == 960 && result.record_count == 21 && result.probe_count == 4);
    double state[960];
    OK(mhs_solution_copy_state(steady, state, 960));
    double peak = 300;
    for (size_t i = 0; i < 960; ++i) {
        CHECK(isfinite(state[i]));
        if (state[i] > peak)
            peak = state[i];
    }
    CHECK(peak > 300);
    mhs_solution_destroy(steady);
    mhs_solution_destroy(transient);
    return 0;
}
