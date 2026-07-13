#include "common/mesh_utils.hpp"
#include "postprocessor/sample_point.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <limits>

namespace mhs::post {

    double sample_solve_least_squares(
        const std::vector<SampleDataPoint>& pts, double node_x, double node_y, double node_z)
    {
        int M = static_cast<int>(pts.size());
        if (M == 0)
            return std::numeric_limits<double>::quiet_NaN();
        if (M == 1)
            return pts[0].T;

        Eigen::MatrixXd A(M + 3, 4);
        Eigen::VectorXd B(M + 3);
        A.setZero();
        B.setZero();

        double sum_w = 0.0;
        for (int i = 0; i < M; ++i) {
            double sqrt_w = std::sqrt(pts[i].weight);
            A(i, 0) = sqrt_w;
            A(i, 1) = sqrt_w * (pts[i].x - node_x);
            A(i, 2) = sqrt_w * (pts[i].y - node_y);
            A(i, 3) = sqrt_w * (pts[i].z - node_z);
            B(i) = sqrt_w * pts[i].T;
            sum_w += pts[i].weight;
        }

        // Tikhonov regularization:
        double reg_w = std::sqrt(sum_w * 1e-6);
        A(M, 1) = reg_w;
        A(M + 1, 2) = reg_w;
        A(M + 2, 3) = reg_w;

        Eigen::Vector4d X = A.colPivHouseholderQr().solve(B);
        return X(0);
    }

    void sample_face_center(mhs::core::FaceDir dir, int ix, int iy, int iz, const mhs::core::MeshGeometry& mesh,
        double& fx, double& fy, double& fz)
    {
        mhs::utils::face_center_3d(dir, ix, iy, iz, mesh, fx, fy, fz);
    }

    double sample_extrapolate_face_temperature(mhs::core::FaceDir dir, mhs::core::BcType bc_type, uint16_t param_idx,
        double T_c, double k, const mhs::core::MeshGeometry& mesh, int ix, int iy, int iz,
        const mhs::core::BCParamTable& bc_params, double time)
    {
        double fx, fy, fz;
        sample_face_center(dir, ix, iy, iz, mesh, fx, fy, fz);
        mhs::core::FieldContext ctx {fx, fy, fz, T_c, time};

        double half_dist = mhs::utils::half_length_along(dir, mesh.dx[ix], mesh.dy[iy], mesh.dz[iz]);

        if (bc_type == mhs::core::BcType::SecondType) {
            double q = bc_params.neumann_q[param_idx].eval(ctx);
            return T_c + (q * half_dist) / k;
        }
        else if (bc_type == mhs::core::BcType::ThirdType) {
            double h = bc_params.cauchy_h[param_idx].eval(ctx);
            double T_inf = bc_params.cauchy_T_inf[param_idx].eval(ctx);
            double cond_h = k / half_dist;
            return (h * T_inf + cond_h * T_c) / (h + cond_h);
        }
        return T_c;
    }

} // namespace mhs::post
