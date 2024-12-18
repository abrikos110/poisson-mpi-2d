#include <mpi.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <chrono>
#include <string>

#include "mpihw.cpp"


double get_time() {
    auto c = std::chrono::high_resolution_clock();
    auto now = c.now().time_since_epoch();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now);
    return ns.count() / 1e9;
}

bool inside(double x, double y) {
    if (0 <= y && y <= 4) {
        if (x < 0) x = -x;
        return x/3 + y/4 <= 1;
    }
    return false;
}

// i'm lazy
double find_boundary(double xout, double yout, double xin, double yin) {
    bool i1 = inside(xout, yout);
    double l = 0, r = 1, q = (l + r) / 2;
    while (l < q && q < r) {
        if (i1 != inside((1-q)*xout + q*xin, (1-q)*yout + q*yin)) {
            r = q;
        }
        else {
            l = q;
        }
        q = (l + r) / 2;
    }
    return q;
}

double dist(double x1, double y1, double x2, double y2) {
    if (x1 == x2) return std::abs(y1 - y2);
    if (y1 == y2) return std::abs(x1 - x2);
    return std::sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

double f(double x, double y) {
    return 1.0 * inside(x, y);
}

double calc_inside_part(double x1, double y1, double x2, double y2) {
    double dx = x2 - x1, dy = y2 - y1;
    double xc = (x1 + x2) / 2, yc = (y1 + y2) / 2;
    double ans = 0, eps = 1e-2;
    for (int i = -3; i <= 3; i += 2) {
        for (int j = -1; j <= 1; j += 2) {
            double x = xc + i * dx / 8;
            double y = yc + j * dy / 4;
            ans += (inside(x+eps*dx, y) + 0.0 + inside(x-eps*dx, y)) / 2;
        }
    }
    return ans / 8;
}

void A(double hx, double hy, dist_grid_2d &dgrid, std::pair<int, int> x_range, std::pair<int, int> y_range, std::vector<double> &ans, std::vector<double> &u, const std::vector<double> &a, const std::vector<double> &b) {
    ans.clear();
    ans.resize(u.size());
    int ylen = y_range.second - y_range.first + 2;
    int ib = x_range.first-1, jb = y_range.first-1;
    #pragma omp parallel for
    for (int i = x_range.first+1; i < x_range.second-1; ++i) { // inside
        for (int j = y_range.first+1; j < y_range.second-1; ++j) {
            i -= ib;
            j -= jb;
            double au_xx = 1.0 / hx / hx * (a[(i+1)*ylen + j] * (u[(i+1)*ylen + j] - u[i*ylen + j]) - a[i*ylen + j] * (u[i*ylen + j] - u[(i-1)*ylen + j]));
            double bu_yy = 1.0 / hy / hy * (b[i*ylen + j+1] * (u[i*ylen + j+1] - u[i*ylen + j]) - b[i*ylen + j] * (u[i*ylen + j] - u[i*ylen + j-1]));
            ans[i*ylen + j] = -au_xx - bu_yy;
            i += ib;
            j += jb;
        }
    }
    // TODO UPDATE HERE
    dgrid.set_send(u);
    dgrid.update_sendrecv();
    dgrid.update_wait();
    dgrid.get_recv(u);
    for (int i = x_range.first; i < x_range.second; ++i) { // boundary
        for (int j = y_range.first; j < y_range.second; j = ((i == x_range.first || i == x_range.second-1 || j == y_range.second-1) ? j+1 : y_range.second-1)) {
            i -= ib;
            j -= jb;
            double au_xx = 1.0 / hx / hx * (a[(i+1)*ylen + j] * (u[(i+1)*ylen + j] - u[i*ylen + j]) - a[i*ylen + j] * (u[i*ylen + j] - u[(i-1)*ylen + j]));
            double bu_yy = 1.0 / hy / hy * (b[i*ylen + j+1] * (u[i*ylen + j+1] - u[i*ylen + j]) - b[i*ylen + j] * (u[i*ylen + j] - u[i*ylen + j-1]));
            ans[i*ylen + j] = -au_xx - bu_yy;
            i += ib;
            j += jb;
        }
    }
}

void add(double xs, std::vector<double> &x, double ys, const std::vector<double> &y) { // x = xs * x + ys * y
    //#pragma omp parallel for
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = xs * x[i] + ys * y[i];
    }
}

double dot_glob(const std::vector<double> &x, const std::vector<double> &y, std::pair<int, int> x_range, std::pair<int, int> y_range) {
    double ans = 0;
    int ib = x_range.first-1, jb = y_range.first-1, ylen = y_range.second - y_range.first + 2;
    for (int i = x_range.first; i < x_range.second; ++i) { // no domain boundary since zero BC
        for (int j = y_range.first; j < y_range.second; ++j) {
            ans += x[(i-ib)*ylen + j-jb] * y[(i-ib)*ylen + j-jb];
        }
    }
    double gans = 0;
    MPI_Allreduce(&ans, &gans, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return gans;
}

double step(double hx, double hy, dist_grid_2d &dgrid, std::pair<int, int> x_range, std::pair<int, int> y_range, std::vector<double> &u, const std::vector<double> &a, const std::vector<double> &b, const std::vector<double> &F) {
    std::vector<double> res(u.size()), Au(u.size());
    A(hx, hy, dgrid, x_range, y_range, Au, u, a, b);
    res = F;
    add(-1, res, 1, Au);
    double rr = dot_glob(res, res, x_range, y_range);
    A(hx, hy, dgrid, x_range, y_range, Au, res, a, b); // Au = A(r)
    double Arr = dot_glob(Au, res, x_range, y_range);
    add(1, u, -rr / Arr, res);
    return std::sqrt(rr * hx * hy);
}

double CG(double hx, double hy, dist_grid_2d &dgrid, std::pair<int, int> x_range, std::pair<int, int> y_range, std::vector<double> &u, const std::vector<double> &a, const std::vector<double> &b, const std::vector<double> &F, int &nits) {
    std::vector<double> r(u.size()), z(u.size()), az(u.size());
    A(hx, hy, dgrid, x_range, y_range, r, u, a, b); add(-1, r, 1, F); // r = F - A u
    z = r;
    for (int it = 0; it < nits; ++it) {
        A(hx, hy, dgrid, x_range, y_range, az, z, a, b);
        double alp = dot_glob(r, r, x_range, y_range) / dot_glob(az, z, x_range, y_range);
        add(1, u, alp, z);
        double nn = std::sqrt(dot_glob(z, z, x_range, y_range) * hx * hy) * std::abs(alp);
        if (nn < 1e-10 || it == nits-1) {
            nits = it + 1;
            return nn;
        }
        double rr = dot_glob(r, r, x_range, y_range);
        add(1, r, -alp, az);
        double bet = dot_glob(r, r, x_range, y_range) / rr;
        add(bet, z, 1, r);
    }
    return 1e300;
}

int main(int argc, char **args) {
    MPI_Init(&argc, &args);
    int world_size, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    std::cerr << "MPI rank " << my_rank << "/" << world_size << std::endl;


    double st = get_time();

    int M, N, use_CG;
    double eps = 0.001;
    if (argc < 4) {
        std::cerr << "Usage: echo M N use_CG xb xe yb ye | " << args[0] << std::endl;
        return 0;
    }
    M = std::stoi(args[1]);
    N = std::stoi(args[2]);
    use_CG = std::stoi(args[3]);
    dist_grid_2d dgrid(my_rank, world_size, { M+1, N+1 });
    std::pair<int, int> x_range = dgrid.x_range(), y_range = dgrid.y_range();

    int xlen = x_range.second - x_range.first + 2;
    int ylen = y_range.second - y_range.first + 2;

    int n = xlen * ylen;
    std::vector<double> a(n), b(n), u(n), F(n);

    double xmin = -3, xmax = 3, ymin = 0, ymax = 4;
    double hx = (xmax - xmin) / M, hy = (ymax - ymin) / N;
    // x = hx * i, y = hy * j
    int ib = x_range.first-1, jb = y_range.first-1;

    { // construct a
        //#pragma omp parallel for
        for (int i = x_range.first-1; i < x_range.second+1; ++i) {
            for (int j = y_range.first-1; j < y_range.second+1; ++j) {
                double x = (i - 0.5) * hx + xmin;
                double y1 = (j - 0.5) * hy + ymin;
                double y2 = (j + 0.5) * hy + ymin;
                bool i1 = inside(x, y1), i2 = inside(x, y2);
                if (i1 == i2) {
                    a[(i-ib)*ylen + (j-jb)] = (i1 ? 1 : 1/eps);
                }
                else {
                    if (i1) {
                        std::swap(y1, y2);
                        std::swap(i1, i2);
                    }
                    double q = find_boundary(x, y1, x, y2);
                    a[(i-ib)*ylen + (j-jb)] = q/eps + 1-q;
                }
            }
        }
    }
    { // construct b
        //#pragma omp parallel for
        for (int i = x_range.first-1; i < x_range.second+1; ++i) {
            for (int j = y_range.first-1; j < y_range.second+1; ++j) {
                double x1 = (i - 0.5) * hx + xmin;
                double x2 = (i + 0.5) * hx + xmin;
                double y = (j - 0.5) * hy + ymin;
                bool i1 = inside(x1, y), i2 = inside(x2, y);
                if (i1 == i2) {
                    b[(i-ib)*ylen + (j-jb)] = (i1 ? 1 : 1/eps);
                }
                else {
                    if (i1) {
                        std::swap(x1, x2);
                        std::swap(i1, i2);
                    }
                    double q = find_boundary(x1, y, x2, y);
                    b[(i-ib)*ylen + (j-jb)] = q/eps + 1-q;
                }
            }
        }
    }
    { // construct F
        //#pragma omp parallel for
        for (int i = x_range.first; i < x_range.second; ++i) {
            for (int j = y_range.first; j < y_range.second; ++j) {
                double x1 = (i - 0.5) * hx + xmin;
                double x2 = (i + 0.5) * hx + xmin;
                double y1 = (j - 0.5) * hy + ymin;
                double y2 = (j + 0.5) * hy + ymin;
                double vv = -100000;
                bool i11 = inside(x1, y1);
                if (i11) vv = f(x1, y1);
                bool i12 = inside(x1, y2);
                if (i12) vv = f(x1, y2);
                bool i21 = inside(x2, y1);
                if (i21) vv = f(x2, y1);
                bool i22 = inside(x2, y2);
                if (i22) vv = f(x2, y2);
                if (i11 == i12 && i12 == i21 && i21 == i22) {
                    F[(i-ib)*ylen + (j-jb)] = f(i*hx + xmin, j*hy + ymin);
                }
                else {
                    // calc S
                    double S = calc_inside_part(x1, y1, x2, y2);
                    F[(i-ib)*ylen + (j-jb)] = S * vv;
                }
            }
        }
    }
    std::cerr << "starting iters" << x_range.first << "," << x_range.second << " " << y_range.first << "," << y_range.second << std::endl;
    int its = 0;
    if (use_CG) {
        its = M*N;
        if (argc >= 5) its = std::stoi(args[4]);
        CG(hx, hy, dgrid, x_range, y_range, u, a, b, F, its);
    }
    else {
        its = 0;
        while (step(hx, hy, dgrid, x_range, y_range, u, a, b, F) > 1e-3) ++its;
    }
    std::cerr << "|dw|=" << step(hx, hy, dgrid, x_range, y_range, u, a, b, F) << " its=" << its << std::endl;
    //auto file = std::fopen(("out" + std::to_string(my_rank) + ".csv").data(), "w");
    auto file = stdout;
    for (int I = 0; I < world_size; ++I) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (I == my_rank) {
            if (x_range.first == 1) --x_range.first;
            if (x_range.second == M) ++x_range.second;
            if (y_range.first == 1) --y_range.first;
            if (y_range.second == N) ++y_range.second;
            for (int i = x_range.first; i < x_range.second; ++i) {
                for (int j = y_range.first; j < y_range.second; ++j) {
                    std::fprintf(file, "%d\t%d\t%g\n", i, j, u[(i-ib)*ylen + (j-jb)]);
                }
            }
        }
    }
    std::fclose(file);
    std::cerr << "final time: " << get_time() - st << std::endl;

    MPI_Finalize();
}
