#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <chrono>


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

void A(int M, int N, double hx, double hy, std::vector<double> &ans, const std::vector<double> &u, const std::vector<double> &a, const std::vector<double> &b) {
    ans.clear();
    ans.resize(u.size());
    #pragma omp parallel for
    for (int i = 2; i < M-1; ++i) { // inside
        for (int j = 2; j < N-1; ++j) {
            double au_xx = 1.0 / hx / hx * (a[(i+1)*(N+1) + j] * (u[(i+1)*(N+1) + j] - u[i*(N+1) + j]) - a[i*(N+1) + j] * (u[i*(N+1) + j] - u[(i-1)*(N+1) + j]));
            double bu_yy = 1.0 / hy / hy * (b[i*(N+1) + j+1] * (u[i*(N+1) + j+1] - u[i*(N+1) + j]) - b[i*(N+1) + j] * (u[i*(N+1) + j] - u[i*(N+1) + j-1]));
            ans[i*(N+1) + j] = -au_xx - bu_yy;
        }
    }
    for (int i = 1; i < M; ++i) { // boundary
        for (int j = 1; j < N; j += ((i == 1 || i == M-1) ? 1 : N-2)) {
            double au_xx = 1.0 / hx / hx * (a[(i+1)*(N+1) + j] * (u[(i+1)*(N+1) + j] - u[i*(N+1) + j]) - a[i*(N+1) + j] * (u[i*(N+1) + j] - u[(i-1)*(N+1) + j]));
            double bu_yy = 1.0 / hy / hy * (b[i*(N+1) + j+1] * (u[i*(N+1) + j+1] - u[i*(N+1) + j]) - b[i*(N+1) + j] * (u[i*(N+1) + j] - u[i*(N+1) + j-1]));
            ans[i*(N+1) + j] = -au_xx - bu_yy;
        }
    }
}

void add(double xs, std::vector<double> &x, double ys, const std::vector<double> &y) { // x = xs * x + ys * y
    //#pragma omp parallel for
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = xs * x[i] + ys * y[i];
    }
}

double dot(const std::vector<double> &x, const std::vector<double> &y) {
    double ans = 0;
    //#pragma omp parallel for reduction(+:ans)
    for (size_t i = 0; i < x.size(); ++i) {
        ans += x[i] * y[i];
    }
    return ans;
}

double step(int M, int N, double hx, double hy, std::vector<double> &u, const std::vector<double> &a, const std::vector<double> &b, const std::vector<double> &F) {
    std::vector<double> res(u.size()), Au(u.size());
    A(M, N, hx, hy, Au, u, a, b);
    res = F;
    add(-1, res, 1, Au);
    double rr = dot(res, res);
    A(M, N, hx, hy, Au, res, a, b); // Au = A(r)
    double Arr = dot(Au, res);
    add(1, u, -rr / Arr, res);
    return std::sqrt(rr * hx * hy);
}

double CG(int M, int N, double hx, double hy, std::vector<double> &u, const std::vector<double> &a, const std::vector<double> &b, const std::vector<double> &F, int &nits) {
    std::vector<double> r(u.size()), z(u.size()), az(u.size());
    A(M, N, hx, hy, r, u, a, b); add(-1, r, 1, F); // r = F - A u
    z = r;
    for (int it = 0; it < nits; ++it) {
        A(M, N, hx, hy, az, z, a, b);
        double alp = dot(r, r) / dot(az, z);
        add(1, u, alp, z);
        double nn = std::sqrt(dot(z, z) * hx * hy * alp * alp);
        if (nn < 1e-10 || it == nits-1) {
            nits = it + 1;
            return nn;
        }
        double rr = dot(r, r);
        add(1, r, -alp, az);
        double bet = dot(r, r) / rr;
        add(bet, z, 1, r);
    }
    return 1e300;
}

int main() {
    double st = get_time();

    int M = 40, N = 40, use_CG, its = M*N;
    double eps = 0.001;
    std::cin >> M >> N >> use_CG >> its;
    int n = (M + 1) * (N + 1);
    std::vector<double> a(n), b(n), u(n), F(n);

    double xmin = -3, xmax = 3, ymin = 0, ymax = 4;
    double hx = (xmax - xmin) / M, hy = (ymax - ymin) / N;
    // x = hx * i, y = hy * j

    { // construct a
        //#pragma omp parallel for
        for (int i = 1; i <= M; ++i) {
            for (int j = 1; j <= N; ++j) {
                double x = (i - 0.5) * hx + xmin;
                double y1 = (j - 0.5) * hy + ymin;
                double y2 = (j + 0.5) * hy + ymin;
                bool i1 = inside(x, y1), i2 = inside(x, y2);
                if (i1 == i2) {
                    a[i*(N+1) + j] = (i1 ? 1 : 1/eps);
                }
                else {
                    if (i1) {
                        std::swap(y1, y2);
                        std::swap(i1, i2);
                    }
                    double q = find_boundary(x, y1, x, y2);
                    a[i*(N+1) + j] = q/eps + 1-q;
                }
            }
        }
    }
    { // construct b
        //#pragma omp parallel for
        for (int i = 1; i <= M; ++i) {
            for (int j = 1; j <= N; ++j) {
                double x1 = (i - 0.5) * hx + xmin;
                double x2 = (i + 0.5) * hx + xmin;
                double y = (j - 0.5) * hy + ymin;
                bool i1 = inside(x1, y), i2 = inside(x2, y);
                if (i1 == i2) {
                    b[i*(N+1) + j] = (i1 ? 1 : 1/eps);
                }
                else {
                    if (i1) {
                        std::swap(x1, x2);
                        std::swap(i1, i2);
                    }
                    double q = find_boundary(x1, y, x2, y);
                    b[i*(N+1) + j] = q/eps + 1-q;
                }
            }
        }
    }
    { // construct F
        //#pragma omp parallel for
        for (int i = 1; i < M; ++i) {
            for (int j = 1; j < N; ++j) {
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
                    F[i*(N+1) + j] = f(i*hx + xmin, j*hy + ymin);
                }
                else {
                    // calc S
                    double S = calc_inside_part(x1, y1, x2, y2);
                    F[i*(N+1) + j] = S * vv;
                }
            }
        }
    }
    if (use_CG) {
        CG(M, N, hx, hy, u, a, b, F, its);
    }
    else {
        its = 0;
        while (step(M, N, hx, hy, u, a, b, F) > 1e-3) ++its;
    }
    std::cerr << "|dw|=" << step(M, N, hx, hy, u, a, b, F) << " its=" << its << std::endl;
    for (int i = 0; i <= M; ++i) {
        for (int j = 0; j <= N; ++j) {
            std::cout << i << "\t" << j << "\t" << u[i*(N+1) + j] << "\n";
        }
    }
    std::cerr << "final time: " << get_time() - st << std::endl;
}
