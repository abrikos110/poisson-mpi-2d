#include <mpi.h>
#include <utility>
#include <cstdarg>
#include <cstdio>
#include <vector>

void assert_mpi(int condition, const char *format, ...) {
    if (!condition) {
        std::va_list args;
        va_start(args, format);

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        std::fprintf(stderr, "Assertion failed on rank %d: ", rank);
        std::vfprintf(stderr, format, args);
        std::fprintf(stderr, "\n");

        va_end(args);

        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

int sqrt_div(int x) {
    int mdiv = 1;
    for (int i = 2; i * i <= x; ++i) {
        if (x % i == 0) mdiv = i;
    }
    return mdiv; // biggest divider <= sqrt(x)
}

struct dist_grid_1d {
    int pn; // process number
    int gs; // grid size
    int piece, rem;
    dist_grid_1d(int pn, int gs)
        : pn(pn), gs(gs) {
        piece = gs / pn;
        rem = gs % piece;
    }
    int split(int pi) {
        return pi * piece + (pi < rem ? pi : rem);
    }
    int inv_split(int i) {
        return i / piece;
    }
};

struct dist_grid_2d {
    std::pair<int, int> pn, pi;
    std::pair<int, int> gs;
    dist_grid_1d x, y;
    int rank;

    std::vector< std::pair<int, int> > neighbours;
    std::vector<int> neig_len_cumsum;
    std::vector<double> send, recv;
    std::vector<MPI_Request> sreq, rreq;

    dist_grid_2d(int rank, int world_size, std::pair<int, int> gs)
        : pn(world_size / sqrt_div(world_size), sqrt_div(world_size)),
          pi(rank / pn.second, rank % pn.second),
          gs(gs),
          x(pn.first, gs.first),
          y(pn.second, gs.second),
          rank(rank) {
        set_neig_vector();
    }
    std::pair<int, int> x_range() {
        int a = x.split(pi.first);
        int b = x.split(pi.first + 1);
        if (a == 0) ++a; // zero BC
        if (b == gs.first) --b; // zero BC
        return { a, b };
    }
    std::pair<int, int> y_range() {
        int a = y.split(pi.second);
        int b = y.split(pi.second + 1);
        if (a == 0) ++a; // zero BC
        if (b == gs.second) --b; // zero BC
        return { a, b };
    }
    static int slen(std::pair<int, int> x) {
        return x.second - x.first;
    }
    static void cumsum(std::vector<int> &x) {
        for (size_t i = 1; i < x.size(); ++i) {
            x[i] += x[i-1];
        }
    }
    void set_neig_vector() {
        neighbours.clear();
        neig_len_cumsum.clear();
        neig_len_cumsum.push_back(0);
        int y_len = slen(y_range()), x_len = slen(x_range());

        for (int i = -1; i <= 1; i += 2) {
            std::pair<int, int> cand = { pi.first + i, pi.second };
            if (0 <= cand.first && cand.first < x.pn) {
                neighbours.push_back(cand);
                neig_len_cumsum.push_back(y_len);
            }
            cand = { pi.first, i + pi.second };
            if (0 <= cand.second && cand.second < y.pn) {
                neighbours.push_back(cand);
                neig_len_cumsum.push_back(x_len);
            }
        }
        cumsum(neig_len_cumsum);
        send.resize(neig_len_cumsum.back());
        recv.resize(neig_len_cumsum.back());
    }
    void set_send(const std::vector<double> &u) {
        int ib = x_range().first - 1, jb = y_range().first - 1;
        int ylen = slen(y_range()) + 2;
        for (size_t neig = 0; neig < neighbours.size(); ++neig) {
            double *ptr = &send[neig_len_cumsum[neig]];
            int isx = neighbours[neig].first != pi.first;
            int len = (isx ? slen(y_range()) : slen(x_range()));
            assert_mpi(len == neig_len_cumsum[neig+1] - neig_len_cumsum[neig], "bad len");
            int fi = (neighbours[neig].first  <= pi.first  ? x_range().first : x_range().second-1);
            int fj = (neighbours[neig].second <= pi.second ? y_range().first : y_range().second-1);
            for (int K = 0; K < len; ++K) {
                if (isx) ptr[K] = u[(fi-ib)*ylen + K+fj-jb];
                else     ptr[K] = u[(K+fi-ib)*ylen + fj-jb];
            }
        }
    }
    void get_recv(std::vector<double> &u) {
        int ib = x_range().first - 1, jb = y_range().first - 1;
        int ylen = slen(y_range()) + 2;
        for (size_t neig = 0; neig < neighbours.size(); ++neig) {
            double *ptr = &recv[neig_len_cumsum[neig]];
            int isx = neighbours[neig].first != pi.first;
            int len = (isx ? slen(y_range()) : slen(x_range()));
            assert_mpi(len == neig_len_cumsum[neig+1] - neig_len_cumsum[neig], "bad len");
            int fi = (neighbours[neig].first  <= pi.first  ? x_range().first - isx : x_range().second-1 + isx);
            int fj = (neighbours[neig].second <= pi.second ? y_range().first - !isx: y_range().second-1 + !isx);
            for (int K = 0; K < len; ++K) {
                if (isx) u[(fi-ib)*ylen + K+fj-jb] = ptr[K];
                else     u[(K+fi-ib)*ylen + fj-jb] = ptr[K];
            }
        }
    }
    void update_sendrecv() {
        sreq.resize(neighbours.size());
        rreq.resize(neighbours.size());
        for (size_t neig = 0; neig < neighbours.size(); ++neig) {
            int dest = neighbours[neig].first * pn.second + neighbours[neig].second;
            int beg = neig_len_cumsum[neig], end = neig_len_cumsum[neig+1];
            MPI_Isend(&send[beg], end - beg, MPI_DOUBLE, dest, 7 + rank, MPI_COMM_WORLD, &sreq[neig]);
            MPI_Irecv(&recv[beg], end - beg, MPI_DOUBLE, dest, 7 + dest, MPI_COMM_WORLD, &rreq[neig]);
        }
    }
    void update_wait() {
        assert_mpi(neighbours.size() == sreq.size() && sreq.size() == rreq.size(), "bad sendrecv size");
        for (size_t neig = 0; neig < neighbours.size(); ++neig) {
            assert_mpi(MPI_SUCCESS == MPI_Wait(&sreq[neig], MPI_STATUS_IGNORE), "bad send");
            assert_mpi(MPI_SUCCESS == MPI_Wait(&rreq[neig], MPI_STATUS_IGNORE), "bad recv");
        }
        sreq.clear();
        rreq.clear();
    }
};

#if 0
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int M = 40, N = 40;

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    dist_grid_2d dgrid(rank, world_size, {M+1, N+1});

    std::pair<int, int> xrange, yrange;
    xrange = dgrid.x_range();
    yrange = dgrid.y_range();

  for(int prank = 0; prank < world_size; ++prank) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (prank != rank) continue;

    std::printf("Hello, world! rank=%d %d-%d,%d-%d / %d,%d\n", rank, xrange.first, xrange.second, yrange.first, yrange.second, dgrid.pn.first, dgrid.pn.second);
    for (size_t i = 0; i < dgrid.neighbours.size(); ++i) {
        std::printf("pi=%d,%d neig=%d,%d len=%d\n", dgrid.pi.first, dgrid.pi.second, dgrid.neighbours[i].first, dgrid.neighbours[i].second, dgrid.neig_len_cumsum[i+1] - dgrid.neig_len_cumsum[i]);
    }
    std::printf("\n");
  }

    MPI_Finalize();

    return 0;
}
#endif
