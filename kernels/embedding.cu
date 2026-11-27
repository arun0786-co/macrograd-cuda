// an embedding: out[n] = table[ids[n]] for a table (V, D) of V rows and N integer ids,
// like macrograd's nn.embedding. the backward adds each row of dout into the table row
// its id picked:
//
//    dtable[ids[n]] += dout[n]   for every n
//
// an id that appears several times gets the sum of all its rows. that's what makes the
// backward interesting on a GPU: threads that handle the same id write to the same place.
//
// 1. atomic: one thread per value of dout, each doing atomicAdd into the table. simple
//    and fast when the ids are spread out, but the adds to one place happen in whatever
//    order the threads get there, so float rounding can differ from run to run, and many
//    copies of one id queue up on the same addresses
// 2. sorted: sort the (id, n) pairs by id with cub's radix sort, which keeps equal ids in
//    the order of n. find where each id's run starts and ends, then one block per distinct
//    id adds up its rows in that order. the result is the same on every run, and no two
//    threads write to the same place
//
// the forward is a gather, one thread per value, and gives a row of nan for an id outside
// 0..V-1. checked against a CPU version in double and against macrograd on the same numbers
// (tests/embedding_macrograd.txt, from tests/make_embedding_fixture.py).
//
// build and run from the repo root:
//   nvcc -O3 -arch=native -std=c++17 kernels/embedding.cu -o build/embedding && ./build/embedding

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>
#include <cub/cub.cuh>
#include "common.h"

const int THREADS = 256;

__global__ void embedding_forward_kernel(const float *table, const int *ids, float *out, int V, int D, int N) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)N * D) {
        int n = i / D, c = i % D, id = ids[n];
        out[i] = id >= 0 && id < V ? table[(size_t)id * D + c] : NAN;
    }
}

void embedding_forward(const float *table, const int *ids, float *out, int V, int D, int N) {
    size_t n = (size_t)N * D;
    embedding_forward_kernel<<<(unsigned)((n + THREADS - 1) / THREADS), THREADS>>>(table, ids, out, V, D, N);
}

// ---- 1. atomic ----

__global__ void backward_atomic_kernel(const int *ids, const float *dout, float *dtable, int D, int N) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)N * D) {
        int n = i / D, c = i % D;
        atomicAdd(&dtable[(size_t)ids[n] * D + c], dout[i]);
    }
}

void embedding_backward_atomic(const int *ids, const float *dout, float *dtable, int V, int D, int N) {
    size_t n = (size_t)N * D;
    backward_atomic_kernel<<<(unsigned)((n + THREADS - 1) / THREADS), THREADS>>>(ids, dout, dtable, D, N);
}

// ---- 2. sorted ----

__global__ void iota(int *x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] = i;
}

// block b adds up the rows of the b-th distinct id: rows order[start .. start + count),
// in that order, and adds the sum into its table row. threads go along the row
__global__ void backward_runs_kernel(const int *unique_ids, const int *starts, const int *counts, const int *num_runs,
                                     const int *order, const float *dout, float *dtable, int D) {
    if ((int)blockIdx.x >= *num_runs) {
        return;
    }
    int id = unique_ids[blockIdx.x], start = starts[blockIdx.x], count = counts[blockIdx.x];
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        float s = 0.0f;
        for (int k = 0; k < count; k++) {
            s += dout[(size_t)order[start + k] * D + c];
        }
        dtable[(size_t)id * D + c] += s;
    }
}

// scratch for the sort, grown when needed
struct SortScratch {
    int *n_in = nullptr, *sorted_ids = nullptr, *order = nullptr, *unique_ids = nullptr, *counts = nullptr,
        *starts = nullptr, *num_runs = nullptr;
    void *temp = nullptr;
    size_t temp_bytes = 0;
    int capacity = 0;
} g_sort;

void ensure_scratch(int N) {
    if (N <= g_sort.capacity) {
        return;
    }
    int **bufs[] = {&g_sort.n_in, &g_sort.sorted_ids, &g_sort.order, &g_sort.unique_ids, &g_sort.counts, &g_sort.starts};
    for (auto p : bufs) {
        if (*p) CUDA_CHECK(cudaFree(*p));
        CUDA_CHECK(cudaMalloc(p, N * sizeof(int)));
    }
    if (!g_sort.num_runs) CUDA_CHECK(cudaMalloc(&g_sort.num_runs, sizeof(int)));
    // the most temporary storage any of the three cub calls asks for
    size_t a = 0, b = 0, c = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, a, g_sort.n_in, g_sort.sorted_ids, g_sort.n_in, g_sort.order, N);
    cub::DeviceRunLengthEncode::Encode(nullptr, b, g_sort.sorted_ids, g_sort.unique_ids, g_sort.counts,
                                       g_sort.num_runs, N);
    cub::DeviceScan::ExclusiveSum(nullptr, c, g_sort.counts, g_sort.starts, N);
    if (g_sort.temp) CUDA_CHECK(cudaFree(g_sort.temp));
    g_sort.temp_bytes = std::max({a, b, c});
    CUDA_CHECK(cudaMalloc(&g_sort.temp, g_sort.temp_bytes));
    g_sort.capacity = N;
}

void embedding_backward_sorted(const int *ids, const float *dout, float *dtable, int V, int D, int N) {
    ensure_scratch(N);
    SortScratch &s = g_sort;
    size_t bytes = s.temp_bytes;
    iota<<<(N + THREADS - 1) / THREADS, THREADS>>>(s.n_in, N);
    // ids are below V, so only the low bits need sorting
    int bits = 1;
    while ((1 << bits) < V) bits++;
    cub::DeviceRadixSort::SortPairs(s.temp, bytes, ids, s.sorted_ids, s.n_in, s.order, N, 0, bits);
    bytes = s.temp_bytes;
    cub::DeviceRunLengthEncode::Encode(s.temp, bytes, s.sorted_ids, s.unique_ids, s.counts, s.num_runs, N);
    bytes = s.temp_bytes;
    // with fewer runs than N the counts after the last run are left over from before, and
    // their sums are never read
    cub::DeviceScan::ExclusiveSum(s.temp, bytes, s.counts, s.starts, N);
    // one block per possible run; the blocks past the real number of runs return at once
    backward_runs_kernel<<<std::min(N, V), std::min(THREADS, ((D + 31) / 32) * 32)>>>(
        s.unique_ids, s.starts, s.counts, s.num_runs, s.order, dout, dtable, D);
}

typedef void (*Backward)(const int *, const float *, float *, int, int, int);
const int VERSIONS = 2;
const char *NAMES[] = {"atomic", "sorted"};
const Backward BACKWARDS[] = {embedding_backward_atomic, embedding_backward_sorted};

template <typename F>
float time_ms(F launch) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    launch();  // warmup
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < 20; it++) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / 20;
}

// ---- tests ----

__global__ void fill_int(int *x, int value, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = value;
}

// the sorted backward only has as many runs as distinct ids, and the scratch past them
// holds whatever an earlier call left there. the test makes that a run of length 1 at id
// 0, so a block that doesn't stop at the real number of runs adds a row it shouldn't
void plant_stale_runs(int N) {
    ensure_scratch(N);
    int n = g_sort.capacity, b = (n + THREADS - 1) / THREADS;
    fill_int<<<b, THREADS>>>(g_sort.unique_ids, 0, n);
    fill_int<<<b, THREADS>>>(g_sort.counts, 1, n);
    fill_int<<<b, THREADS>>>(g_sort.starts, 0, n);
}

struct Device {
    float *table, *out, *dout, *dtable;
    int *ids;
};

const size_t CAP = (size_t)1 << 24;

// ids uniform in 0..V-1, so a small V repeats them a lot. the table gradient starts at
// random values, so the test sees that it's added to. the forward has to be exact; the
// backward is compared with double, relative to the sum of |values| added into each place.
// the sorted version also has to give the same bits on a second run
bool check_shape(int V, int D, int N, Device d, std::mt19937 &rng, double *worst) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, V - 1);
    std::vector<float> table((size_t)V * D), dout((size_t)N * D), dt0((size_t)V * D);
    std::vector<int> ids(N);
    for (auto &x : table) x = u(rng);
    for (auto &x : dout) x = u(rng);
    for (auto &x : dt0) x = u(rng);
    for (auto &x : ids) x = pick(rng);
    std::vector<double> want(dt0.begin(), dt0.end()), size(dt0.size());
    for (size_t i = 0; i < dt0.size(); i++) size[i] = fabs(dt0[i]);
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < D; c++) {
            want[(size_t)ids[n] * D + c] += dout[(size_t)n * D + c];
            size[(size_t)ids[n] * D + c] += fabs(dout[(size_t)n * D + c]);
        }
    }
    CUDA_CHECK(cudaMemcpy(d.table, table.data(), table.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.ids, ids.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.dout, dout.data(), dout.size() * sizeof(float), cudaMemcpyHostToDevice));
    bool ok = true;

    CUDA_CHECK(cudaMemset(d.out, 0xff, ((size_t)N * D + 4096) * sizeof(float)));
    embedding_forward(d.table, d.ids, d.out, V, D, N);
    std::vector<float> out((size_t)N * D + 4096);
    CUDA_CHECK(cudaMemcpy(out.data(), d.out, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    size_t wrong = 0;
    for (int n = 0; n < N; n++) {
        wrong += memcmp(&out[(size_t)n * D], &table[(size_t)ids[n] * D], D * sizeof(float)) != 0;
    }
    for (size_t i = (size_t)N * D; i < out.size(); i++) wrong += !std::isnan(out[i]);
    if (wrong) {
        printf("  V %d, D %d, N %d, forward: %zu wrong rows\n", V, D, N, wrong);
        ok = false;
    }

    for (int v = 0; v < VERSIONS; v++) {
        std::vector<float> got[2];
        for (int run = 0; run < 2; run++) {
            CUDA_CHECK(cudaMemset(d.dtable, 0xff, ((size_t)V * D + 4096) * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d.dtable, dt0.data(), dt0.size() * sizeof(float), cudaMemcpyHostToDevice));
            plant_stale_runs(N);
            BACKWARDS[v](d.ids, d.dout, d.dtable, V, D, N);
            CUDA_CHECK(cudaGetLastError());
            got[run].resize(dt0.size() + 4096);
            CUDA_CHECK(cudaMemcpy(got[run].data(), d.dtable, got[run].size() * sizeof(float), cudaMemcpyDeviceToHost));
        }
        double e = 0.0;
        for (size_t i = 0; i < want.size(); i++) {
            double x = fabs(got[0][i] - want[i]) / fmax(size[i], 1e-30);
            e = fmax(e, std::isnan(x) ? INFINITY : x);
        }
        bool past = true;
        for (size_t i = want.size(); i < got[0].size(); i++) past = past && std::isnan(got[0][i]);
        bool same = memcmp(got[0].data(), got[1].data(), want.size() * sizeof(float)) == 0;
        *worst = fmax(*worst, e);
        // a sum of up to N / V + 1 terms
        double tol = 1e-6 * sqrt((double)N / V + 2);
        if (!(e < tol) || !past || (v == 1 && !same)) {
            printf("  V %d, D %d, N %d, %s: error %.2e (tol %.2e)%s%s\n", V, D, N, NAMES[v], e, tol,
                   past ? "" : ", wrote past the end", v == 1 && !same ? ", differs between runs" : "");
            ok = false;
        }
    }
    return ok;
}

bool check_macrograd(const char *path, Device d) {
    std::ifstream f(path);
    if (!f) {
        printf("macrograd: %s not found, run from the repo root\n", path);
        return false;
    }
    int V, D, N;
    f >> V >> D >> N;
    std::vector<double> table((size_t)V * D), dout((size_t)N * D), out((size_t)N * D), dtable((size_t)V * D);
    std::vector<int> ids(N);
    for (auto &x : table) f >> x;
    for (auto &x : ids) f >> x;
    for (auto &x : dout) f >> x;
    for (auto &x : out) f >> x;
    for (auto &x : dtable) f >> x;
    if (!f) {
        printf("macrograd: couldn't read %s\n", path);
        return false;
    }
    std::vector<float> tf(table.begin(), table.end()), df(dout.begin(), dout.end());
    CUDA_CHECK(cudaMemcpy(d.table, tf.data(), tf.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.ids, ids.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.dout, df.data(), df.size() * sizeof(float), cudaMemcpyHostToDevice));
    embedding_forward(d.table, d.ids, d.out, V, D, N);
    std::vector<float> got_out(out.size()), got_dt(dtable.size());
    CUDA_CHECK(cudaMemcpy(got_out.data(), d.out, got_out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double w = 0.0;
    for (size_t i = 0; i < out.size(); i++) w = fmax(w, fabs(got_out[i] - out[i]));
    bool ok = true;
    for (int v = 0; v < VERSIONS; v++) {
        CUDA_CHECK(cudaMemset(d.dtable, 0, dtable.size() * sizeof(float)));
        BACKWARDS[v](d.ids, d.dout, d.dtable, V, D, N);
        CUDA_CHECK(cudaMemcpy(got_dt.data(), d.dtable, got_dt.size() * sizeof(float), cudaMemcpyDeviceToHost));
        double wv = w;
        for (size_t i = 0; i < dtable.size(); i++) {
            double e = fabs(got_dt[i] - dtable[i]);
            wv = fmax(wv, std::isnan(e) ? INFINITY : e);
        }
        bool good = wv < 1e-6;
        printf("macrograd (%d x %d, %d ids), %s: largest difference %.2e, %s\n", V, D, N, NAMES[v], wv,
               good ? "ok" : "WRONG");
        ok = ok && good;
    }
    return ok;
}

#ifndef MACROGRAD_NO_MAIN
int main() {
    Device d;
    CUDA_CHECK(cudaMalloc(&d.table, CAP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.out, CAP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.dout, CAP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.dtable, CAP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.ids, (1 << 22) * sizeof(int)));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("gpu: %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    std::mt19937 rng(0);

    bool ok = check_macrograd("tests/embedding_macrograd.txt", d);

    // one id, every id the same, ids spread over a big table, rows of 1 to 1000
    double worst = 0.0;
    bool shapes_ok = true;
    int shapes[][3] = {{1, 1, 1}, {1, 7, 1000}, {5, 3, 1}, {65, 33, 4096}, {1000, 64, 300}, {50257, 128, 2048},
                       {10, 1000, 777}, {300, 1, 5000}};
    for (auto &s : shapes) {
        shapes_ok = check_shape(s[0], s[1], s[2], d, rng, &worst) && shapes_ok;
    }
    printf("shapes: %s, largest backward error %.2e\n", shapes_ok ? "ok" : "WRONG", worst);
    // an id outside the table gives a row of nan
    {
        std::vector<int> bad = {0, 5, -1, 2};
        CUDA_CHECK(cudaMemcpy(d.ids, bad.data(), bad.size() * sizeof(int), cudaMemcpyHostToDevice));
        embedding_forward(d.table, d.ids, d.out, 5, 3, 4);
        std::vector<float> o(12);
        CUDA_CHECK(cudaMemcpy(o.data(), d.out, o.size() * sizeof(float), cudaMemcpyDeviceToHost));
        bool good = true;
        for (int i = 0; i < 12; i++) good = good && (std::isnan(o[i]) == (i / 3 == 1 || i / 3 == 2));
        printf("ids outside the table give nan: %s\n", good ? "ok" : "WRONG");
        ok = ok && good;
    }
    ok = ok && shapes_ok;

    // 65536 ids, rows of 256 (64 MB of dout), the table from 10 rows (every id repeated 6500
    // times) to 50257
    printf("\n%-8s %-12s %-12s %-12s\n", "V", "forward ms", "atomic ms", "sorted ms");
    const int N = 65536, D = 256;
    for (int V : {10, 100, 1000, 10000, 50257}) {
        std::uniform_int_distribution<int> pick(0, V - 1);
        std::vector<int> ids(N);
        for (auto &x : ids) x = pick(rng);
        CUDA_CHECK(cudaMemcpy(d.ids, ids.data(), N * sizeof(int), cudaMemcpyHostToDevice));
        float f = time_ms([&] { embedding_forward(d.table, d.ids, d.out, V, D, N); });
        float a = time_ms([&] { embedding_backward_atomic(d.ids, d.dout, d.dtable, V, D, N); });
        float s = time_ms([&] { embedding_backward_sorted(d.ids, d.dout, d.dtable, V, D, N); });
        printf("%-8d %-12.3f %-12.3f %-12.3f\n", V, f, a, s);
    }

    printf("\n%s\n", ok ? "all ok" : "SOMETHING IS WRONG");
    return ok ? 0 : 1;
}
#endif
