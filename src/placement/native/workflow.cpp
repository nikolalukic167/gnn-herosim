#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

// Linear job selection preserves the Python heap's (release time, job index) order.
static double replay(const double* p, const int64_t* a, int jobs, int ops, int machines) {
    std::vector<double> ready(jobs, 0.0), free(machines, 0.0);
    std::vector<int> next(jobs, 0);
    double total = 0.0;
    for (int t = 0; t < jobs * ops; ++t) {
        int j = -1;
        for (int k = 0; k < jobs; ++k)
            if (next[k] < ops && (j < 0 || ready[k] < ready[j])) j = k;
        const int position = j * ops + next[j];
        const int h = static_cast<int>(a[position]);
        const double end = std::max(ready[j], free[h]) + p[position * machines + h];
        free[h] = end;
        ready[j] = end;
        if (++next[j] == ops) total += end;
    }
    return total;
}

extern "C" double workflow_ect(const double* p, int jobs, int ops, int machines, int64_t* a) {
    std::vector<double> ready(jobs, 0.0), free(machines, 0.0);
    std::vector<int> next(jobs, 0);
    double total = 0.0;
    for (int t = 0; t < jobs * ops; ++t) {
        int j = -1;
        for (int k = 0; k < jobs; ++k)
            if (next[k] < ops && (j < 0 || ready[k] < ready[j])) j = k;
        const int position = j * ops + next[j];
        int chosen = -1;
        double end = std::numeric_limits<double>::infinity();
        for (int h = 0; h < machines; ++h) {
            const double finish = std::max(ready[j], free[h]) + p[position * machines + h];
            if (finish < end) { end = finish; chosen = h; }
        }
        a[position] = chosen;
        free[chosen] = end;
        ready[j] = end;
        if (++next[j] == ops) total += end;
    }
    return total;
}

extern "C" void workflow_rank(const double* p, const int64_t* a, const int64_t* alternatives,
                              int jobs, int ops, int machines, double* costs) {
    std::vector<int64_t> candidate(a, a + jobs * ops);
    for (int i = 0; i < jobs * ops; ++i) {
        candidate[i] = alternatives[i];
        costs[i] = replay(p, candidate.data(), jobs, ops, machines);
        candidate[i] = a[i];
    }
}
