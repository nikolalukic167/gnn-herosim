#include "workflow.cpp"

extern "C" double workflow_weighted_start(const double* p, int jobs, int ops, int machines,
                                           double weight, int64_t* a) {
    std::vector<double> ready(jobs, 0.0), free(machines, 0.0), remaining(machines, 0.0);
    std::vector<int> next(jobs, 0);
    for (int i = 0; i < jobs * ops; ++i)
        for (int h = 0; h < machines; ++h)
            if (std::isfinite(p[i*machines+h])) remaining[h] += p[i*machines+h] / 2.0;
    double total = 0.0;
    for (int t = 0; t < jobs * ops; ++t) {
        int j = -1;
        for (int k = 0; k < jobs; ++k)
            if (next[k] < ops && (j < 0 || ready[k] < ready[j])) j = k;
        const int i = j * ops + next[j];
        int chosen = -1;
        double best = std::numeric_limits<double>::infinity();
        for (int h = 0; h < machines; ++h) {
            const double score = std::max(ready[j],free[h]) + p[i*machines+h] + weight * remaining[h];
            if (score < best) { best = score; chosen = h; }
        }
        a[i] = chosen;
        const double finish = std::max(ready[j],free[chosen]) + p[i*machines+chosen];
        free[chosen] = ready[j] = finish;
        for (int h = 0; h < machines; ++h)
            if (std::isfinite(p[i*machines+h])) remaining[h] -= p[i*machines+h] / 2.0;
        if (++next[j] == ops) total += finish;
    }
    return total;
}

extern "C" double workflow_score(const double* p, const int64_t* a, int jobs, int ops, int machines) {
    return replay(p,a,jobs,ops,machines);
}

extern "C" double workflow_refine(const double* p, int64_t* a, int jobs, int ops, int machines,
                                  int rounds, int* used) {
    const int n=jobs*ops;
    std::vector<int64_t> alternative(n);
    std::vector<double> costs(n);
    double score=replay(p,a,jobs,ops,machines);
    *used=0;
    for (int r=0;r<rounds;++r) {
        for (int i=0;i<n;++i)
            for (int h=0;h<machines;++h)
                if (h!=a[i] && std::isfinite(p[i*machines+h])) alternative[i]=h;
        workflow_rank(p,a,alternative.data(),jobs,ops,machines,costs.data());
        ++*used;
        int i=std::min_element(costs.begin(),costs.end())-costs.begin();
        if (costs[i]>=score) break;
        a[i]=alternative[i];score=costs[i];
    }
    return score;
}
