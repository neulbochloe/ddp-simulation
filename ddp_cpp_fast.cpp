#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

constexpr double PI =
    3.141592653589793238462643383279502884;

const double LOG_2PI =
    std::log(2.0 * PI);


// ============================================================
// C++ FAST2 GIBBS KERNEL
//
// This implements the same Gibbs transition as
// ddp_stage2c_fast2.py.
//
// Python remains responsible for:
//   - data generation
//   - posterior reconstruction
//   - KS/CDF diagnostics
//   - UQ
//
// C++ handles:
//   - Gibbs allocation updates
// ============================================================


py::dict run_gibbs_cpp(
    py::array_t<double,
        py::array::c_style |
        py::array::forcecast> x_array,

    py::array_t<double,
        py::array::c_style |
        py::array::forcecast> y_array,

    int n_sweeps,
    int burn_in,
    int thin,
    double M,
    unsigned long long seed
) {

    // --------------------------------------------------------
    // INPUT
    // --------------------------------------------------------

    auto xbuf = x_array.request();
    auto ybuf = y_array.request();

    if (xbuf.ndim != 1 || ybuf.ndim != 1) {
        throw std::runtime_error(
            "x and y must be one-dimensional."
        );
    }

    if (xbuf.size != ybuf.size) {
        throw std::runtime_error(
            "x and y must have the same length."
        );
    }

    const int n =
        static_cast<int>(xbuf.size);

    if (n <= 0) {
        throw std::runtime_error(
            "n must be positive."
        );
    }

    if (n_sweeps <= 0) {
        throw std::runtime_error(
            "n_sweeps must be positive."
        );
    }

    if (burn_in < 0 || burn_in >= n_sweeps) {
        throw std::runtime_error(
            "burn_in must satisfy "
            "0 <= burn_in < n_sweeps."
        );
    }

    if (thin <= 0) {
        throw std::runtime_error(
            "thin must be positive."
        );
    }

    if (M <= 0.0) {
        throw std::runtime_error(
            "M must be positive."
        );
    }


    const double* x =
        static_cast<double*>(xbuf.ptr);

    const double* y =
        static_cast<double*>(ybuf.ptr);


    // --------------------------------------------------------
    // PRECOMPUTE d_i = 4 + x_i^2
    // --------------------------------------------------------

    std::vector<double> d(n);

    for (int i = 0; i < n; ++i) {
        d[i] = 4.0 + x[i] * x[i];
    }

    // Cache all conditional log densities. These depend only on x,y,
    // not on the Gibbs allocation state. O(n^2) memory (8 MB at n=1000).
    std::vector<double> pair_logpdf(static_cast<size_t>(n) * n);
    std::vector<double> new_logw(n);
    for (int i = 0; i < n; ++i) {
        new_logw[i] = std::log(M) - 0.5 *
            (LOG_2PI + std::log(d[i]) + y[i] * y[i] / d[i]);
        for (int k = 0; k < n; ++k) {
            const double cov = 4.0 + x[i] * x[k];
            const double mu = (cov / d[k]) * y[k];
            const double var = d[i] - cov * cov / d[k];
            if (var <= 0.0) {
                pair_logpdf[static_cast<size_t>(i) * n + k] =
                    -std::numeric_limits<double>::infinity();
            } else {
                const double diff = y[i] - mu;
                pair_logpdf[static_cast<size_t>(i) * n + k] =
                    -0.5 * (LOG_2PI + std::log(var) + diff * diff / var);
            }
        }
    }



    // --------------------------------------------------------
    // INITIAL STATE
    //
    // Same initialization as Python:
    // every observation starts in its own cluster.
    // --------------------------------------------------------

    std::vector<long long> labels(n);

    // cluster_members[label]
    //
    // Because cluster size can only be 1 or 2,
    // we store members in a vector.
    std::unordered_map<
        long long,
        std::vector<int>
    > clusters;

    // singleton_member[label] = observation index
    std::unordered_map<
        long long,
        int
    > singleton_member;


    for (int i = 0; i < n; ++i) {

        labels[i] = i;

        clusters[i] = {i};

        singleton_member[i] = i;
    }

    long long next_label = n;


    // --------------------------------------------------------
    // RNG
    // --------------------------------------------------------

    std::mt19937_64 rng(seed);

    std::uniform_real_distribution<double>
        uniform01(0.0, 1.0);


    // --------------------------------------------------------
    // OUTPUT STORAGE
    //
    // Store labels for saved posterior states.
    //
    // Python will reconstruct cluster dictionaries from these.
    // --------------------------------------------------------

    std::vector<
        std::vector<long long>
    > saved_labels;

    std::vector<int> K_trace;


    // permutation workspace
    std::vector<int> permutation(n);

    std::iota(
        permutation.begin(),
        permutation.end(),
        0
    );


    // ========================================================
    // MCMC
    // ========================================================

    for (
        int iteration = 0;
        iteration < n_sweeps;
        ++iteration
    ) {

        // Random scan order
        std::shuffle(
            permutation.begin(),
            permutation.end(),
            rng
        );


        // ====================================================
        // ONE GIBBS SWEEP
        // ====================================================

        for (int position = 0; position < n; ++position) {

            const int i =
                permutation[position];


            // ------------------------------------------------
            // REMOVE OBSERVATION i
            // ------------------------------------------------

            const long long old_label =
                labels[i];

            auto cluster_it =
                clusters.find(old_label);

            if (cluster_it == clusters.end()) {
                throw std::runtime_error(
                    "Old cluster not found."
                );
            }

            std::vector<int>& members =
                cluster_it->second;


            if (members.size() == 1) {

                // singleton -> empty

                singleton_member.erase(
                    old_label
                );

                clusters.erase(
                    old_label
                );

            } else if (members.size() == 2) {

                // pair -> singleton

                int remaining;

                if (members[0] == i) {
                    remaining = members[1];
                } else {
                    remaining = members[0];
                }

                // Replace pair by singleton
                cluster_it->second.clear();
                cluster_it->second.push_back(
                    remaining
                );

                singleton_member[
                    old_label
                ] = remaining;

            } else {

                throw std::runtime_error(
                    "Cluster size exceeded 2."
                );
            }

            labels[i] = -1;


            // ------------------------------------------------
            // SINGLETON CANDIDATES
            // ------------------------------------------------

            const int s =
                static_cast<int>(
                    singleton_member.size()
                );

            std::vector<long long>
                singleton_labels;

            std::vector<int>
                singleton_members;

            singleton_labels.reserve(s);
            singleton_members.reserve(s);


            for (
                const auto& item :
                singleton_member
            ) {

                singleton_labels.push_back(
                    item.first
                );

                singleton_members.push_back(
                    item.second
                );
            }


            // ------------------------------------------------
            // LOG WEIGHTS
            // ------------------------------------------------

            std::vector<double>
                logw(s + 1);


            const double x_i = x[i];
            const double y_i = y[i];
            const double d_i = d[i];


            // Existing singleton candidates
            for (int j = 0; j < s; ++j) {

                const int k =
                    singleton_members[j];

                logw[j] = pair_logpdf[static_cast<size_t>(i) * n + k];
            }


            // ------------------------------------------------
            // NEW CLUSTER
            // ------------------------------------------------

            logw[s] = new_logw[i];


            // ------------------------------------------------
            // DIRECT CATEGORICAL SAMPLING
            //
            // Same mathematical operation as FAST2:
            //
            //   max_logw
            //   exp(logw - max)
            //   U * total
            //   first cumulative weight > threshold
            // ------------------------------------------------

            const double max_logw =
                *std::max_element(
                    logw.begin(),
                    logw.end()
                );


            std::vector<double>
                weights(s + 1);

            double total = 0.0;


            for (int j = 0; j <= s; ++j) {

                weights[j] =
                    std::exp(
                        logw[j] - max_logw
                    );

                total += weights[j];
            }


            const double threshold =
                uniform01(rng) * total;


            double cumulative = 0.0;

            int chosen = s;


            for (int j = 0; j <= s; ++j) {

                cumulative += weights[j];

                if (cumulative > threshold) {

                    chosen = j;
                    break;
                }
            }


            // ------------------------------------------------
            // CHOOSE LABEL
            // ------------------------------------------------

            long long chosen_label;


            if (chosen < s) {

                chosen_label =
                    singleton_labels[chosen];

            } else {

                chosen_label =
                    next_label;

                ++next_label;
            }


            // ------------------------------------------------
            // ADD OBSERVATION
            // ------------------------------------------------

            auto chosen_it =
                clusters.find(
                    chosen_label
                );


            if (chosen_it != clusters.end()) {

                // Existing candidate must be singleton.

                if (
                    chosen_it->second.size()
                    != 1
                ) {

                    throw std::runtime_error(
                        "Attempted to add "
                        "observation to "
                        "non-singleton cluster."
                    );
                }

                chosen_it->second.push_back(i);

                // singleton -> pair
                singleton_member.erase(
                    chosen_label
                );

            } else {

                // new cluster -> singleton

                clusters[
                    chosen_label
                ] = {i};

                singleton_member[
                    chosen_label
                ] = i;
            }


            labels[i] =
                chosen_label;
        }


        // ====================================================
        // POST-BURN-IN STORAGE
        // ====================================================

        if (iteration >= burn_in) {

            K_trace.push_back(
                static_cast<int>(
                    clusters.size()
                )
            );


            const int post_index =
                iteration - burn_in;


            if (
                post_index % thin == 0
            ) {

                saved_labels.push_back(
                    labels
                );
            }
        }
    }


    // ========================================================
    // RETURN TO PYTHON
    // ========================================================

    py::array_t<long long>
        labels_out(
            {
                static_cast<py::ssize_t>(
                    saved_labels.size()
                ),
                static_cast<py::ssize_t>(n)
            }
        );


    auto labels_view =
        labels_out.mutable_unchecked<2>();


    for (
        py::ssize_t t = 0;
        t <
        static_cast<py::ssize_t>(
            saved_labels.size()
        );
        ++t
    ) {

        for (
            py::ssize_t i = 0;
            i < n;
            ++i
        ) {

            labels_view(t, i) =
                saved_labels[t][i];
        }
    }


    py::array_t<int>
        K_out(K_trace.size());

    auto K_view =
        K_out.mutable_unchecked<1>();


    for (
        py::ssize_t t = 0;
        t <
        static_cast<py::ssize_t>(
            K_trace.size()
        );
        ++t
    ) {

        K_view(t) =
            K_trace[t];
    }


    py::dict result;

    result["labels"] =
        std::move(labels_out);

    result["K_trace"] =
        std::move(K_out);

    result["next_label"] =
        next_label;


    return result;
}


// ============================================================
// PYBIND11 MODULE
// ============================================================

PYBIND11_MODULE(ddp_cpp_fast, m) {

    m.doc() =
        "C++ Gibbs kernel for Stage 2C DDP simulation";

    m.def(
        "run_gibbs",
        &run_gibbs_cpp,
        py::arg("x"),
        py::arg("y"),
        py::arg("n_sweeps"),
        py::arg("burn_in"),
        py::arg("thin"),
        py::arg("M"),
        py::arg("seed"),
        "Run the Stage 2C Gibbs allocation sampler."
    );
}