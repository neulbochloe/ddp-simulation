import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor

PARALLEL_WORKERS = 6
REPLICATIONS_PER_COMBINATION = 520

from scipy.stats import norm

import ddp_cpp_fast as ddp_cpp

from ddp_stage2c_optimized_sempgy import (
    generate_data,
    true_mean,
    draw_F_all_x,
    ks_and_cdf,
    X_EVAL,
    CDF_GRID,
    M
)


# ============================================================
# STAGE 2C — FINAL SIMULATION
#
# Fixed-design conditional setup:
#
#     X fixed
#     Y | X=x ~ F_x*
#
# Truth:
#
#     F_x* = N(f_0(x),1), scenario-dependent
#
#
# RQ1 — POSTERIOR LEARNING
#
#   Does the posterior learn F_x*?
#
#   Diagnostics:
#       1. Mean draw-wise KS distance
#       2. Posterior-mean CDF KS distance
#
#
# RQ2 — FUNCTIONAL UNCERTAINTY QUANTIFICATION
#
#   Functional:
#
#       h(y) = y
#
#   Therefore:
#
#       theta_x = int y dF_x(y)
#
#   Under the truth:
#
#       theta_x* = 0
#
#   Diagnostics:
#       1. Frequentist coverage of 95% posterior CI
#       2. Mean CI width
#       3. Mean posterior functional
#
#
# COMPUTATION
#
#   Gibbs allocation sampler:
#       C++ / pybind11
#
#   Posterior reconstruction and diagnostics:
#       Python
#
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

SEED = 20261004


# ------------------------------------------------------------
# FINAL SAMPLE SIZES
# ------------------------------------------------------------

N_VALUES = [
    50,
    200,
    1000
]

SCENARIOS = ["baseline", "linear", "quadratic"]


# ------------------------------------------------------------
# FINAL NUMBER OF REPLICATIONS
#
# Change only this value when needed.
#
# Recommended workflow:
#
#     R = 10       final-code sanity check
#     R = 100      optional pilot
#     R = 5000  # legacy display only; scenario-specific R_current used in main()     final simulation
# ------------------------------------------------------------

R = 5000


# ------------------------------------------------------------
# MCMC
# ------------------------------------------------------------

N_ITER = 110000
BURN_IN = 10000
THIN = 10


# ------------------------------------------------------------
# TRUE FUNCTIONAL
#
# h(y) = y
# F_x* = N(f_0(x),1)
#
# therefore E[Y] = 0
# ------------------------------------------------------------


# Truth depends on scenario and evaluation location.



# ============================================================
# LABEL VECTOR -> CLUSTER DICTIONARY
# ============================================================

def labels_to_clusters(labels):

    clusters = {}

    for i, label in enumerate(labels):

        label = int(label)

        if label not in clusters:
            clusters[label] = []

        clusters[label].append(i)

    # Structural safeguard:
    # Stage 2C clusters must have size <= 2.

    for members in clusters.values():

        if len(members) > 2:

            raise RuntimeError(
                "C++ Gibbs output contains "
                "a cluster larger than 2."
            )

    return clusters


# ============================================================
# RUN ONE REPLICATION
# ============================================================

def run_replication(
    n,
    replication,
    scenario
):

    # --------------------------------------------------------
    # 1. REPRODUCIBLE RNG STREAMS
    # --------------------------------------------------------

    scenario_index = SCENARIOS.index(scenario)
    base_seed = (
        SEED
        + 1000000 * n
        + 10000000000 * scenario_index
        + replication
    )

    data_rng = np.random.default_rng(
        base_seed + 100000
    )

    posterior_rng = np.random.default_rng(
        base_seed + 200000
    )

    cpp_seed = int(
        base_seed + 300000
    )


    # --------------------------------------------------------
    # 2. GENERATE DATA
    # --------------------------------------------------------

    x, y = generate_data(
        n,
        data_rng,
        scenario
    )


    # --------------------------------------------------------
    # 3. C++ GIBBS SAMPLER
    # --------------------------------------------------------

    gibbs_start = time.perf_counter()

    cpp_result = ddp_cpp.run_gibbs(
        x,
        y,
        N_ITER,
        BURN_IN,
        THIN,
        M,
        cpp_seed
    )

    gibbs_runtime = (
        time.perf_counter()
        - gibbs_start
    )


    saved_labels = cpp_result[
        "labels"
    ]

    K_trace = np.asarray(
        cpp_result["K_trace"]
    )


    # --------------------------------------------------------
    # 4. BASIC OUTPUT CHECKS
    # --------------------------------------------------------

    expected_saved = (
        N_ITER - BURN_IN
    ) // THIN

    if saved_labels.shape != (
        expected_saved,
        n
    ):

        raise RuntimeError(
            "Unexpected C++ saved-label shape: "
            f"{saved_labels.shape}"
        )

    if len(K_trace) != (
        N_ITER - BURN_IN
    ):

        raise RuntimeError(
            "Unexpected K-trace length."
        )


    # --------------------------------------------------------
    # 5. STORAGE FOR POSTERIOR DIAGNOSTICS
    # --------------------------------------------------------

    n_x = len(X_EVAL)
    n_grid = len(CDF_GRID)

    draw_ks_sum = np.zeros(
        n_x,
        dtype=float
    )

    cdf_sum = np.zeros(
        (n_x, n_grid),
        dtype=float
    )

    functional_draws = [
        np.empty(
            expected_saved,
            dtype=float
        )
        for _ in range(n_x)
    ]


    # --------------------------------------------------------
    # 6. POSTERIOR RECONSTRUCTION
    # --------------------------------------------------------

    posterior_start = time.perf_counter()

    for t, labels in enumerate(
        saved_labels
    ):

        clusters = labels_to_clusters(
            labels
        )

        atoms_func, weights = draw_F_all_x(
            clusters, x, y, posterior_rng
        )
        atoms_all_x = [atoms_func(x_eval) for x_eval in X_EVAL]

        for q in range(n_x):

            atoms = atoms_all_x[q]

            # --------------------------------------------
            # RQ1:
            # draw-wise KS + CDF accumulation
            # --------------------------------------------

            draw_ks, cdf = ks_and_cdf(
                atoms,
                weights,
                float(true_mean(X_EVAL[q], scenario))
            )

            draw_ks_sum[q] += draw_ks

            cdf_sum[q] += cdf


            # --------------------------------------------
            # RQ2:
            # theta_x = int y dF_x(y)
            # --------------------------------------------

            functional_draws[q][t] = (
                np.sum(
                    weights * atoms
                )
            )


    posterior_runtime = (
        time.perf_counter()
        - posterior_start
    )


    # --------------------------------------------------------
    # 7. SUMMARIZE THIS REPLICATION
    # --------------------------------------------------------

    rq1 = []
    rq2 = []


    for q in range(n_x):

        true_cdf = norm.cdf(
            CDF_GRID,
            loc=float(true_mean(X_EVAL[q], scenario)),
            scale=1.0
        )

        # --------------------------------------------
        # RQ1
        # --------------------------------------------

        mean_draw_ks = (
            draw_ks_sum[q]
            / expected_saved
        )

        posterior_mean_cdf = (
            cdf_sum[q]
            / expected_saved
        )

        posterior_mean_cdf_ks = (
            np.max(
                np.abs(
                    posterior_mean_cdf
                    - true_cdf
                )
            )
        )

        rq1.append({
            "draw_ks":
                mean_draw_ks,

            "cdf_ks":
                posterior_mean_cdf_ks
        })


        # --------------------------------------------
        # RQ2
        # --------------------------------------------

        func = functional_draws[q]

        posterior_mean = np.mean(
            func
        )

        ci_lower, ci_upper = (
            np.quantile(
                func,
                [0.025, 0.975]
            )
        )

        ci_width = (
            ci_upper
            - ci_lower
        )

        true_functional = float(true_mean(X_EVAL[q], scenario))
        covered = (ci_lower <= true_functional <= ci_upper)

        rq2.append({
            "posterior_mean":
                posterior_mean,

            "ci_lower":
                ci_lower,

            "ci_upper":
                ci_upper,

            "ci_width":
                ci_width,

            "covered":
                covered
        })


    # --------------------------------------------------------
    # 8. RETURN
    # --------------------------------------------------------

    return {

        "mean_K":
            np.mean(K_trace),

        "rq1":
            rq1,

        "rq2":
            rq2,

        "timing": {

            "gibbs":
                gibbs_runtime,

            "posterior":
                posterior_runtime,

            "total":
                gibbs_runtime
                + posterior_runtime
        }
    }


# ============================================================
# RUN ONE SAMPLE SIZE
# ============================================================

def _run_replication_task(task):
    """Picklable top-level worker for macOS spawn multiprocessing."""
    n, replication, scenario = task
    return run_replication(n, replication, scenario)


def run_sample_size(
    n,
    R_current,
    scenario
):

    print()
    print("=" * 100)
    print(
        f"FINAL STAGE 2C SIMULATION — scenario={scenario}, n={n}"
    )
    print("=" * 100)

    print(
        f"Replications:       {R_current}"
    )

    print(
        f"MCMC iterations:    {N_ITER}"
    )

    print(
        f"Burn-in:            {BURN_IN}"
    )

    print(
        f"Thinning:           {THIN}"
    )

    print(
        f"Saved draws / rep:  "
        f"{(N_ITER - BURN_IN) // THIN}"
    )

    print()


    results = []

    sample_start = time.perf_counter()


    # Independent replications are executed in parallel. Executor.map
    # returns results in replication order, preserving deterministic seeds.
    with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        tasks = ((n, r, scenario) for r in range(R_current))
        for r, result in enumerate(executor.map(_run_replication_task, tasks)):
            results.append(result)

            if (
                R_current <= 20
                or r == 0
                or (r + 1) % 100 == 0
                or (r + 1) == R_current
            ):
                elapsed = time.perf_counter() - sample_start
                remaining = elapsed / (r + 1) * (R_current - r - 1)
                print(
                    f"Replication {r + 1:5d}/{R_current}: "
                    f"mean K = {result['mean_K']:.2f}, "
                    f"time = {result['timing']['total']:.3f} sec, "
                    f"ETA (rough) = {remaining / 60:.1f} min",
                    flush=True,
                )


    sample_runtime = (
        time.perf_counter()
        - sample_start
    )


    # ========================================================
    # AGGREGATE
    # ========================================================

    mean_K = np.mean([
        z["mean_K"]
        for z in results
    ])


    mean_gibbs_runtime = np.mean([
        z["timing"]["gibbs"]
        for z in results
    ])

    mean_posterior_runtime = np.mean([
        z["timing"]["posterior"]
        for z in results
    ])

    mean_total_runtime = np.mean([
        z["timing"]["total"]
        for z in results
    ])


    # ========================================================
    # RQ1 SUMMARY
    # ========================================================

    rq1_summary = []

    for q, x_eval in enumerate(
        X_EVAL
    ):

        draw_ks_values = np.asarray([
            z["rq1"][q]["draw_ks"]
            for z in results
        ])

        cdf_ks_values = np.asarray([
            z["rq1"][q]["cdf_ks"]
            for z in results
        ])


        rq1_summary.append({

            "x":
                x_eval,

            "draw_ks_mean":
                np.mean(
                    draw_ks_values
                ),

            "draw_ks_sd":
                np.std(
                    draw_ks_values,
                    ddof=1
                )
                if R_current > 1
                else np.nan,

            "cdf_ks_mean":
                np.mean(
                    cdf_ks_values
                ),

            "cdf_ks_sd":
                np.std(
                    cdf_ks_values,
                    ddof=1
                )
                if R_current > 1
                else np.nan
        })


    # ========================================================
    # RQ2 SUMMARY
    # ========================================================

    rq2_summary = []

    for q, x_eval in enumerate(
        X_EVAL
    ):

        posterior_means = np.asarray([
            z["rq2"][q][
                "posterior_mean"
            ]
            for z in results
        ])

        ci_widths = np.asarray([
            z["rq2"][q][
                "ci_width"
            ]
            for z in results
        ])

        coverage = np.mean([
            z["rq2"][q][
                "covered"
            ]
            for z in results
        ])


        rq2_summary.append({

            "x":
                x_eval,

            "coverage":
                coverage,

            "mean_ci_width":
                np.mean(
                    ci_widths
                ),

            "mean_posterior_functional":
                np.mean(
                    posterior_means
                ),

            "sd_posterior_functional":
                np.std(
                    posterior_means,
                    ddof=1
                )
                if R_current > 1
                else np.nan
        })


    # ========================================================
    # PRINT FINAL RESULT FOR THIS n
    # ========================================================

    print()
    print("=" * 100)
    print(
        f"RQ1 — POSTERIOR LEARNING: n = {n}"
    )
    print("=" * 100)

    print(
        f"{'x':>8}"
        f"{'draw-KS':>14}"
        f"{'SD':>14}"
        f"{'CDF-KS':>14}"
        f"{'SD':>14}"
    )

    print("-" * 64)


    for z in rq1_summary:

        print(
            f"{z['x']:>8.2f}"
            f"{z['draw_ks_mean']:>14.4f}"
            f"{z['draw_ks_sd']:>14.4f}"
            f"{z['cdf_ks_mean']:>14.4f}"
            f"{z['cdf_ks_sd']:>14.4f}"
        )


    print()
    print("=" * 100)
    print(
        f"RQ2 — FUNCTIONAL UQ: n = {n}"
    )
    print("=" * 100)

    print(
        f"{'x':>8}"
        f"{'coverage':>14}"
        f"{'CI width':>14}"
        f"{'post mean':>14}"
        f"{'SD mean':>14}"
    )

    print("-" * 64)


    for z in rq2_summary:

        print(
            f"{z['x']:>8.2f}"
            f"{z['coverage']:>14.4f}"
            f"{z['mean_ci_width']:>14.4f}"
            f"{z['mean_posterior_functional']:>14.4f}"
            f"{z['sd_posterior_functional']:>14.4f}"
        )


    print()
    print("=" * 100)
    print(
        f"COMPUTATION: n = {n}"
    )
    print("=" * 100)

    print(
        f"Mean occupied K:             "
        f"{mean_K:.2f}"
    )

    print(
        f"Mean C++ Gibbs / rep:        "
        f"{mean_gibbs_runtime:.4f} sec"
    )

    print(
        f"Mean posterior work / rep:   "
        f"{mean_posterior_runtime:.4f} sec"
    )

    print(
        f"Mean total / rep:            "
        f"{mean_total_runtime:.4f} sec"
    )

    print(
        f"Actual runtime for n={n}:     "
        f"{sample_runtime:.2f} sec "
        f"({sample_runtime / 60:.2f} min)"
    )


    return {

        "scenario": scenario,

        "n":
            n,

        "R":
            R_current,

        "mean_K":
            mean_K,

        "rq1":
            rq1_summary,

        "rq2":
            rq2_summary,

        "mean_gibbs_runtime":
            mean_gibbs_runtime,

        "mean_posterior_runtime":
            mean_posterior_runtime,

        "mean_total_runtime":
            mean_total_runtime,

        "actual_runtime":
            sample_runtime
    }


# ============================================================
# FINAL CROSS-n SUMMARY
# ============================================================

def print_cross_n_summary(
    all_results
):

    print()
    print()
    print("=" * 100)
    print(
        "FINAL STAGE 2C — CROSS-n SUMMARY"
    )
    print("=" * 100)


    # --------------------------------------------------------
    # COMPUTATION
    # --------------------------------------------------------

    print()
    print("COMPUTATION")
    print("-" * 100)

    print(
        f"{'n':>8}"
        f"{'R':>10}"
        f"{'mean K':>14}"
        f"{'Gibbs sec':>14}"
        f"{'post sec':>14}"
        f"{'total sec':>14}"
    )

    print("-" * 74)


    for result in all_results:

        print(
            f"{result['n']:>8d}"
            f"{result['R']:>10d}"
            f"{result['mean_K']:>14.2f}"
            f"{result['mean_gibbs_runtime']:>14.4f}"
            f"{result['mean_posterior_runtime']:>14.4f}"
            f"{result['mean_total_runtime']:>14.4f}"
        )


    # --------------------------------------------------------
    # RQ1: POSTERIOR-MEAN CDF KS
    # --------------------------------------------------------

    print()
    print(
        "RQ1 — POSTERIOR-MEAN CDF KS"
    )

    print("-" * 100)

    print(
        f"{'n':>8}"
        + "".join(
            f"{x_eval:>14.2f}"
            for x_eval in X_EVAL
        )
    )

    print("-" * 78)


    for result in all_results:

        values = [
            z["cdf_ks_mean"]
            for z in result["rq1"]
        ]

        print(
            f"{result['n']:>8d}"
            + "".join(
                f"{value:>14.4f}"
                for value in values
            )
        )


    # --------------------------------------------------------
    # RQ2: COVERAGE
    # --------------------------------------------------------

    print()
    print(
        "RQ2 — 95% FUNCTIONAL CI COVERAGE"
    )

    print("-" * 100)

    print(
        f"{'n':>8}"
        + "".join(
            f"{x_eval:>14.2f}"
            for x_eval in X_EVAL
        )
    )

    print("-" * 78)


    for result in all_results:

        values = [
            z["coverage"]
            for z in result["rq2"]
        ]

        print(
            f"{result['n']:>8d}"
            + "".join(
                f"{value:>14.4f}"
                for value in values
            )
        )


    print("=" * 100)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print(
        "STAGE 2C — FINAL DDP SIMULATION"
    )
    print("=" * 100)

    print()

    print(
        "Setup: X fixed, "
        "Y | X=x ~ F_x*"
    )

    print(f"Scenarios: {SCENARIOS}")

    print()

    print(
        f"Sample sizes: "
        f"{N_VALUES}"
    )

    print(
        f"Replications per n: "
        f"{R}"
    )

    print(
        f"MCMC: "
        f"{N_ITER} iterations, "
        f"{BURN_IN} burn-in, "
        f"thin = {THIN}"
    )

    print(
        f"Evaluation x: "
        f"{X_EVAL}"
    )

    print()

    overall_start = (
        time.perf_counter()
    )


    all_results = []


    for scenario in SCENARIOS:
        for n in N_VALUES:
            R_current = REPLICATIONS_PER_COMBINATION
            result = run_sample_size(n, R_current, scenario)
            all_results.append(result)

    for scenario in SCENARIOS:
        print(f"\nSCENARIO: {scenario}")
        print_cross_n_summary(
            [r for r in all_results if r["scenario"] == scenario]
        )


    overall_runtime = (
        time.perf_counter()
        - overall_start
    )


    print()

    print(
        f"TOTAL WALL-CLOCK RUNTIME: "
        f"{overall_runtime:.2f} sec "
        f"({overall_runtime / 60:.2f} min, "
        f"{overall_runtime / 3600:.3f} h)"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()