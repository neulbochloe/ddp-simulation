import numpy as np
import matplotlib.pyplot as plt
import time
import math
from scipy.stats import norm

# ============================================================
# STAGE 2C — STRUCTURE-OPTIMIZED IMPLEMENTATION
#
# Same statistical setup:
#   X_i fixed
#   Y_i | X_i=x_i ~ F_x^*
#   F_x^* = N(0,1), for all x
#
# Fitted pure single-p DDP:
#   F_x = sum_j p_j delta_{theta_j(x)}
#   theta_j(x) = a_j + b_j x
#   a_j ~ N(0,4), b_j ~ N(0,1)
#
# No mixture kernel.
#
# IMPORTANT:
# For distinct fixed x_i and continuous iid Y_i, a cluster containing
# two observations determines a line exactly. A third continuous
# observation lies exactly on that line with probability zero.
# Hence, after removing observation i, only singleton clusters can
# have positive probability of accepting i (a.s. under this setup).
#
# Posterior weights and trajectories are drawn ONCE per retained
# posterior draw and shared across all evaluation x-values, as
# required by the single-p construction.
# ============================================================

# ----------------------------
# 0. SETTINGS
# ----------------------------

SEED = 20261004
M = 1.0

# Validation first
N = 50
R = 10

N_ITER = 1000
BURN_IN = 200
THIN = 10

X_EVAL = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
CDF_GRID = np.linspace(-6.0, 8.0, 2801)

J_RESIDUAL = 100

LOG_2PI = math.log(2.0 * math.pi)


# ----------------------------
# 1. DATA
# ----------------------------

def true_mean(x, scenario):
    if scenario == "baseline":
        return np.zeros_like(x, dtype=float)
    elif scenario == "linear":
        return 1.0 + x
    elif scenario == "quadratic":
        return x * (1.0 + x)
    elif scenario == "sobolev":
        return np.sin(2.0 * np.pi * x)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


def generate_data(n, rng, scenario):
    x = np.linspace(0.0, 1.0, n)
    eps = rng.normal(0.0, 1.0, size=n)
    y = true_mean(x, scenario) + eps
    return x, y


# ----------------------------
# 2. FAST SCALAR NORMAL LOGPDF
# ----------------------------

def normal_logpdf_scalar(y, mu, var):
    return -0.5 * (LOG_2PI + math.log(var) + ((y - mu) ** 2) / var)


def log_m0_scalar(y_i, x_i):
    var = 4.0 + x_i * x_i
    return normal_logpdf_scalar(y_i, 0.0, var)


# ----------------------------
# 3. CLUSTER STATE
# ----------------------------

def initialize_clusters(n):
    labels = np.arange(n, dtype=np.int64)
    clusters = {i: [i] for i in range(n)}
    next_label = n
    return labels, clusters, next_label


def remove_observation(i, labels, clusters):
    old_label = int(labels[i])
    members = clusters[old_label]

    if len(members) == 1:
        del clusters[old_label]
    else:
        # Under this model clusters are a.s. size <= 2.
        if members[0] == i:
            members.pop(0)
        else:
            members.pop()

    labels[i] = -1


def add_observation(i, label, labels, clusters):
    if label in clusters:
        clusters[label].append(i)
    else:
        clusters[label] = [i]
    labels[i] = label


# ----------------------------
# 4. SINGLETON PREDICTIVE
# ----------------------------

def singleton_log_predictive(i, k, x, y):
    x_i = x[i]
    x_k = x[k]
    y_i = y[i]
    y_k = y[k]

    denom = 4.0 + x_k * x_k
    cov = 4.0 + x_i * x_k

    mu = (cov / denom) * y_k
    var = 4.0 + x_i * x_i - (cov * cov) / denom

    # With distinct x_i, var should be strictly positive.
    if var <= 0.0:
        return -math.inf

    return normal_logpdf_scalar(y_i, mu, var)


# ----------------------------
# 5. STRUCTURE-OPTIMIZED GIBBS SWEEP
# ----------------------------

def gibbs_sweep_structure(labels, clusters, next_label, x, y, rng):
    n = len(y)

    for i in rng.permutation(n):
        remove_observation(i, labels, clusters)

        # Only singleton clusters can accept i with positive probability
        # a.s. under the present continuous/noiseless atomic setup.
        singleton_labels = []
        singleton_members = []

        for label, members in clusters.items():
            if len(members) == 1:
                singleton_labels.append(label)
                singleton_members.append(members[0])

        s = len(singleton_labels)

        # Unnormalized log probabilities:
        # singleton cluster count is 1, so log(count)=0.
        logw = np.empty(s + 1, dtype=float)

        for j in range(s):
            logw[j] = singleton_log_predictive(
                i,
                singleton_members[j],
                x,
                y
            )

        # New cluster
        logw[s] = math.log(M) + log_m0_scalar(y[i], x[i])

        # Stable normalization without scipy.special.logsumexp
        m = np.max(logw)
        probs = np.exp(logw - m)
        probs /= probs.sum()

        chosen = rng.choice(s + 1, p=probs)

        if chosen < s:
            chosen_label = singleton_labels[chosen]
        else:
            chosen_label = next_label
            next_label += 1

        add_observation(i, chosen_label, labels, clusters)

    return labels, clusters, next_label


# ----------------------------
# 6. DRAW OCCUPIED TRAJECTORIES
# ----------------------------

def draw_occupied_trajectories(clusters, x, y, rng):
    cluster_labels = list(clusters.keys())
    K = len(cluster_labels)

    counts = np.empty(K, dtype=float)
    a = np.empty(K, dtype=float)
    b = np.empty(K, dtype=float)

    for j, label in enumerate(cluster_labels):
        members = clusters[label]
        counts[j] = len(members)

        if len(members) == 1:
            k = members[0]
            xk = x[k]
            yk = y[k]

            denom = 4.0 + xk * xk
            mean_b = xk * yk / denom
            var_b = 4.0 / denom

            bj = rng.normal(mean_b, math.sqrt(var_b))
            aj = yk - bj * xk

        else:
            r = members[0]
            s = members[1]

            bj = (y[s] - y[r]) / (x[s] - x[r])
            aj = y[r] - bj * x[r]

        a[j] = aj
        b[j] = bj

    return counts, a, b


# ----------------------------
# 7. RESIDUAL DP — ONE JOINT DRAW
# ----------------------------

def draw_residual_trajectories(rng):
    v = rng.beta(1.0, M, size=J_RESIDUAL - 1)

    weights = np.empty(J_RESIDUAL, dtype=float)
    remaining = 1.0

    for j in range(J_RESIDUAL - 1):
        weights[j] = remaining * v[j]
        remaining *= (1.0 - v[j])

    weights[-1] = remaining
    weights /= weights.sum()

    a = rng.normal(0.0, 2.0, size=J_RESIDUAL)
    b = rng.normal(0.0, 1.0, size=J_RESIDUAL)

    return weights, a, b


# ----------------------------
# 8. JOINT POSTERIOR DRAW ACROSS ALL x
# ----------------------------

def draw_F_all_x(clusters, x, y, rng):
    counts, occ_a, occ_b = draw_occupied_trajectories(
        clusters, x, y, rng
    )

    K = len(counts)

    # One common posterior weight draw for all x
    alpha = np.empty(K + 1, dtype=float)
    alpha[:K] = counts
    alpha[K] = M

    mass_draw = rng.dirichlet(alpha)
    occupied_weights = mass_draw[:K]
    residual_mass = mass_draw[K]

    residual_weights, res_a, res_b = draw_residual_trajectories(rng)
    residual_weights *= residual_mass

    weights = np.concatenate((occupied_weights, residual_weights))
    weights /= weights.sum()

    all_a = np.concatenate((occ_a, res_a))
    all_b = np.concatenate((occ_b, res_b))

    # Shape: (n_x, n_atoms)
    def compute_atoms(x_eval):
        return all_a + x_eval * all_b
    #atoms = all_a[None, :] + X_EVAL[:, None] * all_b[None, :]

    return compute_atoms, weights


# ----------------------------
# 9. KS + CDF
# ----------------------------

def ks_and_cdf(atoms, weights, true_mu):
    order = np.argsort(atoms)
    z = atoms[order]
    w = weights[order]

    cum = np.cumsum(w)

    before = np.empty_like(cum)
    before[0] = 0.0
    before[1:] = cum[:-1]

    # True CDF: N(true_mu, 1)
    true_at_atoms = norm.cdf(z - true_mu)

    # Exact KS distance for a discrete posterior draw
    d_right = np.max(np.abs(cum - true_at_atoms))
    d_left = np.max(np.abs(before - true_at_atoms))
    ks = max(d_right, d_left)

    # Posterior CDF evaluated on a fixed grid
    idx = np.searchsorted(z, CDF_GRID, side="right")

    cdf = np.zeros_like(CDF_GRID)
    mask = idx > 0
    cdf[mask] = cum[idx[mask] - 1]

    return ks, cdf


# ----------------------------
# 10. ONE REPLICATION
# ----------------------------

def run_replication(n, rng):
    x, y = generate_data(n, rng)

    labels, clusters, next_label = initialize_clusters(n)

    n_x = len(X_EVAL)
    n_saved = ((N_ITER - BURN_IN - 1) // THIN) + 1

    ks_draws = np.empty((n_x, n_saved), dtype=float)
    functional_draws = np.empty((n_x, n_saved), dtype=float)
    cdf_sum = np.zeros((n_x, len(CDF_GRID)), dtype=float)

    K_sum = 0.0
    K_count = 0
    save_index = 0

    # Timing profile
    gibbs_time = 0.0
    posterior_time = 0.0
    diagnostic_time = 0.0

    for iteration in range(N_ITER):
        t0 = time.perf_counter()

        labels, clusters, next_label = gibbs_sweep_structure(
            labels,
            clusters,
            next_label,
            x,
            y,
            rng
        )

        gibbs_time += time.perf_counter() - t0

        if iteration < BURN_IN:
            continue

        K_sum += len(clusters)
        K_count += 1

        if (iteration - BURN_IN) % THIN != 0:
            continue

        # One joint single-p posterior draw across all evaluation x
        t0 = time.perf_counter()

        atoms_all_x, weights = draw_F_all_x(
            clusters,
            x,
            y,
            rng
        )

        posterior_time += time.perf_counter() - t0

        t0 = time.perf_counter()

        # h(y)=y for all x at once
        functional_draws[:, save_index] = atoms_all_x @ weights

        # KS and CDF
        for q in range(n_x):
            ks, cdf = ks_and_cdf(
                atoms_all_x[q],
                weights
            )
            ks_draws[q, save_index] = ks
            cdf_sum[q] += cdf

        diagnostic_time += time.perf_counter() - t0
        save_index += 1

    # ------------------------
    # Summary
    # ------------------------

    true_cdf = norm.cdf(CDF_GRID)

    result = {
        "mean_K": K_sum / K_count,
        "ks": {},
        "functional": {},
        "timing": {
            "gibbs": gibbs_time,
            "posterior": posterior_time,
            "diagnostics": diagnostic_time
        }
    }

    for q, x_eval in enumerate(X_EVAL):
        ks_array = ks_draws[q, :save_index]
        func_array = functional_draws[q, :save_index]

        posterior_mean_cdf = cdf_sum[q] / save_index
        posterior_mean_cdf_ks = np.max(
            np.abs(posterior_mean_cdf - true_cdf)
        )

        ci_lower, ci_upper = np.quantile(
            func_array,
            [0.025, 0.975]
        )

        result["ks"][x_eval] = {
            "mean_drawwise": float(np.mean(ks_array)),
            "posterior_mean_cdf": float(posterior_mean_cdf_ks)
        }

        result["functional"][x_eval] = {
            "posterior_mean": float(np.mean(func_array)),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "ci_width": float(ci_upper - ci_lower),
            "covered": int(ci_lower <= 0.0 <= ci_upper)
        }

    return result


# ----------------------------
# 11. VALIDATION RUN
# ----------------------------

def run_validation():
    print("=" * 72)
    print("STAGE 2C STRUCTURE-OPTIMIZED — VALIDATION RUN")
    print("=" * 72)
    print("Same pure single-p DDP / no mixture kernel.")
    print("Truth: F_x* = N(0,1) for every x")
    print(f"n = {N}")
    print(f"R = {R}")
    print(f"MCMC iterations = {N_ITER}")
    print(f"Burn-in = {BURN_IN}")
    print(f"Thinning = {THIN}")
    print()

    results = []
    runtimes = []

    total_start = time.perf_counter()

    for r in range(R):
        rep_rng = np.random.default_rng(SEED + r)

        start = time.perf_counter()
        result = run_replication(N, rep_rng)
        runtime = time.perf_counter() - start

        results.append(result)
        runtimes.append(runtime)

        print(
            f"Replication {r + 1:2d}/{R}: "
            f"mean K = {result['mean_K']:.2f}, "
            f"time = {runtime:.3f} sec"
        )

    total_runtime = time.perf_counter() - total_start

    # ------------------------
    # RQ1
    # ------------------------

    print()
    print("=" * 72)
    print("RQ1: POSTERIOR LEARNING")
    print("=" * 72)

    mean_draw_ks = []
    mean_center_ks = []

    for x_eval in X_EVAL:
        draw_values = np.array([
            result["ks"][x_eval]["mean_drawwise"]
            for result in results
        ])

        center_values = np.array([
            result["ks"][x_eval]["posterior_mean_cdf"]
            for result in results
        ])

        draw_mean = np.mean(draw_values)
        center_mean = np.mean(center_values)

        mean_draw_ks.append(draw_mean)
        mean_center_ks.append(center_mean)

        print(
            f"x = {x_eval:4.2f}: "
            f"mean draw-wise KS = {draw_mean:.4f}, "
            f"posterior-mean CDF KS = {center_mean:.4f}"
        )

    # ------------------------
    # RQ2
    # ------------------------

    print()
    print("=" * 72)
    print("RQ2: FUNCTIONAL UQ — h(y)=y")
    print("=" * 72)

    for x_eval in X_EVAL:
        coverage = np.mean([
            result["functional"][x_eval]["covered"]
            for result in results
        ])

        widths = np.array([
            result["functional"][x_eval]["ci_width"]
            for result in results
        ])

        posterior_means = np.array([
            result["functional"][x_eval]["posterior_mean"]
            for result in results
        ])

        print(
            f"x = {x_eval:4.2f}: "
            f"coverage = {coverage:.3f}, "
            f"mean CI width = {np.mean(widths):.4f}, "
            f"mean posterior functional = {np.mean(posterior_means):.4f}"
        )

    # ------------------------
    # COMPUTATION
    # ------------------------

    mean_K = np.mean([result["mean_K"] for result in results])
    mean_runtime = np.mean(runtimes)

    mean_gibbs = np.mean([
        result["timing"]["gibbs"] for result in results
    ])

    mean_posterior = np.mean([
        result["timing"]["posterior"] for result in results
    ])

    mean_diagnostics = np.mean([
        result["timing"]["diagnostics"] for result in results
    ])

    print()
    print("=" * 72)
    print("COMPUTATION")
    print("=" * 72)
    print(f"Mean occupied K:                 {mean_K:.2f}")
    print(f"Total runtime:                   {total_runtime:.3f} sec")
    print(f"Mean runtime / replication:      {mean_runtime:.3f} sec")
    print()
    print("Mean timing profile / replication")
    print(f"  Gibbs sampler:                 {mean_gibbs:.3f} sec")
    print(f"  Posterior reconstruction:      {mean_posterior:.3f} sec")
    print(f"  KS/CDF/UQ diagnostics:         {mean_diagnostics:.3f} sec")
    print()
    print(
        f"Estimated serial R=1000 runtime: "
        f"{mean_runtime * 1000 / 3600:.3f} hours"
    )
    print(
        f"Estimated serial R=5000 runtime: "
        f"{mean_runtime * 5000 / 3600:.3f} hours"
    )

    # ------------------------
    # PLOT
    # ------------------------

    plt.figure(figsize=(7, 4))

    plt.plot(
        X_EVAL,
        mean_draw_ks,
        marker="o",
        label="Mean draw-wise KS"
    )

    plt.plot(
        X_EVAL,
        mean_center_ks,
        marker="s",
        label="Posterior-mean CDF KS"
    )

    plt.xlabel("x")
    plt.ylabel("KS distance")
    plt.title("Stage 2C: Structure-Optimized Validation")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------
# 12. MAIN
# ----------------------------

if __name__ == "__main__":
    run_validation()
