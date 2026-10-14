"""Single-replication linear DDP plots; run from the Codes directory.
Requires ddp_cpp_fast and ddp_stage2c_optimized_sempgy.
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import ddp_cpp_fast as ddp_cpp
from ddp_stage2c_optimized_sempgy import generate_data, true_mean, draw_F_all_x, M

N_ITER = 110_000
BURN_IN = 10_000
THIN = 10
SEED = 20261004
X_PLOT = np.linspace(0.0, 1.0, 1000)


def labels_to_clusters(labels):
    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(i)
    if any(len(members) > 2 for members in clusters.values()):
        raise RuntimeError('A cluster contains more than two observations.')
    return clusters


def plot_one(n, outdir):
    seed = SEED + 1_000_000 * n + 10_000_000_000
    data_rng = np.random.default_rng(seed + 100_000)
    posterior_rng = np.random.default_rng(seed + 200_000)
    x, y = generate_data(n, data_rng, 'linear')
    print(f'Running linear scenario, n={n:,}, one replication...', flush=True)
    start = time.perf_counter()
    cpp_result = ddp_cpp.run_gibbs(x, y, N_ITER, BURN_IN, THIN, M, int(seed + 300_000))
    saved_labels = cpp_result['labels']
    expected = (N_ITER - BURN_IN) // THIN
    if saved_labels.shape != (expected, n):
        raise RuntimeError(f'Unexpected labels shape: {saved_labels.shape}')

    # Every conditional-mean draw is affine: mu_t(x)=A_t+B_t*x.
    # Save only A_t and B_t, avoiding a 1000-by-10000-by-K tensor.
    intercepts = np.empty(expected, dtype=float)
    slopes = np.empty(expected, dtype=float)
    for t, labels in enumerate(saved_labels):
        clusters = labels_to_clusters(labels)
        atoms_func, weights = draw_F_all_x(clusters, x, y, posterior_rng)
        intercepts[t] = np.dot(weights, atoms_func(0.0))
        slopes[t] = np.dot(weights, atoms_func(1.0)) - intercepts[t]
        if (t + 1) % 1000 == 0:
            print(f'  Posterior draws reconstructed: {t + 1}/{expected}', flush=True)

    # Quantiles computed in small blocks to keep peak memory bounded.
    post_mean = intercepts.mean() + X_PLOT * slopes.mean()
    ci_lower = np.empty_like(X_PLOT)
    ci_upper = np.empty_like(X_PLOT)
    for start_i in range(0, len(X_PLOT), 50):
        end_i = min(start_i + 50, len(X_PLOT))
        values = intercepts[:, None] + slopes[:, None] * X_PLOT[None, start_i:end_i]
        ci_lower[start_i:end_i], ci_upper[start_i:end_i] = np.quantile(
            values, [0.025, 0.975], axis=0
        )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(X_PLOT, ci_lower, ci_upper, color='gray', alpha=0.35,
                    label='95% credible interval')
    ax.plot(X_PLOT, post_mean, label='Posterior mean', linewidth=2)
    ax.plot(X_PLOT, true_mean(X_PLOT, 'linear'), label='True conditional mean', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('Conditional mean')
    ax.set_title(rf'Linear scenario ($n={n:,}$, one replication)')
    ax.legend()
    fig.tight_layout()
    outfile = outdir / f'linear_n{n}.png'
    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile} (elapsed {time.perf_counter() - start:.1f}s)', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, nargs='+', default=[50, 1000])
    parser.add_argument('--outdir', default='ddp_figures')
    parser.add_argument('--allow-million', action='store_true')
    args = parser.parse_args()
    if any(n <= 0 for n in args.n):
        parser.error('Sample sizes must be positive.')
    if 1_000_000 in args.n and not args.allow_million:
        parser.error('n=1,000,000 requires a different scalable sampler. The current C++ '
                     'sampler returns a ~40GB+ labels array and is computationally infeasible '
                     'at this size. Do not bypass without redesigning the sampler.')
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for n in args.n:
        plot_one(n, outdir)


if __name__ == '__main__':
    main()
