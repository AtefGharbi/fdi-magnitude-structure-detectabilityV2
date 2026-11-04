"""Empirical validation of the controlled correlation manipulation (Supplementary Note S2).

Reproduces, seed by seed, the random draws of run_v2.make_sets (channel set S, sign
pattern u, sensor noise, z and c) and checks that the deviation
    p = t * ( sqrt(1-rho) * z + sqrt(rho) * c * u )
has (i) the same per-channel marginal in every condition, (ii) pairwise correlation
rho*u_i*u_j, (iii) covariance eigenvalues t^2(1-rho+k*rho) [once] and t^2(1-rho) [k-1 times],
and (iv) participation ratio PR = (tr S)^2 / tr(S^2) as predicted.
An end-to-end check runs run_v2.dataset() on real power-flow data for two seeds per system
and confirms that the injected deviation in the dataset equals the replicated one.

Usage:  python3 validate_manipulation.py            (writes validation.json; run fig_validation.py for Figure S4)
        python3 validate_manipulation.py --e2e      (adds the end-to-end check; needs pandapower)
"""
import sys, json
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from scipy import stats

SIGMA, N_NORMAL, N_ATT = 0.01, 1200, 300
NF = {"14-bus": 58, "118-bus": 582}
MAIN_K = {"14-bus": 10, "118-bus": 47}
SWEEP_K = {"14-bus": [3, 5, 10, 20, 29], "118-bus": [10, 20, 30, 47, 70, 99]}
SEEDS = [0, 1, 2, 3, 4, 42, 123, 999] + list(range(5, 17))
RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
SNR = 4.0
N_LARGE = 200_000


def draws(system, seed, k):
    """Identical RNG calls, in identical order, to run_v2.make_sets."""
    r = np.random.default_rng([seed, k, 1])
    S = r.choice(NF[system], k, replace=False); u = r.choice([-1.0, 1.0], k)
    r2 = np.random.default_rng([seed, k, 2])
    noise = r2.normal(0, SIGMA, size=(N_NORMAL + N_ATT, NF[system]))
    Z = r2.normal(size=(N_ATT, k)); C = r2.normal(size=N_ATT)
    return S, u, noise[N_NORMAL:][:, S], Z, C


def deviation(Z, C, u, rho, snr=SNR):
    return snr * SIGMA * (np.sqrt(1 - rho) * Z + np.sqrt(rho) * C[:, None] * u[None, :])


def theory(k, rho):
    lam1, lam = 1 - rho + k * rho, 1 - rho
    pr = k ** 2 / (lam1 ** 2 + (k - 1) * lam ** 2)
    return lam1, lam, pr


def pr_of(cov):
    ev = np.linalg.eigvalsh(cov)
    return ev.sum() ** 2 / (ev ** 2).sum(), np.sort(ev)[::-1]


def stats_for(P, u, t2):
    k = P.shape[1]
    R = np.corrcoef(P.T); off = ~np.eye(k, dtype=bool)
    signed = (R * np.outer(u, u))[off]
    cov = np.cov(P.T); pr, ev = pr_of(cov)
    return dict(r_signed_mean=float(signed.mean()), r_signed_sd=float(signed.std()),
                var_ratio=float(np.diag(cov).mean() / t2), lam1=float(ev[0] / t2),
                lam_rest=float(ev[1:].mean() / t2), pr=float(pr), ev=(ev / t2).tolist())


def main():
    out = {"main": {}, "sweep": {}, "marginals": {}, "separability": {}, "large_n": {}}
    t2 = (SNR * SIGMA) ** 2
    for system in NF:
        k = MAIN_K[system]
        for rho in RHOS:
            rows = []
            for seed in SEEDS:
                S, u, nz, Z, C = draws(system, seed, k)
                P = deviation(Z, C, u, rho)
                a = stats_for(P, u, t2)
                b = stats_for(P + nz, u, t2 + SIGMA ** 2)      # deviation + sensor noise
                rows.append(dict(seed=seed, dev=a, dev_noise=dict(r_signed_mean=b["r_signed_mean"])))
            out["main"][f"{system}|{rho}"] = rows
            # large-sample check of the generator (seed 0 channel set and sign pattern)
            S, u, *_ = draws(system, 0, k)
            g = np.random.default_rng([999, k, int(rho * 100)])
            P = deviation(g.normal(size=(N_LARGE, k)), g.normal(size=N_LARGE), u, rho)
            out["large_n"][f"{system}|{rho}"] = stats_for(P, u, t2)
        # marginals: rho=0 vs rho=1 (each channel, same seed draws); channels within rho=1 share c,
        # so one test per seed uses channel 0 only; the large-N test uses fresh draws per channel.
        ks, vr, sk = [], [], []
        for seed in SEEDS:
            S, u, nz, Z, C = draws(system, seed, k)
            a, b = deviation(Z, C, u, 0.0), deviation(Z, C, u, 1.0)
            ks.append(stats.ks_2samp(a[:, 0], b[:, 0]).pvalue)
            vr.append(float(b.var(0).mean() / a.var(0).mean()))
        S, u, *_ = draws(system, 0, k); ksch = []
        for j in range(k):
            g = np.random.default_rng([j, 77])
            a = deviation(g.normal(size=(N_LARGE, k)), g.normal(size=N_LARGE), u, 0.0)[:, j]
            b = deviation(g.normal(size=(N_LARGE, k)), g.normal(size=N_LARGE), u, 1.0)[:, j]
            ksch.append(stats.ks_2samp(a, b).pvalue)
        vr = np.array(vr)
        out["marginals"][system] = dict(ks_seed_ch0_median_p=float(np.median(ks)), ks_seed_ch0_frac_below_05=float(np.mean(np.array(ks) < 0.05)),
                                         var_ratio_mean=float(vr.mean()), var_ratio_ci=[float(vr.mean()-2.093*vr.std(ddof=1)/np.sqrt(len(vr))), float(vr.mean()+2.093*vr.std(ddof=1)/np.sqrt(len(vr)))],
                                         ks_largeN_min_p=float(min(ksch)), ks_largeN_frac_below_05=float(np.mean(np.array(ksch) < 0.05)), n_channels=k)
        # separability of rho=0 vs rho=1 from noise-free deviations (contrast: 0.502 for the per-sample construction)
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        g = np.random.default_rng(11); S, u, *_ = draws(system, 0, k); n = 5000
        X = np.vstack([deviation(g.normal(size=(n, k)), g.normal(size=n), u, 0.0),
                       deviation(g.normal(size=(n, k)), g.normal(size=n), u, 1.0)])
        y = np.r_[np.zeros(n), np.ones(n)]
        out["separability"][system] = float(cross_val_score(RandomForestClassifier(100, random_state=0), X, y, cv=5).mean())
        # channel-count sweep (8 original seeds)
        for k2 in SWEEP_K[system]:
            for rho in (0.0, 1.0):
                v = []
                for seed in SEEDS[:8]:
                    S, u, nz, Z, C = draws(system, seed, k2)
                    s = stats_for(deviation(Z, C, u, rho), u, t2)
                    v.append((s["r_signed_mean"], s["pr"]))
                v = np.array(v)
                out["sweep"][f"{system}|{k2}|{rho}"] = dict(r=float(v[:, 0].mean()), pr=float(v[:, 1].mean()),
                                                             pr_theory=theory(k2, rho)[2])
    if "--e2e" in sys.argv:
        out["e2e"] = e2e()
    json.dump(out, open(os.path.join(ROOT, "validation.json"), "w"), indent=1)
    return out


def e2e():
    """End-to-end: run_v2.dataset() on real operating points; deviation in X equals replicated p."""
    import run_v2
    res = {}
    for system in NF:
        for seed in (0, 42):
            k = MAIN_K[system]
            sets = run_v2.make_sets(system, seed, k)
            B, S, u, noise, Z, C = sets
            S2, u2, *_ = draws(system, seed, k)
            for rho in (0.0, 1.0):
                X, y = run_v2.dataset(sets, rho, SNR)
                D = X[y == 1] - B[N_NORMAL:] - noise[N_NORMAL:]
                p = deviation(Z, C, u, rho)
                other = np.delete(D, S, axis=1)
                if rho == 1.0:
                    X0, _ = run_v2.dataset(sets, 0.0, SNR); A0, A1 = X0[y == 1], X[y == 1]
                    dC = np.cov(A1.T) - np.cov(A0.T)
                    off = np.setdiff1d(np.arange(dC.shape[0]), S)
                    bvar = B[N_NORMAL:][:, S].var(0) / (SNR * SIGMA) ** 2
                    res[f"{system}|{seed}|featcov"] = dict(
                        max_abs_dC_outside_S=float(np.abs(dC[np.ix_(off, off)]).max()),
                        background_var_over_t2_median=float(np.median(bvar)),
                        background_var_over_t2_range=[float(bvar.min()), float(bvar.max())])
                res[f"{system}|{seed}|{rho}"] = dict(
                    same_channels=bool(np.array_equal(S, S2) and np.array_equal(u, u2)),
                    max_abs_diff_on_S=float(np.abs(D[:, S] - p).max()),
                    max_abs_off_S=float(np.abs(other).max()))
    return res


if __name__ == "__main__":
    main()
