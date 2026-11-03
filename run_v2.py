"""Corrected controlled experiment (revision).

Per (system, seed, k): one fixed set S of k affected channels and one fixed
sign pattern u in {-1,+1}^k (shared by every condition). Deviation on S:
    p = t * ( sqrt(1-rho) * z + sqrt(rho) * c * u ),  z ~ N(0, I_k), c ~ N(0,1), t = snr*sigma
so every affected channel has marginal N(0, t^2) in every condition and the
pairwise correlation between affected channels is rho*u_i*u_j. Only rho
(cross-channel covariance) differs between structure conditions; marginal
magnitude, channel identity, channel count and expected energy k*t^2 are
identical. Common random numbers: the same operating points, z, c and sensor
noise are reused across conditions within (system, seed, k).
Each condition is evaluated as its own binary task (normal vs. condition).
"""
import os, json, time
import numpy as np
import multiprocessing as mp

RES = "/home/claude/exp/results_v2"; os.makedirs(RES, exist_ok=True)
ORIG = [0, 1, 2, 3, 4, 42, 123, 999]; NEW = list(range(5, 17))
MAIN_K = {"14-bus": 10, "118-bus": 47}
SWEEP_K = {"14-bus": [3, 5, 20, 29], "118-bus": [10, 20, 30, 70, 99]}
N_NORMAL, N_ATT, SIGMA = 1200, 300, 0.01


def make_sets(system, seed, k):
    from common import clean_bank
    B = clean_bank(system, seed, N_NORMAL + N_ATT)
    nf = B.shape[1]
    r = np.random.default_rng([seed, k, 1])
    S = r.choice(nf, k, replace=False); u = r.choice([-1.0, 1.0], k)
    r2 = np.random.default_rng([seed, k, 2])
    noise = r2.normal(0, SIGMA, size=B.shape)
    Z = r2.normal(size=(N_ATT, k)); C = r2.normal(size=N_ATT)
    return B, S, u, noise, Z, C


def dataset(sets, rho, snr):
    B, S, u, noise, Z, C = sets
    Xn = B[:N_NORMAL] + noise[:N_NORMAL]
    A = B[N_NORMAL:].copy()
    A[:, S] += snr * SIGMA * (np.sqrt(1 - rho) * Z + np.sqrt(rho) * C[:, None] * u[None, :])
    A += noise[N_NORMAL:]
    return np.vstack([Xn, A]), np.r_[np.zeros(N_NORMAL, int), np.ones(N_ATT, int)]


def evaluate(system, seed, k, conds, full=True):
    from common import topology
    from detectors import rf_bin, diag_precision_bin, full_precision_bin, train_pi
    topo = topology(system); sets = make_sets(system, seed, k); out = {}
    for rho, snr in conds:
        X, y = dataset(sets, rho, snr)
        d = dict(rf=rf_bin(X, y, seed), pi=train_pi(X, y, topo, seed, return_diag="auc"))
        if full:
            d.update(dp=diag_precision_bin(X, y, seed), fp=full_precision_bin(X, y, seed))
        out[f"rho{rho}_snr{snr}"] = d
    return out


def run(job):
    kind, system, seed = job
    path = f"{RES}/{kind}_{system}_s{seed}.json"
    if os.path.exists(path): return path
    t = time.time(); R = {"system": system, "seed": seed}
    base = [(0.0, 2.0), (0.0, 4.0), (1.0, 2.0), (1.0, 4.0)]
    if kind == "main":
        R["k"] = MAIN_K[system]
        R["res"] = evaluate(system, seed, MAIN_K[system], base + [(0.25, 4.0), (0.5, 4.0), (0.75, 4.0), (0.0, 0.5), (1.0, 0.5), (0.0, 1.0), (1.0, 1.0)])
    elif kind == "ksweep":
        R["by_k"] = {k: evaluate(system, seed, k, base, full=False) for k in SWEEP_K[system]}
    elif kind == "diag":
        import run_all
        return run_all.run(("diag", system, seed))
    R["sec"] = time.time() - t
    json.dump(R, open(path, "w")); return path


def jobs():
    J = [("main", s_, sd) for sd in ORIG for s_ in ("14-bus", "118-bus")]
    J += [("diag", "118-bus", sd) for sd in ORIG]
    J += [("ksweep", s_, sd) for sd in ORIG for s_ in ("118-bus", "14-bus")]
    J += [("main", s_, sd) for sd in NEW for s_ in ("14-bus", "118-bus")]
    J += [("diag", "14-bus", sd) for sd in ORIG]
    return J


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:  # quick single-job test
        print(run((sys.argv[1], sys.argv[2], int(sys.argv[3])))); raise SystemExit
    with mp.Pool(2, maxtasksperchild=3) as pool:
        for o in pool.imap_unordered(run, jobs()):
            print(time.strftime("%H:%M:%S"), o, flush=True)
    print("ALL DONE", flush=True)
