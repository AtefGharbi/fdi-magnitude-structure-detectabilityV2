import os, sys, json, time
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
import multiprocessing as mp

RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
ORIG = [0, 1, 2, 3, 4, 42, 123, 999]
NEW = list(range(5, 17))  # 12 additional seeds -> 20 total
SWEEP_K = {"14-bus": [3, 5, 10, 20, 29], "118-bus": [10, 20, 30, 47, 70, 99]}


def jobs():
    J = []
    for s in ORIG + NEW:
        J.append(("fac", "14-bus", s)); J.append(("fac", "118-bus", s))
    for s in ORIG:
        J.append(("sweep", "118-bus", s)); J.append(("sweep", "14-bus", s))
    for s in ORIG:
        J.append(("diag", "118-bus", s))
    for s in ORIG:
        J.append(("diag", "14-bus", s))
    return J


def run(job):
    kind, system, seed = job
    out = f"{RES}/{kind}_{system}_s{seed}.json"
    if os.path.exists(out):
        return out
    from common import generate_factorial, generate_sweep, generate_h1, topology
    from detectors import rf_eval, train_pi, ls_susceptance, residual_np
    import numpy as np
    topo = topology(system); t = time.time(); R = {"seed": seed, "system": system}
    if kind == "fac":
        k, nn_, npc = (10, 3000, 750) if system == "14-bus" else (47, 1200, 300)
        X, y = generate_factorial(system, seed, k, nn_, npc)
        R.update(k=k, rf=rf_eval(X, y, seed), pi=train_pi(X, y, topo, seed),
                 pi_cal=train_pi(X, y, topo, seed, k_init="ls", per_line_scale=True))
    elif kind == "sweep":
        R["by_k"] = {}
        for k in SWEEP_K[system]:
            X, y = generate_sweep(system, seed, k, 1200, 300)
            R["by_k"][k] = dict(rf=rf_eval(X, y, seed), pi=train_pi(X, y, topo, seed))
    elif kind == "diag":
        X, y = generate_h1(system, seed)
        fi, ti, nb, nl = topo
        R["rf"] = rf_eval(X, y, seed)
        f1, d = train_pi(X, y, topo, seed, return_diag=True); R["pi"] = f1; R["pi_diag"] = d
        R["mlp_nores"] = train_pi(X, y, topo, seed, use_residual=False)
        R["pi_lsinit"] = train_pi(X, y, topo, seed, k_init="ls")
        R["pi_cal"] = train_pi(X, y, topo, seed, k_init="ls", per_line_scale=True)
        R["pi_cal_big"] = train_pi(X, y, topo, seed, k_init="ls", per_line_scale=True, width=2.0, epochs=240)
        def feat_tr(Xtr, ytr):
            run.k = ls_susceptance(Xtr[ytr == 0].astype(float), nb, nl, fi, ti)
            return residual_np(Xtr.astype(float), run.k, nb, nl, fi, ti)
        feat_te = lambda Xte: residual_np(Xte.astype(float), run.k, nb, nl, fi, ti)
        R["rf_plus_res"] = rf_eval(X, y, seed, feat_tr, feat_te)
        R["rf_res_only"] = None
    R["sec"] = time.time() - t
    json.dump(R, open(out, "w"))
    return out


if __name__ == "__main__":
    with mp.Pool(2, maxtasksperchild=4) as pool:
        for o in pool.imap_unordered(run, jobs()):
            print(time.strftime("%H:%M:%S"), o, flush=True)
    print("ALL DONE", flush=True)
