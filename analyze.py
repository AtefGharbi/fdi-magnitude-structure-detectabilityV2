import json, glob, numpy as np
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
from scipy import stats, integrate

RES = os.path.join(ROOT, "results_v2")
READOUTS = ["rf", "pi", "dp", "fp"]
SYS = ["14-bus", "118-bus"]


def bf10_jzs(d, r=np.sqrt(2) / 2):
    """Rouder et al. (2009) JZS Bayes factor for a one-sample / paired t-test."""
    d = np.asarray(d); n = len(d); t = d.mean() / (d.std(ddof=1) / np.sqrt(n)); nu = n - 1
    null = (1 + t ** 2 / nu) ** (-(nu + 1) / 2)
    f = lambda g: (1 + n * g * r ** 2) ** -0.5 * (1 + t ** 2 / ((1 + n * g * r ** 2) * nu)) ** (-(nu + 1) / 2) \
        * (2 * np.pi) ** -0.5 * g ** -1.5 * np.exp(-1 / (2 * g))
    return integrate.quad(f, 0, np.inf, limit=200)[0] / null


def tost(d, delta):
    d = np.asarray(d); n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    if se == 0: return 0.0
    p1 = 1 - stats.t.cdf((m + delta) / se, n - 1); p2 = stats.t.cdf((m - delta) / se, n - 1)
    return float(max(p1, p2))


def summ(d, delta=0.02):
    d = np.asarray(d, float); n = len(d); m = d.mean(); sd = d.std(ddof=1)
    ci = stats.t.interval(0.95, n - 1, loc=m, scale=sd / np.sqrt(n)) if sd > 0 else (m, m)
    try: p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError: p = 1.0
    return dict(n=n, mean=float(m), ci=[float(ci[0]), float(ci[1])], p=p,
                dz=float(m / sd) if sd > 0 else 0.0, bf10=float(bf10_jzs(d)) if sd > 0 else None,
                tost_p=tost(d, delta), n_neg=int((d < 0).sum()))


def load(kind, system):
    return [json.load(open(f)) for f in sorted(glob.glob(f"{RES}/{kind}_{system}_s*.json"))]


def val(res, cond, ro, metric):  # metric 0=F1, 1=AUC
    return res[cond][ro][metric]


def analysis():
    A = {"main": {}, "rho": {}, "lowsnr": {}, "ksweep": {}, "means": {}}
    for s in SYS:
        R = load("main", s); A["main"][s] = {}; A["means"][s] = {}; A["rho"][s] = {}; A["lowsnr"][s] = {}
        A["main"][s]["n_seeds"] = len(R)
        for ro in READOUTS:
            for mi, mname in [(1, "auc"), (0, "f1")]:
                g = lambda c: np.array([val(r["res"], c, ro, mi) for r in R])
                i2, i4, c2, c4 = g("rho0.0_snr2.0"), g("rho0.0_snr4.0"), g("rho1.0_snr2.0"), g("rho1.0_snr4.0")
                key = f"{ro}_{mname}"
                A["main"][s][key] = dict(structure=summ((c2 + c4) / 2 - (i2 + i4) / 2),
                                         magnitude=summ((i4 + c4) / 2 - (i2 + c2) / 2),
                                         interaction=summ((c4 - c2) - (i4 - i2)))
                A["means"][s][key] = {c: [float(g(c).mean()), float(g(c).std(ddof=1))] for c in R[0]["res"]}
                rhos = [0.0, 0.25, 0.5, 0.75, 1.0]
                M = np.array([g(f"rho{r}_snr4.0") for r in rhos]).T  # seeds x rho
                slopes = [np.polyfit(rhos, row, 1)[0] for row in M]
                rs = [stats.spearmanr(rhos, row)[0] if np.std(row) > 0 else 0 for row in M]
                A["rho"][s][key] = dict(slope=summ(slopes), spearman_mean=float(np.nanmean(rs)),
                                        means=M.mean(0).tolist(), sds=M.std(0, ddof=1).tolist())
                A["lowsnr"][s][key] = {f"snr{q}": summ(g(f"rho1.0_snr{q}") - g(f"rho0.0_snr{q}"))
                                       for q in ["0.5", "1.0"]}
                A["lowsnr"][s][key]["chance_check"] = {c: float(g(c).mean()) for c in
                                                       ["rho0.0_snr0.5", "rho1.0_snr0.5", "rho0.0_snr1.0", "rho1.0_snr1.0"]}
        K = load("ksweep", s); A["ksweep"][s] = {"n_seeds": len(K)}
        main_k = R[0]["k"]
        # use main-run seeds that also have sweep results for the main k
        sweep_seeds = {k["seed"] for k in K}
        Rm = {r["seed"]: r for r in R if r["seed"] in sweep_seeds}
        ks = sorted({int(k) for kk in K for k in kk["by_k"]} | {main_k})
        for k in ks:
            A["ksweep"][s][k] = {}
            for ro in ["rf", "pi"]:
                for mi, mname in [(1, "auc"), (0, "f1")]:
                    if k == main_k:
                        src = [Rm[sd]["res"] for sd in sorted(Rm)]
                    else:
                        src = [kk["by_k"][str(k)] for kk in K]
                    g = lambda c: np.array([x[c][ro][mi] for x in src])
                    i2, i4, c2, c4 = g("rho0.0_snr2.0"), g("rho0.0_snr4.0"), g("rho1.0_snr2.0"), g("rho1.0_snr4.0")
                    A["ksweep"][s][k][f"{ro}_{mname}"] = dict(
                        cond_means=dict(i2=float(i2.mean()), i4=float(i4.mean()), c2=float(c2.mean()), c4=float(c4.mean())),
                        cond_sds=dict(i2=float(i2.std(ddof=1)), i4=float(i4.std(ddof=1)), c2=float(c2.std(ddof=1)), c4=float(c4.std(ddof=1))),
                        structure=summ((c2 + c4) / 2 - (i2 + i4) / 2), magnitude=summ((i4 + c4) / 2 - (i2 + c2) / 2))
    return A


def diag_summary():
    out = {}
    for s in SYS:
        D = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ROOT, "results", f"diag_{s}_s*.json")))]
        if not D: continue
        out[s] = {"n": len(D)}
        for v in ["rf", "pi", "mlp_nores", "pi_lsinit", "pi_cal", "pi_cal_big", "rf_plus_res"]:
            M = np.array([d[v] for d in D])
            out[s][v] = dict(mean=M.mean(0).tolist(), sd=M.std(0, ddof=1).tolist())
        for key in D[0]["pi_diag"]:
            vals = np.array([d["pi_diag"][key] for d in D]); out[s]["diag_" + key] = [float(vals.mean()), float(vals.std(ddof=1))]
        # paired tests vs published PI on independent / correlated F1
        for v in ["mlp_nores", "pi_lsinit", "pi_cal", "pi_cal_big"]:
            for ci, cname in [(1, "ind"), (2, "cor")]:
                out[s][f"{v}_minus_pi_{cname}"] = summ([d[v][ci] - d["pi"][ci] for d in D])
    return out


if __name__ == "__main__":
    A = analysis(); A["diag"] = diag_summary()
    json.dump(A, open(os.path.join(ROOT, "analysis.json"), "w"), indent=1, default=float)
    for s in SYS:
        print("=====", s, "seeds", A["main"][s]["n_seeds"])
        for key, v in A["main"][s].items():
            if key == "n_seeds": continue
            st, mg, it = v["structure"], v["magnitude"], v["interaction"]
            print(f" {key:8s} STRUCT {st['mean']:+.3f} [{st['ci'][0]:+.3f},{st['ci'][1]:+.3f}] p={st['p']:.2g} dz={st['dz']:.2f} BF10={st['bf10']:.3g} | "
                  f"MAG {mg['mean']:+.3f} p={mg['p']:.2g} dz={mg['dz']:.2f} | INT {it['mean']:+.3f} p={it['p']:.2g}")
        for key, v in A["rho"][s].items():
            print(f" rho {key:8s} means {np.round(v['means'],3)} slope {v['slope']['mean']:+.3f} p={v['slope']['p']:.2g}")
        for key, v in A["lowsnr"][s].items():
            print(f" low {key:8s} " + " ".join(f"{q}: {v[q]['mean']:+.3f} TOSTp={v[q]['tost_p']:.2g} BF10={v[q]['bf10']}" for q in ["snr0.5", "snr1.0"]), v["chance_check"])
        for k, v in A["ksweep"][s].items():
            if k == "n_seeds": continue
            print(f" k={k:3}", " ".join(f"{m}: S{v[m]['structure']['mean']:+.3f}(p={v[m]['structure']['p']:.2g}) M{v[m]['magnitude']['mean']:+.3f} c={np.round(list(v[m]['cond_means'].values()),2)}" for m in ["rf_auc", "pi_auc"]))
    print(json.dumps(A["diag"], indent=0, default=float)[:3000])
