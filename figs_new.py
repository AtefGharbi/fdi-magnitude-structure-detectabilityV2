import json, numpy as np, matplotlib
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
matplotlib.use("Agg"); import matplotlib.pyplot as plt

A = json.load(open(os.path.join(ROOT, "analysis.json"))); T = json.load(open(os.path.join(ROOT, "theory.json")))
NAVY, BLUE, GOLD, LGOLD, LNAVY, GREY = "#203864", "#2E75B6", "#BF9000", "#E6C766", "#8EA9DB", "#7F7F7F"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
RO = [("rf", "Random Forest"), ("pi", "Physics-informed"), ("dp", "Diagonal-precision"), ("fp", "Full-precision")]
RO_C = {"rf": NAVY, "pi": GOLD, "dp": GREY, "fp": BLUE}
OUT = os.path.join(ROOT, "figures") + os.sep; os.makedirs(OUT, exist_ok=True)

# ---------- Figure 6: corrected 2x2 ----------
fig, axs = plt.subplots(1, 2, figsize=(10, 3.9), dpi=300, sharey=True)
conds = [("rho0.0_snr2.0", "Independent 2σ", LGOLD), ("rho1.0_snr2.0", "Correlated 2σ", LNAVY),
         ("rho0.0_snr4.0", "Independent 4σ", GOLD), ("rho1.0_snr4.0", "Correlated 4σ", NAVY)]
for ax, s in zip(axs, ["14-bus", "118-bus"]):
    x = np.arange(4); w = 0.2
    for j, (c, lab, col) in enumerate(conds):
        m = [A["means"][s][f"{r}_auc"][c][0] for r, _ in RO]; sd = [A["means"][s][f"{r}_auc"][c][1] for r, _ in RO]
        ax.bar(x + (j - 1.5) * w, m, w, yerr=sd, capsize=2, color=col, label=lab, edgecolor="white", linewidth=0.4, error_kw=dict(lw=0.8))
    ax.axhline(0.5, ls=":", color="k", lw=0.8)
    k = 10 if s == "14-bus" else 47
    ax.set_title(f"IEEE {s} (k = {k}, {A['main'][s]['n_seeds']} seeds)", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["Random\nForest", "Physics-\ninformed", "Diagonal-\nprecision", "Full-\nprecision"])
    ax.set_ylim(0.4, 1.05); ax.yaxis.grid(True, ls="--", alpha=0.4); ax.set_axisbelow(True)
axs[0].set_ylabel("ROC-AUC vs. normal (mean ± SD)")
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=4, frameon=False)
plt.tight_layout(rect=(0, 0, 1, 0.92)); plt.savefig(OUT + "fig4_corrected_factorial.png", dpi=300); plt.close()

# ---------- Figure 7: correlation sweep + theory ----------
fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), dpi=300)
rhos = [0, 0.25, 0.5, 0.75, 1.0]
ax = axs[0]
for snr, col in zip(["0.5", "1.0", "2.0", "4.0"], [LGOLD, GOLD, BLUE, NAVY]):
    au = [T[f"10_{snr}_{r}"]["auc"] for r in [0, 0.25, 0.5, 0.75, 1.0]]
    ax.plot(rhos, np.array(au) - au[0], "o-", color=col, ms=4, label=f"{float(snr):g}σ")
ax.axhline(0, color="k", lw=0.6)
ax.set_title("(a) Ideal observer, isotropic noise (k = 10)", fontsize=10)
ax.set_xlabel("Cross-channel correlation ρ"); ax.set_ylabel("ΔAUC relative to ρ = 0")
ax.legend(title="per-channel SNR", frameon=False, fontsize=8, title_fontsize=8)
pr = [T[f"10_4.0_{r}"]["pr"] for r in [0, 0.25, 0.5, 0.75, 1.0]]
for a, s in zip(axs[1:], ["14-bus", "118-bus"]):
    for r, n in RO:
        v = A["rho"][s][f"{r}_auc"]
        a.errorbar(rhos, v["means"], yerr=v["sds"], fmt="o-", color=RO_C[r], ms=4, capsize=2, lw=1.2, label=n)
    a.set_title(f"({'b' if s == '14-bus' else 'c'}) IEEE {s}, 4σ", fontsize=10)
    a.set_xlabel("Cross-channel correlation ρ"); a.set_ylabel("ROC-AUC"); a.set_ylim(0.4, 1.02)
    a.yaxis.grid(True, ls="--", alpha=0.4)
    sec = a.secondary_xaxis("top"); sec.set_xticks(rhos)
    kk = 10 if s == "14-bus" else 47
    ev = lambda r: np.r_[(1 - r) + r * kk, np.full(kk - 1, 1 - r)]
    sec.set_xticklabels([f"{(ev(r).sum() ** 2 / (ev(r) ** 2).sum()):.1f}" for r in rhos], fontsize=7)
    sec.set_xlabel("participation ratio", fontsize=8)
h, l = axs[2].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9)
plt.tight_layout(rect=(0, 0.07, 1, 1)); plt.savefig(OUT + "fig5_correlation_sweep.png", dpi=300); plt.close()

# ---------- Figure 8: k sensitivity ----------
if "ksweep" in A and any(len(v) > 1 for v in A["ksweep"].values()):
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), dpi=300)
    labs = dict(i2="Independent 2σ", c2="Correlated 2σ", i4="Independent 4σ", c4="Correlated 4σ")
    cols = dict(i2=LGOLD, c2=LNAVY, i4=GOLD, c4=NAVY)
    nfeat = {"14-bus": 58, "118-bus": 582}
    for a, s, tag in zip(axs[:2], ["14-bus", "118-bus"], "ab"):
        ks = sorted(int(k) for k in A["ksweep"][s] if k != "n_seeds")
        for c in ["i2", "c2", "i4", "c4"]:
            a.errorbar(ks, [A["ksweep"][s][str(k)]["rf_auc"]["cond_means"][c] for k in ks],
                       yerr=[A["ksweep"][s][str(k)]["rf_auc"]["cond_sds"][c] for k in ks], fmt="o-", ms=4, capsize=2, color=cols[c], label=labs[c])
        a.set_xlabel("affected channels k"); a.set_ylabel("Random Forest ROC-AUC"); a.set_ylim(0.4, 1.02)
        a.set_title(f"({tag}) IEEE {s}", fontsize=10); a.yaxis.grid(True, ls="--", alpha=0.4)
    axs[0].legend(frameon=False, fontsize=8)
    a = axs[2]
    for s, mk in [("14-bus", "o"), ("118-bus", "s")]:
        ks = sorted(int(k) for k in A["ksweep"][s] if k != "n_seeds")
        for r, ls in [("rf", "-"), ("pi", "--")]:
            st = [A["ksweep"][s][str(k)][f"{r}_auc"]["structure"] for k in ks]
            xx = [100 * k / nfeat[s] for k in ks]
            a.errorbar(xx, [q["mean"] for q in st], yerr=[[q["mean"] - q["ci"][0] for q in st], [q["ci"][1] - q["mean"] for q in st]],
                       fmt=mk + ls, ms=4, capsize=2, color=RO_C[r], label=f"{'RF' if r == 'rf' else 'PI'}, {s}")
    a.axhline(0, color="k", lw=0.6); a.set_xscale("log"); a.set_xticks([2, 5, 10, 20, 50]); a.set_xticklabels(["2", "5", "10", "20", "50"]); a.minorticks_off()
    a.set_xlabel("affected channels (% of features)"); a.set_ylabel("Structure effect ΔAUC (corr − indep)")
    a.set_title("(c) Structure effect vs. channel fraction", fontsize=10); a.legend(frameon=False, fontsize=7)
    plt.tight_layout(); plt.savefig(OUT + "fig6_k_sensitivity.png", dpi=300); plt.close()

# ---------- Figure 9: PI diagnostics ----------
K = json.load(open(os.path.join(ROOT, "kdump.json")))
D = A.get("diag", {})
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), dpi=300, gridspec_kw=dict(width_ratios=[1, 1.5]))
a = axs[0]
for s, col, mk in [("118-bus", GOLD, "s"), ("14-bus", NAVY, "o")]:
    a.scatter(np.abs(K[s]["k_ls"]), np.abs(K[s]["k_learn"]), s=10, color=col, marker=mk, label=f"IEEE {s}", alpha=0.8)
lim = [0.5, 3e5]; a.plot(lim, lim, "k:", lw=0.8); a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lim); a.set_ylim(lim)
a.axhline(5, color=GREY, lw=0.6, ls="--"); a.text(1, 7, "initial value k = 5", fontsize=7, color=GREY)
a.set_xlabel("|k| physical (least-squares, normal data)"); a.set_ylabel("|k| learned by reference detector")
a.set_title("(a) Per-line susceptance after training (seed 0)", fontsize=10); a.legend(frameon=False, fontsize=8)
a = axs[1]
V = [("rf", "RF"), ("pi", "PI\nreference"), ("mlp_nores", "No\nresidual"), ("pi_lsinit", "LS\ninit"),
     ("pi_cal", "LS init +\nper-line"), ("pi_cal_big", "Calibrated\n2× size"), ("rf_plus_res", "RF +\nresidual")]
sys_avail = [s for s in ["14-bus", "118-bus"] if s in D]
x = np.arange(len(V)); w = 0.2; j = 0
for s in sys_avail:
    for ci, cname, col in [(1, "sparse", GOLD if s == "118-bus" else LGOLD), (2, "dense", NAVY if s == "118-bus" else LNAVY)]:
        m = [D[s][v]["mean"][ci] for v, _ in V]; sd = [D[s][v]["sd"][ci] for v, _ in V]
        a.bar(x + (j - (2 * len(sys_avail) - 1) / 2) * w, m, w, yerr=sd, capsize=2, color=col, label=f"{s}, {cname}", edgecolor="white", linewidth=0.4, error_kw=dict(lw=0.7))
        j += 1
a.set_xticks(x); a.set_xticklabels([n for _, n in V], fontsize=7.5); a.set_ylabel("F1 (mean ± SD, 8 seeds)"); a.set_ylim(0, 1.08)
a.set_title("(b) Detector variants on the motivating-example data", fontsize=10); a.yaxis.grid(True, ls="--", alpha=0.4)
a.legend(frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.28))
plt.tight_layout(); plt.savefig(OUT + "fig7_pi_diagnostics.png", dpi=300); plt.close()
print("figures written")
