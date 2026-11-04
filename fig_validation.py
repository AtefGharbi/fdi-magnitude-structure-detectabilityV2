"""Supplementary Figure S4: empirical validation of the correlation manipulation.
Reads validation.json (validate_manipulation.py) and redraws the representative matrices from the seed draws."""
import json, numpy as np, matplotlib
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from validate_manipulation import draws, deviation, theory, MAIN_K, SEEDS, RHOS, SWEEP_K

D = json.load(open(os.path.join(ROOT, "validation.json")))
NAVY, BLUE, GOLD, GREY = "#203864", "#2E75B6", "#BF9000", "#7F7F7F"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
SC = {"14-bus": NAVY, "118-bus": GOLD}

fig = plt.figure(figsize=(12, 7.2), dpi=300)
TOP = [[0.04 + i * 0.165, 0.56, 0.15, 0.36] for i in range(3)]
BOT = [[0.06, 0.08, 0.2, 0.36], [0.33, 0.08, 0.2, 0.36], [0.61, 0.08, 0.36, 0.36]]

# (a-c) correlation matrices, 14-bus, seed 0, k = 10, the 300 deviation samples used in the experiment
S, u, nz, Z, C = draws("14-bus", 0, 10)
order = np.argsort(-u, kind="stable")
for i, rho in enumerate([0.0, 0.5, 1.0]):
    ax = fig.add_axes(TOP[i])
    R = np.corrcoef(deviation(Z, C, u, rho).T)[np.ix_(order, order)]
    im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(f"({'abc'[i]}) 14-bus, ρ = {rho:g}", fontweight="bold", loc="left")
    ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
    npos = int((u > 0).sum()); ax.axhline(npos - 0.5, color="k", lw=0.5); ax.axvline(npos - 0.5, color="k", lw=0.5)
    ax.set_xlabel("channels sorted by uᵢ", fontsize=8)
cax = fig.add_axes([0.54, 0.6, 0.008, 0.28]); fig.colorbar(im, cax=cax).set_label("Pearson r", fontsize=8)

# (d) mean signed off-diagonal correlation vs rho
ax = fig.add_axes([0.66, 0.58, 0.31, 0.34])
for s in MAIN_K:
    m = [np.mean([r["dev"]["r_signed_mean"] for r in D["main"][f"{s}|{rho}"]]) for rho in RHOS]
    mn = [np.mean([r["dev_noise"]["r_signed_mean"] for r in D["main"][f"{s}|{rho}"]]) for rho in RHOS]
    ax.plot(RHOS, m, "o", color=SC[s], ms=5, label=f"{s}: deviation")
    ax.plot(RHOS, mn, "s", mfc="white", color=SC[s], ms=4.5, label=f"{s}: + sensor noise")
x = np.linspace(0, 1, 50); ax.plot(x, x, "-", color=GREY, lw=1, label="theory ρ")
ax.plot(x, x * 16 / 17, "--", color=GREY, lw=1, label="theory ρ·SNR²/(SNR²+1)")
ax.set_xlabel("nominal ρ"); ax.set_ylabel("mean signed off-diag. r")
ax.set_title("(d) Off-diagonal correlation", fontweight="bold", loc="left"); ax.legend(fontsize=6, frameon=False, loc="upper left")

# (e,f) eigenvalues vs rho
for j, s in enumerate(MAIN_K):
    ax = fig.add_axes(BOT[j]); k = MAIN_K[s]
    l1 = np.array([[r["dev"]["lam1"] for r in D["main"][f"{s}|{rho}"]] for rho in RHOS])
    lr = np.array([[r["dev"]["lam_rest"] for r in D["main"][f"{s}|{rho}"]] for rho in RHOS])
    ax.errorbar(RHOS, l1.mean(1), l1.std(1), fmt="o", color=SC[s], ms=5, capsize=2, label="λ₁ (empirical)")
    ax.errorbar(RHOS, lr.mean(1), lr.std(1), fmt="s", mfc="white", color=SC[s], ms=4.5, capsize=2, label="mean λ₂…λₖ (empirical)")
    ax.plot(x, 1 - x + k * x, "-", color=GREY, lw=1, label="theory 1−ρ+kρ")
    ax.plot(x, 1 - x, "--", color=GREY, lw=1, label="theory 1−ρ")
    ax.set_yscale("symlog", linthresh=1); ax.set_xlabel("nominal ρ"); ax.set_ylabel("covariance eigenvalue / t²")
    ax.set_title(f"({'ef'[j]}) {s}, k = {k}", fontweight="bold", loc="left")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")

# (g) PR empirical vs theoretical, all seeds, main k and channel-count sweep
ax = fig.add_axes(BOT[2])
for s in MAIN_K:
    k = MAIN_K[s]
    th = [theory(k, rho)[2] for rho in RHOS]
    em = [[r["dev"]["pr"] for r in D["main"][f"{s}|{rho}"]] for rho in RHOS]
    for t_, e_ in zip(th, em):
        ax.plot([t_] * len(e_), e_, "o", color=SC[s], ms=3, alpha=0.5)
    ax.plot([], [], "o", color=SC[s], ms=4, label=f"{s}, k = {k}, 20 seeds × 5 ρ")
    sw = [(v["pr_theory"], v["pr"]) for kk, v in D["sweep"].items() if kk.startswith(s) and kk.endswith("|0.0")]
    sw = np.array(sw); ax.plot(sw[:, 0], sw[:, 1], "^", color=SC[s], mfc="white", ms=5, label=f"{s}, k-sweep, ρ = 0 (8-seed mean)")
x = np.logspace(0, 2, 100); ax.plot(x, x, "-", color=GREY, lw=1, label="identity")
ax.plot(x, 1 / (1 / x + 1 / 300), "--", color=GREY, lw=1, label="finite-sample expectation, n = 300")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("theoretical participation ratio"); ax.set_ylabel("empirical participation ratio")
ax.set_title("(g) Participation ratio", fontweight="bold", loc="left")
ax.legend(fontsize=6.5, frameon=False, loc="upper left")
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True); plt.savefig(os.path.join(ROOT, "figures", "figS4_validation.png"), dpi=300, bbox_inches="tight"); print("ok")
