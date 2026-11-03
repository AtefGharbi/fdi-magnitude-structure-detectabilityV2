"""
Shared code for the revision experiments.

Data generation and detector code are copied from the public repository
(fdi-magnitude-structure-detectabilityV1) with ONE change: the pandapower
network is built once per process and its loads are rescaled in place,
instead of re-creating the case for every sample. The random-number stream is
consumed in exactly the same order as in the original scripts, so the same
seed yields the same dataset (power-flow differences < 1e-6, vs. noise 1e-2).
"""
import os, warnings, logging
import numpy as np
import pandapower as pp
import pandapower.networks as pn

warnings.filterwarnings("ignore")
logging.getLogger("pandapower").setLevel(logging.ERROR)

DATA = "/home/claude/exp/data"
os.makedirs(DATA, exist_ok=True)
CASES = {"14-bus": pn.case14, "118-bus": pn.case118}
_NET = {}


def _net(system):
    if system not in _NET:
        net = CASES[system]()
        _NET[system] = (net, net.load["p_mw"].values.copy())
    return _NET[system]


def get_measurements(net):
    return np.concatenate([net.res_bus.vm_pu.values, net.res_bus.va_degree.values,
                           net.res_line.p_from_mw.values, net.res_line.q_from_mvar.values])


def solve(system, rng, load_variation=0.15):
    net, p0 = _net(system)
    fac = rng.uniform(1 - load_variation, 1 + load_variation, len(p0))
    net.load["p_mw"] = p0 * fac
    try:
        pp.runpp(net, numba=True)
    except Exception:
        return None
    return get_measurements(net)


def add_sensor_noise(m, noise_std, rng):
    return m + rng.normal(0, noise_std, size=m.shape)


# ---------------- original H1 constructions ----------------
def inject_independent_deviation(m, rng, noise_std=0.01, attack_ratio=0.08, snr_factor=4.0):
    n_aff = max(1, int(len(m) * attack_ratio))
    idx = rng.choice(len(m), n_aff, replace=False)
    d = m.copy(); d[idx] += rng.normal(0, noise_std * snr_factor, size=n_aff)
    return d


def inject_correlated_deviation(m, n_bus, rng, noise_std=0.01, snr_factor=2.0):
    c = rng.uniform(-1, 1, size=n_bus)
    proj = rng.normal(size=(len(m), n_bus)) / np.sqrt(n_bus)
    v = proj @ c
    v = v / (np.std(v) + 1e-8) * (noise_std * snr_factor)
    return m + v


def generate_h1(system, seed, n_normal=3000, n_ind=1000, n_cor=1000, noise_std=0.01):
    path = f"{DATA}/h1_{system}_s{seed}.npz"
    if os.path.exists(path):
        z = np.load(path); return z["X"], z["y"]
    rng = np.random.default_rng(seed)
    n_bus = len(_net(system)[0].bus)
    S, L = [], []
    for _ in range(n_normal):
        m = solve(system, rng)
        if m is None: continue
        S.append(add_sensor_noise(m, noise_std, rng)); L.append(0)
    for _ in range(n_ind):
        m = solve(system, rng)
        if m is None: continue
        m = inject_independent_deviation(m, rng, noise_std)
        S.append(add_sensor_noise(m, noise_std, rng)); L.append(1)
    for _ in range(n_cor):
        m = solve(system, rng)
        if m is None: continue
        m = inject_correlated_deviation(m, n_bus, rng, noise_std)
        S.append(add_sensor_noise(m, noise_std, rng)); L.append(2)
    X = np.array(S, dtype=np.float32); y = np.array(L)
    np.savez(path, X=X, y=y)
    return X, y


# ---------------- factorial construction ----------------
def inject_factorial_deviation(m, rng, structure, snr_factor, k, noise_std=0.01):
    idx = rng.choice(len(m), k, replace=False)
    target = noise_std * snr_factor
    if structure == "independent":
        p = rng.normal(0, 1.0, size=k)
    else:
        d = rng.normal(size=k); d = d / (np.linalg.norm(d) + 1e-12)
        p = d * rng.normal()
    p = p / (np.std(p) + 1e-12) * target
    out = m.copy(); out[idx] += p
    return out


CONDS = [(1, "independent", 2.0), (2, "independent", 4.0),
         (3, "correlated", 2.0), (4, "correlated", 4.0)]


def generate_factorial(system, seed, k, n_normal, n_per, noise_std=0.01):
    """Exact replica of factorial_experiment.py generator (RNG order preserved)."""
    path = f"{DATA}/fac_{system}_k{k}_n{n_normal}_s{seed}.npz"
    if os.path.exists(path):
        z = np.load(path); return z["X"], z["y"]
    rng = np.random.default_rng(seed)
    S, L = [], []
    for _ in range(n_normal):
        m = solve(system, rng)
        if m is None: continue
        S.append(add_sensor_noise(m, noise_std, rng)); L.append(0)
    for lab, st, snr in CONDS:
        for _ in range(n_per):
            m = solve(system, rng)
            if m is None: continue
            m = inject_factorial_deviation(m, rng, st, snr, k, noise_std)
            S.append(add_sensor_noise(m, noise_std, rng)); L.append(lab)
    X = np.array(S); y = np.array(L)
    np.savez(path, X=X, y=y)
    return X, y


def clean_bank(system, seed, n):
    """Noise-free operating points for the k-sensitivity sweep (shared across k)."""
    path = f"{DATA}/bank_{system}_n{n}_s{seed}.npy"
    if os.path.exists(path):
        return np.load(path)
    rng = np.random.default_rng(seed)
    B = []
    while len(B) < n:
        m = solve(system, rng)
        if m is not None: B.append(m)
    B = np.array(B); np.save(path, B)
    return B


def generate_sweep(system, seed, k, n_normal, n_per, noise_std=0.01):
    """Same operating points for every k; injection/noise RNG depends on (seed, k)."""
    B = clean_bank(system, seed, n_normal + 4 * n_per)
    rng = np.random.default_rng([seed, k, 2026])
    X, y, i = [], [], 0
    for _ in range(n_normal):
        X.append(add_sensor_noise(B[i], noise_std, rng)); y.append(0); i += 1
    for lab, st, snr in CONDS:
        for _ in range(n_per):
            m = inject_factorial_deviation(B[i], rng, st, snr, k, noise_std)
            X.append(add_sensor_noise(m, noise_std, rng)); y.append(lab); i += 1
    return np.array(X), np.array(y)


def topology(system):
    net = CASES[system]()
    return (net.line["from_bus"].values.astype(int), net.line["to_bus"].values.astype(int),
            net.bus.shape[0], net.line.shape[0])
