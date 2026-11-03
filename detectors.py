"""Random Forest and physics-informed (PI) detectors.

`train_pi` with default arguments is the published V2 detector
(physics_informed_detector.py), generalised to any number of classes.
Optional arguments implement the diagnostic variants used for Revision 2.
"""
import numpy as np
import torch, torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def rf_bin(X, y, seed):
    Xtr, Xte, ytr, yte = split(X, y, seed)
    clf = RandomForestClassifier(n_estimators=100, random_state=seed).fit(Xtr, ytr)
    return (float(f1_score(yte, clf.predict(Xte))), float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])))


def _f1_thresh(s_tr, ytr, s_te, yte):
    """Threshold chosen on the TRAINING split to maximise F1, then applied to test."""
    qs = np.quantile(s_tr, np.linspace(0.5, 0.995, 200))
    best = max(qs, key=lambda q: f1_score(ytr, (s_tr > q).astype(int)))
    return float(f1_score(yte, (s_te > best).astype(int))), float(roc_auc_score(yte, s_te))


def diag_precision_bin(X, y, seed):
    """Diagonal-precision readout: s = sum_i pi_i (x_i - mu_i)^2, pi_i = 1/var_i from normal training data."""
    Xtr, Xte, ytr, yte = split(X, y, seed)
    N = Xtr[ytr == 0]; mu = N.mean(0); pi = 1.0 / (N.var(0) + 1e-12)
    s = lambda A: (((A - mu) ** 2) * pi).sum(1)
    return _f1_thresh(s(Xtr), ytr, s(Xte), yte)


def full_precision_bin(X, y, seed):
    """Full-precision (Gaussian likelihood-ratio) readout with Ledoit-Wolf covariances:
    s = log N(x; mu1, S1) - log N(x; mu0, S0); uses off-diagonal precision terms."""
    from sklearn.covariance import LedoitWolf
    Xtr, Xte, ytr, yte = split(X, y, seed)
    sc = StandardScaler().fit(Xtr[ytr == 0]); A, B = sc.transform(Xtr), sc.transform(Xte)
    keep = np.isfinite(A).all(0) & (A.std(0) > 0); A, B = A[:, keep], B[:, keep]
    def logpdf_fn(D):
        lw = LedoitWolf().fit(D); m = lw.location_; P = lw.precision_
        _, logdet = np.linalg.slogdet(lw.covariance_)
        return lambda Z: -0.5 * np.einsum("ij,jk,ik->i", Z - m, P, Z - m) - 0.5 * logdet
    l0, l1 = logpdf_fn(A[ytr == 0]), logpdf_fn(A[ytr == 1])
    s = lambda Z: l1(Z) - l0(Z)
    return _f1_thresh(s(A), ytr, s(B), yte)


def lr_bin(X, y, seed):
    Xtr, Xte, ytr, yte = split(X, y, seed)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(Xtr), ytr)
    Xs = sc.transform(Xte)
    return (float(f1_score(yte, clf.predict(Xs))), float(roc_auc_score(yte, clf.predict_proba(Xs)[:, 1])))

torch.set_num_threads(1)


def split(X, y, seed):
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


def per_class_f1(y_test, y_pred, n_classes):
    return [float(f1_score(y_test, y_pred, labels=[c], average=None)[0]) for c in range(n_classes)]


def rf_eval(X, y, seed, extra_train=None, extra_test=None):
    Xtr, Xte, ytr, yte = split(X, y, seed)
    if extra_train is not None:
        Xtr = np.hstack([Xtr, extra_train(Xtr, ytr)]); Xte = np.hstack([Xte, extra_test(Xte)])
    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf.fit(Xtr, ytr)
    return per_class_f1(yte, clf.predict(Xte), len(np.unique(y)))


def ls_susceptance(Xraw, n_bus, n_line, fi, ti):
    """Closed-form per-line least-squares k on normal-labelled samples (P = k * dtheta)."""
    ang = Xraw[:, n_bus:2 * n_bus] * np.pi / 180.0
    P = Xraw[:, 2 * n_bus:2 * n_bus + n_line]
    dth = ang[:, fi] - ang[:, ti]
    return (P * dth).sum(0) / ((dth ** 2).sum(0) + 1e-12)


def residual_np(Xraw, k, n_bus, n_line, fi, ti):
    ang = Xraw[:, n_bus:2 * n_bus] * np.pi / 180.0
    P = Xraw[:, 2 * n_bus:2 * n_bus + n_line]
    return P - k[None, :] * (ang[:, fi] - ang[:, ti])


class PINet(nn.Module):
    def __init__(self, input_dim, n_bus, n_line, n_classes, use_residual=True, width=1.0, k_init=None):
        super().__init__()
        self.n_bus, self.n_line, self.use_residual = n_bus, n_line, use_residual
        k0 = torch.full((n_line,), 5.0) if k_init is None else torch.tensor(k_init, dtype=torch.float32)
        self.k_vec = nn.Parameter(k0)
        comb = input_dim + (n_line if use_residual else 0)
        h1 = int(max(128, comb // 2) * width); h2 = int(max(64, comb // 4) * width)
        self.classifier = nn.Sequential(nn.Linear(comb, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(),
                                        nn.Linear(h2, n_classes))

    def residual(self, xr, fi, ti):
        ang = xr[:, self.n_bus:2 * self.n_bus] * (np.pi / 180.0)
        P = xr[:, 2 * self.n_bus:2 * self.n_bus + self.n_line]
        return P - self.k_vec.unsqueeze(0) * (ang[:, fi] - ang[:, ti])

    def forward(self, xs, xr, fi, ti, scale):
        r = self.residual(xr, fi, ti)
        if not self.use_residual:
            return self.classifier(xs), r
        return self.classifier(torch.cat([xs, r / scale], 1)), r


def train_pi(X, y, topo, seed, lambda_phy=0.3, epochs=120, batch_size=128, lr=1e-3,
             use_residual=True, width=1.0, k_init="const", per_line_scale=False, return_diag=False):
    fi, ti, n_bus, n_line = topo
    n_classes = len(np.unique(y))
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, Xte, ytr, yte = split(X, y, seed)
    sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    Xtr_r, Xte_r, Xtr_st, Xte_st = T(Xtr), T(Xte), T(Xtr_s), T(Xte_s)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    fi_t, ti_t = torch.tensor(fi), torch.tensor(ti)
    k_ls = ls_susceptance(Xtr[ytr == 0].astype(np.float64), n_bus, n_line, fi, ti)
    model = PINet(X.shape[1], n_bus, n_line, n_classes, use_residual, width,
                  k_init=k_ls if k_init == "ls" else None)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        r0 = model.residual(Xtr_r[ytr_t == 0], fi_t, ti_t)
        scale = (r0.std(0) + 1e-6) if per_line_scale else (torch.std(r0) + 1e-6)
    n = Xtr_st.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits, r = model(Xtr_st[idx], Xtr_r[idx], fi_t, ti_t, scale)
            loss = ce(logits, ytr_t[idx])
            nm = ytr_t[idx] == 0
            if nm.sum() > 0:
                loss = loss + lambda_phy * torch.mean((r[nm] / scale) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logits, _ = model(Xte_st, Xte_r, fi_t, ti_t, scale)
        pred = logits.argmax(1).numpy()
        prob = torch.softmax(logits, 1).numpy()
    f1 = per_class_f1(yte, pred, n_classes)
    if return_diag == "auc":
        return f1[1], float(roc_auc_score(yte, prob[:, 1]))
    if not return_diag:
        return f1
    k_learn = model.k_vec.detach().numpy().astype(np.float64)
    Xn = Xte[yte == 0].astype(np.float64)
    rms = lambda k: float(np.sqrt(np.mean(residual_np(Xn, k, n_bus, n_line, fi, ti) ** 2)))
    pflow_rms = float(np.sqrt(np.mean(Xn[:, 2 * n_bus:2 * n_bus + n_line] ** 2)))
    diag = dict(k_learn_median=float(np.median(k_learn)), k_ls_median=float(np.median(k_ls)),
                k_rel_err_median=float(np.median(np.abs(k_learn - k_ls) / np.abs(k_ls))),
                res_rms_learned=rms(k_learn), res_rms_ls=rms(k_ls), pflow_rms=pflow_rms)
    return f1, diag
