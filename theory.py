"""Ideal-observer detectability for a zero-mean deviation with matched marginals.
Background: x ~ N(0, s^2 I_k). Deviation adds t^2 * R_rho, R_rho = (1-rho) I + rho u u^T (u = +-1).
Exact log-likelihood-ratio statistic; AUC by Monte Carlo. Participation ratio PR = (tr R)^2 / tr(R^2)."""
import numpy as np, json
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
from sklearn.metrics import roc_auc_score
def pr(k,rho): ev=np.r_[(1-rho)+rho*k, np.full(k-1,1-rho)]; return ev.sum()**2/(ev**2).sum()
def auc(k,rho,snr,n=40000,seed=0):
    r=np.random.default_rng(seed); u=np.ones(k)
    S0=np.eye(k); S1=np.eye(k)+snr**2*((1-rho)*np.eye(k)+rho*np.outer(u,u))
    L0=np.linalg.cholesky(S0); L1=np.linalg.cholesky(S1); D=np.linalg.inv(S0)-np.linalg.inv(S1)
    x0=r.normal(size=(n,k))@L0.T; x1=r.normal(size=(n,k))@L1.T
    s=lambda x: np.einsum('ij,jk,ik->i',x,D,x)
    return roc_auc_score(np.r_[np.zeros(n),np.ones(n)], np.r_[s(x0),s(x1)])
out={}
for k in [3,5,10,20,47]:
  for snr in [0.5,1.0,2.0,4.0]:
    for rho in [0,0.25,0.5,0.75,1.0]:
      out[f"{k}_{snr}_{rho}"]=dict(k=k,snr=snr,rho=rho,pr=pr(k,rho),auc=auc(k,rho,snr))
json.dump(out,open(os.path.join(ROOT,'theory.json'),'w'))
for snr in [0.5,1.0,2.0,4.0]:
  print('k=10 snr',snr,[ (rho, round(out[f"10_{snr}_{rho}"]['pr'],2), round(out[f"10_{snr}_{rho}"]['auc'],4)) for rho in [0,0.25,0.5,0.75,1.0]])
