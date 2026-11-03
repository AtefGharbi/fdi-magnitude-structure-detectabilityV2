import numpy as np
from scipy import stats
from common import inject_factorial_deviation
rng=np.random.default_rng(1); k=10; N=50000
P={s:np.array([inject_factorial_deviation(np.zeros(k),rng,s,4.0,k) for _ in range(N)]) for s in ['independent','correlated']}
for name,f in [('L2 norm',lambda p:np.linalg.norm(p,axis=1)),('max |p|',lambda p:np.abs(p).max(1)),('mean',lambda p:p.mean(1)),('channel-0 value',lambda p:p[:,0])]:
    a,b=f(P['independent']),f(P['correlated']); print(f"{name:16s} indep {a.mean():.5f}±{a.std():.5f}  corr {b.mean():.5f}±{b.std():.5f}  KS p={stats.ks_2samp(a,b).pvalue:.3f}")
for s in P:
    C=np.corrcoef(P[s].T); off=C[~np.eye(k,dtype=bool)]
    print(s,'across-sample channel correlation: mean off-diag',round(off.mean(),4),' max |r|',round(np.abs(off).max(),4))
# can a classifier tell them apart?
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
X=np.vstack([np.sort(P['independent'][:5000],1),np.sort(P['correlated'][:5000],1)]); y=np.r_[np.zeros(5000),np.ones(5000)]
print('RF accuracy separating the two structure conditions (sorted values, noise-free):',cross_val_score(RandomForestClassifier(100,random_state=0),X,y,cv=5).mean().round(3))
