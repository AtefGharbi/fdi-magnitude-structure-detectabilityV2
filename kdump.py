import numpy as np, torch, json
import detectors as D
from common import generate_h1, topology
out={}
for s in ['14-bus','118-bus']:
    X,y=generate_h1(s,0); topo=topology(s); fi,ti,nb,nl=topo
    # replicate train_pi but capture model k
    cap={}
    orig=D.PINet.__init__
    def init(self,*a,**k): orig(self,*a,**k); cap['m']=self
    D.PINet.__init__=init
    f1=D.train_pi(X,y,topo,0); D.PINet.__init__=orig
    Xtr,Xte,ytr,yte=D.split(X,y,0)
    kls=D.ls_susceptance(Xtr[ytr==0].astype(float),nb,nl,fi,ti)
    out[s]=dict(k_learn=cap['m'].k_vec.detach().numpy().tolist(),k_ls=kls.tolist(),f1=f1)
    print(s,f1,np.median(kls))
json.dump(out,open('kdump.json','w'))
