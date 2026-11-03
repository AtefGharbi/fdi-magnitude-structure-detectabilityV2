import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R='/home/claude/repo_dl/repo/fdia_datasets/multiseed/'
NAVY,BLUE='#203864','#2E75B6'
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(7.2,4.4),dpi=300)
cols=['multiclass_f1_normal','multiclass_f1_simple','multiclass_f1_stealthy']
labs=['Normal','Sparse / high-amplitude\n(8% of channels, 4σ)','Dense / low-amplitude\n(all channels, 2σ)']
x=np.arange(3); w=0.36
for j,(s,c) in enumerate([('14-bus',NAVY),('118-bus',BLUE)]):
    d=pd.read_csv(R+f'{s}_perseed_results.csv'); m=d[cols].mean().values; sd=d[cols].std().values
    b=ax.bar(x+(j-0.5)*w,m,w,yerr=sd,capsize=4,color=c,label=f'IEEE {s}',edgecolor='white',linewidth=0.5)
    for xi,mi,si in zip(x+(j-0.5)*w,m,sd): ax.text(xi,mi+si+0.02,f'{mi:.3f}',ha='center',fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel('F1-score (mean ± SD, 8 seeds)'); ax.set_ylim(0,1.12)
ax.yaxis.grid(True,ls='--',alpha=0.4); ax.set_axisbelow(True)
ax.legend(loc='lower center',bbox_to_anchor=(0.5,1.01),ncol=2,frameon=False)
plt.tight_layout(); plt.savefig('/home/claude/figs/fig3_new.png',dpi=300)
