from plot_common import *
r=data();fig,ax=plt.subplots(figsize=(6.2,3.7))
phases=['spectrum','setup','cg','probes_basis_other','svd','projection']
names=['Global spectrum','AMG setup','PCG','Probes and basis','Snapshot SVD','BCI projection']
colors=['#999999','#D55E00','#0072B2','#56B4E9','#CC79A7','#009E73']
bottom=np.zeros(3)
for phase,label,color in zip(phases,names,colors):
    values=np.array([r['summary'][m]['phase_s'][phase]['mean'] for m in METHODS])
    ax.bar(np.arange(3),values,bottom=bottom,label=label,color=color,width=.6)
    bottom+=values
ax.set_xticks(np.arange(3),LABELS)
ax.set_ylabel('Mean extraction time (s)')
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.28),ncol=3,frameon=False,columnspacing=1.)
save(fig,'fig_02_phases')
