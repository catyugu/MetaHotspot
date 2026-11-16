from plot_common import *
r=data();fig,axes=plt.subplots(1,2,figsize=(7.2,3.0))
validation=r['validation']['runs']
for i,(method,color) in enumerate(zip(METHODS,COLORS)):
    rows=[x for x in validation if x['method']==method]
    values=np.array([[p['steady_field_relative'] for p in row['points']] for row in rows])
    x=np.arange(13)+(i-1)*.19
    axes[0].vlines(x,values.min(axis=0),values.max(axis=0),color=color,alpha=.65)
    axes[0].scatter(x,values.mean(axis=0),s=12,color=color,label=LABELS[i].replace('\n',' '))
    values=np.array([[p['transient_field_relative'] for p in row['points'] if 'transient_field_relative' in p] for row in rows])
    x=np.arange(3)+(i-1)*.15
    axes[1].vlines(x,values.min(axis=0),values.max(axis=0),color=color)
    axes[1].scatter(x,values.mean(axis=0),s=18,color=color)
axes[0].set_xticks(np.arange(13),[f'C{i}' for i in range(1,9)]+['N']+[f'R{i}' for i in range(1,5)],fontsize=7.5)
axes[1].set_xticks([0,1,2],['All HTC=1','All HTC=10⁴','Nominal'])
for ax,label in zip(axes,['(a) Steady full-field error','(b) Transient full-field error']):
    ax.set_title(label);ax.set_yscale('log');ax.set_ylabel('Max error / max steady FOM rise')
axes[0].set_xlabel('Corners C1–C8, nominal N, random R1–R4')
axes[1].set_xlabel('Maximum over 50 BDF1 time steps')
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.1),ncol=3,frameon=False)
save(fig,'fig_03_accuracy')
