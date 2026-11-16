from plot_common import *
r=data(); fig,ax=plt.subplots(figsize=(5.4,3.1))
means=[r['summary'][m]['total_s']['mean'] for m in METHODS]
std=[r['summary'][m]['total_s']['std'] for m in METHODS]
ax.bar(np.arange(3),means,yerr=std,capsize=4,color=COLORS,width=.6,alpha=.85)
for i,m in enumerate(METHODS):
    values=r['summary'][m]['total_s']['values']
    ax.scatter(i+np.array([-.11,0,.11]),values,s=20,color='black',zorder=3)
    ax.text(i,means[i]+std[i]+1.5,f'{means[i]:.2f} s',ha='center',fontsize=9)
ax.set_xticks(np.arange(3),LABELS)
ax.set_ylabel('Complete extraction time (s)')
ax.set_ylim(0,max(means)*1.23)
ax.set_title('Case-1: 146,704 cells, four sources, three paired seeds',pad=12)
save(fig,'fig_01_runtime')
