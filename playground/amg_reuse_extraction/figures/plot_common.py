import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
HERE=Path(__file__).resolve().parent
METHODS=['fresh','port_cache','shift_cache']
LABELS=['Fresh AMG','HTC reuse\nwithin port','HTC + port\nreuse']
COLORS=['#0072B2','#E69F00','#009E73']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':9,
                     'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.linewidth':.7,
                     'axes.spines.top':False,'axes.spines.right':False,'lines.linewidth':1.3,
                     'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':200})

def data():return json.loads((HERE.parent/'results/results.json').read_text())
def save(fig,name):
    fig.tight_layout()
    for suffix in ('pdf','svg','png'):fig.savefig(HERE/f'{name}.{suffix}',bbox_inches='tight')
    plt.close(fig)
