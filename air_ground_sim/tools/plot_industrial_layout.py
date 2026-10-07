"""Development plot of final cell coordinates and audited straight corridors.

Uses matplotlib (optional authoring dependency, not required by the simulator).
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch

ROOT=Path(__file__).resolve().parents[1]
plan=json.loads((ROOT/'configs/factory_industrial_layout.json').read_text())
audit=json.loads((ROOT/'outputs/factory_industrial_audit/report.json').read_text())
source=ET.parse(ROOT/'assets/factory_industrial/authoring/model.xml').getroot()
out=ROOT/'outputs/factory_industrial_audit'
fig,ax=plt.subplots(figsize=(14,10.5),constrained_layout=True)
fig.set_facecolor('#f4f6f8');ax.set_facecolor('#eef1f2')
ax.add_patch(Rectangle((-200,-150),400,300,fc='#e6ebec',ec='#283a47',lw=1.8,zorder=0))
for xy,w,h in [((-6,-142),12,284),((-192,-5),384,10),((-192,-87),384,10)]:
    ax.add_patch(Rectangle(xy,w,h,color='#c6d1d5',zorder=1))
for n,cell in enumerate(plan['relocated_cells'],1):
    x,y,z=map(float,cell['after']['pos'].split())
    ax.add_patch(Rectangle((x-5.45,y-18.06),10.9,36.12,fc='#648b9a',ec='#2c5669',lw=.7,zorder=3))
    ax.text(x,y,str(n).zfill(2),ha='center',va='center',fontsize=6.5,color='white',zorder=4)
for module in plan['modules']:
    x,y,z=module['position'];kind=module['kind']
    if kind=='assembly':ax.scatter(x,y,s=12,c='#dd793c',marker='s',zorder=5)
    elif kind=='control':ax.scatter(x,y,s=6,c='#536773',marker='s',zorder=5)
    elif kind=='wip':ax.scatter(x,y,s=8,c='#2f8294',marker='s',zorder=5)
    elif kind=='pallet_staging':ax.add_patch(Rectangle((x-1.6,y-1.25),3.2,2.5,fc='#a67838',ec='#74521e',lw=.4,zorder=4))
    elif kind=='training_bay':ax.add_patch(Rectangle((x-4.6,y-4.9),9.2,9.8,fc='none',ec='#c27630',lw=1,zorder=4))
for g in source.find('worldbody').findall('geom'):
    name=g.get('name','')
    if name.startswith('rack_') and name.endswith('upright_001'):
        x,y,z=map(float,g.get('pos').split())
        ax.add_patch(Rectangle((x,y),16,2,fc='#c99e5c',ec='#997334',lw=.6,zorder=3))
    elif name.startswith('column_foot_'):
        x,y,z=map(float,g.get('pos').split())
        ax.add_patch(Rectangle((x-.48,y-.48),.96,.96,color='#647079',zorder=4))
for route in audit['roads']:
    walking='walk' in route['name']
    start,end=route['start'],route['end']
    ax.plot([start[0],end[0]],[start[1],end[1]],color='#518b68' if walking else '#567baa',
            lw=.65,alpha=.6,ls='--' if walking else '-',zorder=2)
ax.scatter(-180,-81.75,s=75,marker='*',c='#cc483d',zorder=6)
ax.annotate('Default robot start',(-180,-81.75),(-198,-70),fontsize=8,color='#ad382f',
            arrowprops=dict(arrowstyle='-',lw=.8,color='#ad382f'),zorder=7)
ax.text(-194,105,'UTILITY / SERVICES',fontsize=11,color='#536773',weight='bold')
ax.text(-194,-113,'WAREHOUSE',fontsize=10,color='#84652f',rotation=90,va='center',weight='bold')
ax.set(xlim=(-204,204),ylim=(-154,154),xlabel='World X (m)',ylabel='World Y (m)')
ax.set_aspect('equal');ax.grid(alpha=.12,linewidth=.5)
ax.set_title('Manufacturing + logistics | corrected factory layout',loc='left',fontsize=17,pad=15,weight='bold',color='#253d4b')
ax.legend(handles=[Patch(fc='#648b9a',label='Production cells (numbered)'),
    Patch(fc='#dd793c',label='Assembly / QC'),Patch(fc='#2f8294',label='WIP carts'),
    Patch(fc='#c99e5c',label='Storage racks'),Patch(fc='#518b68',label='Audited walking segments'),
    Patch(fc='#567baa',label='Audited vehicle segments')],loc='upper center',bbox_to_anchor=(.5,-.065),
    ncol=3,frameon=False,fontsize=9)
fig.savefig(out/'layout.png',dpi=180)
fig.savefig(out/'layout.svg')
plt.close(fig)
print(out/'layout.png')
