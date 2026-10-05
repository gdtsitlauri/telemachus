from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw


LABELS={"nominal":"Point estimate","confidence_gate":"Confidence gate","budget_voi":"Budget VOI",
        "telemachus":"TELEMACHUS","no_source":"No source routing","no_model_update":"No model update","fixed_compute":"Fixed compute"}


def create_plots(results):
    results=Path(results);summary=json.loads((results/"summary.json").read_text())
    order=[m for m in LABELS if m in summary["methods"]]
    out=results/"plots";out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
    fig,axes=plt.subplots(1,2,figsize=(12,5),gridspec_kw={"width_ratios":[1.35,1]})
    left=np.zeros(len(order));y=np.arange(len(order))
    for endpoint,color,label in [("success","#227e65","Arrival"),("collision","#bf4b45","Collision"),("deferred","#a4adbc","Stopped / timeout")]:
        values=np.array([summary["methods"][m][endpoint] for m in order])*100
        axes[0].barh(y,values,left=left,color=color,label=label,height=.67);left+=values
    axes[0].set_yticks(y,[LABELS[m] for m in order]);axes[0].invert_yaxis()
    axes[0].set_xlim(0,100);axes[0].set_xlabel("Episodes (%)");axes[0].legend(loc="upper left",bbox_to_anchor=(0,-.12),ncol=3,frameon=False)
    values=[summary["methods"][m]["cost_mean"] for m in order]
    colors=["#246d93" if m=="telemachus" else "#a6bacb" for m in order]
    axes[1].barh(y,values,color=colors,height=.67);axes[1].set_yticks(y,[]);axes[1].invert_yaxis()
    for i,v in enumerate(values):axes[1].text(v+.35,i,f"{v:.1f}",va="center")
    axes[1].set_xlim(0,max(values)*1.2);axes[1].set_xlabel("Mean resource cost (declared proxy units)")
    n=summary["methods"][order[0]]["n"]
    fig.suptitle(f"TELEMACHUS v4 — {summary['split']}, {n} paired episodes per method",fontweight="bold")
    fig.tight_layout(rect=(0,.03,1,.95));fig.savefig(out/"outcomes_and_cost.png",bbox_inches="tight");plt.close(fig)
    strata=sorted(summary["by_scenario"][order[0]])
    matrix=np.array([[summary["by_scenario"][m][s]["success"] for s in strata] for m in order])
    fig,ax=plt.subplots(figsize=(9,4.8));im=ax.imshow(matrix,vmin=0,vmax=1,cmap="YlGnBu",aspect="auto")
    ax.set_xticks(range(len(strata)),strata);ax.set_yticks(range(len(order)),[LABELS[m] for m in order])
    for i in range(len(order)):
        for j in range(len(strata)):
            ax.text(j,i,f"{matrix[i,j]*100:.0f}%",ha="center",va="center",color="white" if matrix[i,j]>.65 else "black")
    fig.colorbar(im,ax=ax,label="Arrival rate");ax.set_title("Arrival by independently generated scenario")
    fig.tight_layout();fig.savefig(out/"scenario_success.png");plt.close(fig)
    return {"plots":[str(out/"outcomes_and_cost.png"),str(out/"scenario_success.png")]}


def render_episode(result,path):
    g=np.asarray(result["grid"],bool);trajectory=np.asarray(result["trajectory"]);goal=np.asarray(result["goal"])
    size=440;cell=size/len(g);frames=[]
    base=Image.new("RGB",(size,size+68),"#f3f5f7");d=ImageDraw.Draw(base)
    for r,c in np.argwhere(g):
        # Cartesian positive y points up.
        y=size-(r+1)*cell
        d.rectangle((c*cell,y,(c+1)*cell-1,y+cell-1),fill="#48596b")
    def pixel(xy):return (float(xy[0])*size/10,size-float(xy[1])*size/10)
    gx,gy=pixel(goal);d.ellipse((gx-8,gy-8,gx+8,gy+8),fill="#e6a02c")
    for i in range(1,len(trajectory)+1,2):
        image=base.copy();draw=ImageDraw.Draw(image)
        coords=[pixel(p[:2]) for p in trajectory[:i]]
        if len(coords)>1:draw.line(coords,fill="#228a70",width=3)
        x,y=coords[-1];theta=trajectory[i-1,2]
        draw.ellipse((x-6,y-6,x+6,y+6),fill="#226694")
        draw.line((x,y,x+13*np.cos(theta),y-13*np.sin(theta)),fill="#092d46",width=3)
        draw.text((12,size+10),f"TELEMACHUS v4 | {result['scenario']['name']} | step {i-1}",fill="#23313f")
        draw.text((12,size+32),f"Outcome: {result['outcome']} | cost: {result['resources']['cost']:.1f}",fill="#23313f")
        frames.append(image)
    if not frames:frames=[base]
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=100,loop=0)
    return str(path)
