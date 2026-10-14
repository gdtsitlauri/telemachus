"""Build an offline interactive viewer from recorded experiment outputs."""
from pathlib import Path
import json


TEMPLATE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TELEMACHUS v5 · Experiment viewer</title>
<style>
:root{--ink:#203244;--muted:#63768a;--line:#dce5ed;--bg:#f5f8fa;--green:#27836d;--red:#c5564c;--blue:#286f98}
*{box-sizing:border-box}body{font:15px system-ui,sans-serif;color:var(--ink);background:var(--bg);margin:0}
main{max-width:1200px;margin:auto;padding:34px 24px}header{display:flex;justify-content:space-between;align-items:end;gap:20px}
h1{font-size:36px;letter-spacing:-1px;margin:5px 0}h2{font-size:21px;margin:0 0 15px}.eyebrow{font-size:12px;letter-spacing:2px;color:var(--blue);font-weight:700}
p{line-height:1.55}.muted,small{color:var(--muted)}.badge{border:1px solid var(--line);padding:8px 12px;border-radius:20px;font-size:12px;background:white}
.tabs{display:flex;gap:8px;margin:26px 0 18px;flex-wrap:wrap}button,select{font:inherit;border:1px solid var(--line);background:white;padding:9px 14px;border-radius:7px;color:var(--ink)}button{cursor:pointer}button.active{background:var(--ink);color:white}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card,.panel{background:white;border:1px solid var(--line);border-radius:12px;padding:20px}.card strong{display:block;font-size:29px;margin-top:9px}.card span{color:var(--muted);font-size:13px}
.panel{margin-top:18px}table{width:100%;border-collapse:collapse;font-size:13px}th{text-align:left;color:var(--muted);font-weight:500}td,th{padding:11px 8px;border-bottom:1px solid #eaf0f4}tr.primary{background:#ecf5f6}td.numeric,th.numeric{text-align:right}
.bar{display:flex;height:8px;border-radius:4px;overflow:hidden;min-width:90px;margin-top:5px}.arrival{background:var(--green)}.collision{background:var(--red)}.stop{background:#b3becc}
.twocol{display:grid;grid-template-columns:1fr 1fr;gap:18px}.controls{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px}.controls label{font-size:12px;color:var(--muted);display:grid;gap:4px}
svg{display:block;width:100%;max-height:400px;background:#f8fafb;border:1px solid var(--line);border-radius:8px}input[type=range]{width:100%;margin:16px 0}pre{white-space:pre-wrap;font:12px/1.6 ui-monospace,monospace;max-height:230px;overflow:auto;color:var(--muted)}
.note{font-size:12px;color:var(--muted);line-height:1.7}.legend{display:flex;gap:18px;font-size:12px;margin:12px 0}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}.scroll{overflow-x:auto}
@media(max-width:750px){main{padding:22px 14px}h1{font-size:28px}.cards{grid-template-columns:1fr 1fr}.twocol{grid-template-columns:1fr}header{display:block}.badge{display:inline-block;margin-top:10px}td,th{padding:10px 6px}}
</style><main>
<header><div><div class="eyebrow">PHYSICAL AI · RECORDED EXPERIMENTS</div><h1>TELEMACHUS v5</h1><div class="muted">Comparison of the methods and inspection of the trajectories from the recorded results.</div></div><div class="badge">CPU · Synthetic environments</div></header>
<div class="tabs" id="tabs"></div><div class="cards" id="cards"></div>
<section class="panel"><h2>Results on shared environments</h2><div class="scroll"><table><thead><tr><th>Method</th><th class="numeric">Success</th><th class="numeric">Collision</th><th class="numeric">Stop</th><th class="numeric">Mean cost</th><th class="numeric">Cost / success</th></tr></thead><tbody id="table"></tbody></table></div><div class="legend"><span><i class="dot arrival"></i>Success</span><span><i class="dot collision"></i>Collision</span><span><i class="dot stop"></i>Stop / timeout</span></div><p class="note">Cost is the declared resource unit of the simulation. A low cost with more stops does not show better efficiency. Intervals and per-seed comparisons are in the summary.json files.</p></section>
<div class="twocol"><section class="panel"><h2>One complete trajectory</h2><div class="controls"><label>Method<select id="method"></select></label><label>Scenario<select id="scenario"></select></label></div><svg id="world" viewBox="0 0 440 440" role="img" aria-label="Recorded robot trajectory"></svg><input id="tick" aria-label="Physical step" type="range" min="0" value="0"><div class="muted" id="episode"></div></section>
<section class="panel"><h2>Decisions and resources</h2><div class="note" id="sample"></div><pre id="trace"></pre><h2 style="margin-top:22px">Evaluation scope</h2><p class="note">The final v5 and the frozen v4 assess risk over the same prediction horizon (1.2 s). The command actually executed lasts 0.4 s. The simpler v5 methods share one control core, so that the choice of resources is isolated.</p><p class="note">The trajectories are evaluation data of the simulator. The active policy receives measurements, without the true position, map or fault flags. The Monte Carlo check concerns the internal distribution and one decision; it is not a guarantee of real or overall safety.</p><p class="note">The examples show the first predefined seed of each protocol. Successes, failures and stops are included.</p></section></div>
<p class="note">Open this file locally. No connection or data upload is needed. The full logs, protocols and trained models are in the same package.</p>
</main><script type="application/json" id="data">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent);let split='holdout';
const $=id=>document.getElementById(id),pct=v=>(100*v).toFixed(1)+'%',num=v=>v===null?'—':v.toFixed(2);
const labels={legacy_v4:'Frozen v4',nominal:'Point estimate',confidence_gate:'Confidence gate',budget_voi:'Budget VOI',telemachus:'TELEMACHUS v5',no_learning:'Heuristic acquisition',no_reserve:'No mission reserve',no_source:'Source-blind learner',no_model_update:'No model update',fixed_compute:'Fixed compute'};
const title={holdout:'Budget 120',holdout_budget30:'Budget 30',stress:'Stress · Budget 120'};
function activate(s){split=s;for(const b of $('tabs').children)b.classList.toggle('active',b.dataset.split===s);const d=D[s],r=d.summary.methods.telemachus;
 $('cards').innerHTML=`<div class="card"><span>Missions per method</span><strong>${r.n}</strong></div><div class="card"><span>TELEMACHUS · Success</span><strong>${pct(r.success)}</strong></div><div class="card"><span>TELEMACHUS · Collision</span><strong>${pct(r.collision)}</strong></div><div class="card"><span>TELEMACHUS · Mean cost</span><strong>${num(r.cost_mean)}</strong></div>`;
 $('table').innerHTML=Object.keys(labels).filter(m=>d.summary.methods[m]).map(m=>{const a=d.summary.methods[m];return `<tr class="${m==='telemachus'?'primary':''}"><td>${labels[m]}<div class="bar"><span class="arrival" style="width:${100*a.success}%"></span><span class="collision" style="width:${100*a.collision}%"></span><span class="stop" style="width:${100*a.deferred}%"></span></div></td><td class="numeric">${pct(a.success)}</td><td class="numeric">${pct(a.collision)}</td><td class="numeric">${pct(a.deferred)}</td><td class="numeric">${num(a.cost_mean)}</td><td class="numeric">${num(a.cost_per_success)}</td></tr>`}).join('');
 $('method').innerHTML=Object.keys(labels).filter(m=>d.examples[m]).map(m=>`<option value="${m}">${labels[m]}</option>`).join('');$('method').value='telemachus';
 $('scenario').innerHTML=Object.keys(d.examples.telemachus).map(s=>`<option>${s}</option>`).join('');choose();}
function choose(){const r=D[split].examples[$('method').value][$('scenario').value];$('tick').max=r.trajectory.length-1;$('tick').value=0;draw();}
function draw(){const r=D[split].examples[$('method').value][$('scenario').value],i=+$('tick').value;let pieces=[];const n=r.grid.length,c=440/n;
 for(let y=0;y<n;y++)for(let x=0;x<n;x++)if(r.grid[y][x])pieces.push(`<rect x="${x*c}" y="${440-(y+1)*c}" width="${c}" height="${c}" fill="#52677b"/>`);
 const xy=p=>[p[0]*44,440-p[1]*44],g=xy(r.goal),p=xy(r.trajectory[i]);pieces.push(`<circle cx="${g[0]}" cy="${g[1]}" r="8" fill="#e0a64c"/>`);
 if(i)pieces.push(`<polyline points="${r.trajectory.slice(0,i+1).map(v=>xy(v).join(',')).join(' ')}" fill="none" stroke="#27836d" stroke-width="3"/>`);
 pieces.push(`<circle cx="${p[0]}" cy="${p[1]}" r="6" fill="#286f98"/>`);$('world').innerHTML=pieces.join('');
 $('episode').textContent=`Seed ${r.seed} · step ${i}/${r.steps} · ${r.outcome} · ${r.reason}`;
 $('sample').textContent=`MC samples ${r.resources.mc_samples} · Observations ${r.resources.observations} · Queries ${r.resources.queries} · Probes ${r.resources.probes} · CPU ${num(r.wall_seconds)} s`;
 const t=r.trace.filter(t=>t.tick===i).map(t=>({route:t.route,samples:t.n_samples,control:t.selected_control,risk_upper:t.risk_ucb,support_lower:t.support_lcb,cost:t.cost,resource_predictions:t.resource_predictions}));$('trace').textContent=t.length?JSON.stringify(t,null,2):'No recorded decision at this physical step.';}
for(const s of Object.keys(title)){let b=document.createElement('button');b.textContent=title[s];b.dataset.split=s;b.onclick=()=>activate(s);$('tabs').appendChild(b)}
$('method').onchange=choose;$('scenario').onchange=choose;$('tick').oninput=draw;activate(split);
</script></html>'''


def build_review(root):
    root=Path(root);data={}
    for folder in ("holdout","holdout_budget30","stress"):
        p=root/"results"/folder
        summary=json.loads((p/"summary.json").read_text())
        protocol=json.loads((p/"protocol.json").read_text())
        examples={}
        for method in protocol["methods"]:
            examples[method]={}
            for scenario in protocol["scenarios"]:
                name=scenario["name"]
                r=json.loads((p/"episodes"/f"{name}_{protocol['seeds'][0]}_{method}.json").read_text())
                examples[method][name]={k:r[k] for k in ("seed","steps","outcome","reason","grid","goal","trajectory","trace","resources","wall_seconds")}
        data[folder]={"summary":summary,"examples":examples}
    raw=json.dumps(data,separators=(",",":"),allow_nan=False).replace("<","\\u003c")
    output=root/"results/review.html";output.write_text(TEMPLATE.replace("__DATA__",raw))
    return output


if __name__=="__main__":print(build_review(Path(__file__).resolve().parents[1]))
