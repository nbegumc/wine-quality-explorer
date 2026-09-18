import {query, discretize} from './inference.mjs';

const $ = selector => document.querySelector(selector);
const labels = {fixed_acidity:'Fixed acidity',volatile_acidity:'Volatile acidity',citric_acid:'Citric acid',residual_sugar:'Residual sugar',chlorides:'Chlorides',free_sulfur_dioxide:'Free sulfur dioxide',total_sulfur_dioxide:'Total sulfur dioxide',density:'Density',pH:'pH',sulphates:'Sulphates',alcohol:'Alcohol',quality:'Quality'};
const units = {alcohol:'% vol',density:'g/cm³',pH:'',free_sulfur_dioxide:'mg/L',total_sulfur_dioxide:'mg/L'};
const steps = {alcohol:0.1,density:0.00001,pH:0.01,free_sulfur_dioxide:0.5,total_sulfur_dioxide:0.5,fixed_acidity:0.1,volatile_acidity:0.005,citric_acid:0.01,residual_sugar:0.05,chlorides:0.001,sulphates:0.01};
const descriptions = {quality:'The sensory score assigned to the wine. The model predicts a probability for each score from 3 to 8.',alcohol:'Alcohol concentration. The model can associate this measurement with quality and other wine properties.',volatile_acidity:'Volatile acidity measured in the wine. Its association with quality is learned from the training observations.',sulphates:'Measured sulphate concentration. This is distinct from the free and total sulfur dioxide measurements.',pH:'The acidity scale. pH and fixed acidity describe different properties.',density:'Mass per unit volume, associated with the composition of the wine.'};
const fmt = n => Number(n).toLocaleString(undefined,{maximumFractionDigits:4});
const STABLE=0.85,WEAK=0.5; // Arrow-stability buckets; 0.85 is the original R project's averaging threshold.
const pct = n => `${(n*100).toFixed(1)}%`;
let experiment, evidence={}, prior=[], probabilities=[], inspectedModel, currentSample=null, node='quality';

function featureLabel(name) { return labels[name] || name; }
function numericEvidence(input) {
  if (!input || typeof input !== 'object' || Array.isArray(input)) throw new Error('Pass an object containing measurements.');
  const result = {};
  for(const [name,value] of Object.entries(input)) {
    if(!experiment.network.features.includes(name)) throw new Error(`Unknown measurement: ${name}`);
    if(value === null) continue;
    if(typeof value!=='number'||!Number.isFinite(value)) throw new Error('Measurements must be finite numbers.');
    const range=experiment.ranges[name];
    if(value<range.min || value>range.max) throw new Error(`${featureLabel(name)} must be between ${range.min} and ${range.max}.`);
    result[name]=value;
  }
  return result;
}
function referencePrediction() {
  // The validation-selected prediction model needs every measurement; the network does not.
  const ref=experiment.reference_model;
  if(!ref||ref.features.some(n=>!(n in evidence)))return null;
  const z=ref.features.map((n,i)=>(evidence[n]-ref.mean[i])/ref.scale[i]);
  const logits=ref.coef.map((row,k)=>ref.intercept[k]+row.reduce((s,w,i)=>s+w*z[i],0));
  const max=Math.max(...logits),weights=logits.map(l=>Math.exp(l-max)),norm=weights.reduce((a,b)=>a+b,0);
  const aligned=experiment.network.classes.map(()=>0);
  ref.classes.forEach((q,k)=>{aligned[q-3]=weights[k]/norm;});
  return aligned;
}
function buildControls() {
  const primary=['alcohol','volatile_acidity','sulphates'];
  for(const name of [...primary,...experiment.network.features.filter(n=>!primary.includes(n))]) {
    const range=experiment.ranges[name],unit=units[name]??'g/L';
    const container=document.createElement('div');container.className='measurement';container.dataset.feature=name;
    container.innerHTML=`<div class="measurement-top"><label for="include-${name}"><input type="checkbox" id="include-${name}">${featureLabel(name)} <span class="unit">${unit}</span></label><span class="bin-badge" id="bin-${name}">Unknown</span></div><div class="measurement-values"><input type="range" id="range-${name}" min="${range.min}" max="${range.max}" step="${steps[name]}" value="${range.median}" aria-label="${featureLabel(name)} slider"><input type="number" id="value-${name}" min="${range.min}" max="${range.max}" step="any" value="${range.median}" aria-label="${featureLabel(name)} value"></div><span class="bin-note">Low &lt; ${fmt(experiment.network.cuts[name][0])} · High ≥ ${fmt(experiment.network.cuts[name][1])}</span>`;
    $(primary.includes(name)?'#primary-inputs':'#extra-inputs').append(container);
    $(`#include-${name}`).addEventListener('change',event=>{
      currentSample=null;$('#sample').value='';
      if(event.target.checked)evidence[name]=Number($(`#value-${name}`).value);else delete evidence[name];
      updatePrediction();
    });
    $(`#range-${name}`).addEventListener('input',event=>{
      evidence[name]=Number(event.target.value);currentSample=null;$('#sample').value='';updatePrediction();
    });
    $(`#value-${name}`).addEventListener('change',event=>{
      if(event.target.value===''||!event.target.checkValidity()){event.target.reportValidity();return;}
      evidence[name]=Number(event.target.value);currentSample=null;$('#sample').value='';updatePrediction();
    });
  }
  for(const example of experiment.examples) {
    const option=document.createElement('option');option.value=example.id;option.textContent=`Wine #${example.id+1} · actual score ${example.quality}`;$('#sample').append(option);
  }
  $('#sample').addEventListener('change',event=>{
    const example=experiment.examples.find(e=>String(e.id)===event.target.value);if(!example)return;
    evidence={...example.measurements};currentSample=example;updatePrediction();
  });
  $('#clear-evidence').addEventListener('click',()=>{evidence={};currentSample=null;$('#sample').value='';updatePrediction();});
}
function syncControls() {
  const bins=discretize(experiment.network,evidence);
  for(const name of experiment.network.features) {
    const included=name in evidence;
    $(`#include-${name}`).checked=included;
    $(`#bin-${name}`).textContent=included?experiment.network.states[bins[name]]:'Unknown';
    const group=$(`[data-feature="${name}"] .measurement-values`);group.classList.toggle('disabled',!included);
    for(const kind of ['range','value']) {const input=$(`#${kind}-${name}`);input.disabled=!included;if(included)input.value=evidence[name];}
  }
  $('#sample-note').textContent=currentSample?`Original row ${currentSample.id+1}; actual quality ${currentSample.quality}. This wine was excluded from training.`:'';
}
function updatePrediction() {
  probabilities=query(experiment.network,evidence);syncControls();
  const top=probabilities.indexOf(Math.max(...probabilities)),expected=probabilities.reduce((n,p,i)=>n+p*(i+3),0),count=Object.keys(evidence).length;
  $('#top-score').textContent=String(top+3);$('#expected-score').textContent=expected.toFixed(2);$('#high-probability').textContent=pct(probabilities[4]+probabilities[5]);
  $('#evidence-count').textContent=count?`${count} of 11 measurements`:'No evidence';
  const scale=Math.max(.05,...probabilities,...prior);
  $('#probability-chart').innerHTML=probabilities.map((p,i)=>`<div class="prob-column"><span class="prob-value">${pct(p)}</span><div class="prob-bars"><div class="prob-bar" style="height:${p/scale*100}%" title="Quality ${i+3}: ${pct(p)} with evidence"></div><div class="prob-bar baseline" style="height:${prior[i]/scale*100}%" title="Quality ${i+3}: ${pct(prior[i])} without evidence"></div></div><span class="quality-label">${i+3}</span></div>`).join('');
  $('#probability-chart').setAttribute('aria-label',probabilities.map((p,i)=>`Quality ${i+3}: ${pct(p)}`).join(', '));
  let description=`The network puts ${pct(probabilities[top])} on score ${top+3}, its most likely value. `;
  description+=count===experiment.network.features.length?'Every measurement is observed, so nothing is integrated out.':count?'Unknown measurements are integrated out using the learned network.':'This is the network’s marginal distribution before observing any measurements.';
  if(currentSample)description+=` This wine’s actual score is ${currentSample.quality}.`;
  $('#prediction-description').textContent=description;
  const ref=experiment.reference_model,reference=referencePrediction();
  if(!ref)$('#reference-line').hidden=true;
  else if(reference){const best=reference.indexOf(Math.max(...reference));$('#reference-line').innerHTML=`<strong>Prediction reference (${ref.name.toLowerCase()}):</strong> score ${best+3} most likely at ${pct(reference[best])}, 7 or 8 at ${pct(reference[4]+reference[5])}. All 11 measurements are known, so the reference applies.`;}
  else $('#reference-line').innerHTML=`<strong>Prediction reference (${ref.name.toLowerCase()}):</strong> needs all 11 measurements. With ${count} of 11, only the network can answer.`;
}
function renderModelTable() {
  const split=$('#metric-split').value;
  $('#model-table tbody').innerHTML=experiment.models.map(model=>{
    const m=model[split],tag=model.selected_overall?'Prediction reference':model.selected_bn?'Reasoning model':'';
    return `<tr class="${model.selected_bn?'selected-row':''}"><td>${model.name}${tag?`<span class="table-tag">${tag}</span>`:''}</td><td>${pct(m.accuracy)}</td><td>${m.macro_f1.toFixed(3)}</td><td>${pct(m.macro_recall)}</td><td>${m.mae.toFixed(3)}</td><td>${m.log_loss.toFixed(3)}</td></tr>`;
  }).join('');
}
function renderModelDetail() {
  const model=experiment.models.find(m=>m.name===inspectedModel),matrix=model.test.confusion,max=Math.max(...matrix.flat(),1);
  $('#confusion-matrix').innerHTML=`<table aria-label="Confusion matrix"><thead><tr><th scope="col">Actual ↓</th>${experiment.network.classes.map(q=>`<th scope="col">${q}</th>`).join('')}</tr></thead><tbody>${matrix.map((row,i)=>`<tr><th scope="row">${i+3}</th>${row.map((n,j)=>`<td title="Actual ${i+3}, predicted ${j+3}: ${n} wines" style="background:rgba(117,35,80,${.035+.85*n/max});color:${n/max>.5?'white':'#17212f'}">${n}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
  $('#recall-chart').innerHTML=model.test.per_class.map(c=>`<div class="recall-row"><span>Score ${c.quality}</span><div class="track"><div class="fill" style="width:${c.recall*100}%"></div></div><span>${pct(c.recall)}<br><small>n = ${c.support}</small></span></div>`).join('');
  $('#calibration-chart').innerHTML=`<svg viewBox="0 0 560 270" role="img" aria-label="Calibration plot: model confidence against observed accuracy"><line x1="55" y1="215" x2="515" y2="215" stroke="#bbc5d2"/><line x1="55" y1="215" x2="55" y2="25" stroke="#bbc5d2"/><line x1="55" y1="215" x2="515" y2="25" stroke="#a4b0bf" stroke-dasharray="5,5"/>${[0,.25,.5,.75,1].map(t=>`<text x="${55+460*t}" y="235" text-anchor="middle" font-size="12" fill="#617082">${Math.round(t*100)}%</text><text x="44" y="${219-190*t}" text-anchor="end" font-size="12" fill="#617082">${Math.round(t*100)}%</text>`).join('')}${model.calibration.map(b=>`<circle cx="${55+460*b.confidence}" cy="${215-190*b.accuracy}" r="${Math.min(13,4+Math.sqrt(b.count)/2)}" fill="#752350" opacity=".8"><title>Mean probability ${pct(b.confidence)}, accuracy ${pct(b.accuracy)}, ${b.count} wines</title></circle>`).join('')}<text x="285" y="262" text-anchor="middle" font-size="13" fill="#617082">Highest model probability</text></svg>`;
}
function selectNode(name) {node=name;renderNetwork();}
function renderNetwork() {
  const network=experiment.network;
  // A fixed, labelled arrangement keeps graph positions stable while inspecting.
  const positions={quality:[525,75],alcohol:[200,235],sulphates:[525,235],volatile_acidity:[850,235],residual_sugar:[175,425],density:[525,425],citric_acid:[875,425],chlorides:[175,615],pH:[410,615],fixed_acidity:[650,615],total_sulfur_dioxide:[890,615],free_sulfur_dioxide:[850,735]};
  const neighbours=new Set([node,...network.arcs.filter(([a,b])=>a===node||b===node).flat()]);
  const force=new Set(network.forced_arcs.map(a=>a.join('|')));
  // Bootstrap arc strength is optional: a bare scripts/train.py run exports no arc_strength block.
  const stability=network.arc_strength,strengths=new Map((stability?.pairs??[]).map(p=>[`${p.a}|${p.b}`,p]));
  const strengthOf=(a,b)=>{if(!stability)return null;const p=strengths.get([a,b].sort().join('|'));return p?{strength:p.strength,agree:a<b?p.direction:1-p.direction}:{strength:0,agree:0};};
  const bucket=s=>s>=STABLE?'stable':s>=WEAK?'moderate':'weak',pct0=n=>`${Math.round(n*100)}%`;
  const edges=network.arcs.map(([a,b])=>{
    const [x1,y1]=positions[a],[x2,y2]=positions[b],dx=x2-x1,dy=y2-y1,len=Math.hypot(dx,dy),pad=dy===0?95:35;
    const sx=x1+dx/len*pad,sy=y1+dy/len*pad,ex=x2-dx/len*pad,ey=y2-dy/len*pad;
    const curve=Math.abs(dx)<25?35:Math.abs(dy)<25?40:0;
    const forced=force.has(`${a}|${b}`),active=a===node||b===node,info=strengthOf(a,b);
    const detail=info?` · in ${pct0(info.strength)} of ${stability.repeats.toLocaleString()} bootstrap graphs · ${pct0(info.agree)} in this direction`:'';
    return `<path d="M ${sx} ${sy} Q ${(sx+ex)/2+curve} ${(sy+ey)/2-curve} ${ex} ${ey}" class="edge ${forced?'forced':''} ${info?bucket(info.strength):''} ${active?'':'dimmed'}" marker-end="url(#${forced?'arrow-wine':'arrow'})"><title>${featureLabel(a)} → ${featureLabel(b)}${forced?' (prior assumption)':''}${detail}</title></path>`;
  }).join('');
  const nodes=[...network.features,'quality'].map(name=>{
    const [x,y]=positions[name],words=featureLabel(name),wide=words.length>18?200:words.length>13?170:145;
    return `<g class="node ${name==='quality'?'quality':''} ${name===node?'selected':''} ${neighbours.has(name)?'':'dimmed'}" data-node="${name}" tabindex="0" role="button" aria-label="Inspect ${words}" transform="translate(${x},${y})"><rect x="${-wide/2}" y="-25" width="${wide}" height="50" rx="8"/><text text-anchor="middle" dominant-baseline="middle">${words}</text></g>`;
  }).join('');
  $('#network-svg').setAttribute('viewBox','0 0 1050 790');
  $('#network-svg').innerHTML=`<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7" fill="#9aa9ba"/></marker><marker id="arrow-wine" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7" fill="#752350"/></marker></defs>${edges}${nodes}`;
  const stable=stability?network.arcs.filter(([a,b])=>strengthOf(a,b).strength>=STABLE).length:null;
  $('#network-summary').textContent=`12 variables · ${network.arcs.length} arrows${stability?` · ${stable} stable`:''}`;
  document.querySelectorAll('[data-node]').forEach(el=>{el.addEventListener('click',()=>selectNode(el.dataset.node));el.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();selectNode(el.dataset.node);}});});
  $('#selected-node').textContent=featureLabel(node);$('#node-description').textContent=descriptions[node]||'A laboratory measurement included in the learned joint probability model.';
  for(const [selector,names] of [['#node-parents',network.arcs.filter(([,b])=>b===node).map(([a])=>a)],['#node-children',network.arcs.filter(([a])=>a===node).map(([,b])=>b)]]) {
    $(selector).innerHTML=names.length?names.map(n=>{const info=strengthOf(n,node);return `<button class="node-chip" data-pick="${n}">${featureLabel(n)}${info?` · ${pct0(info.strength)}`:''}</button>`;}).join(''):'<span class="micro">None in this network</span>';
  }
  document.querySelectorAll('[data-pick]').forEach(el=>el.addEventListener('click',()=>selectNode(el.dataset.pick)));
  const drawn=new Set(network.arcs.map(a=>a.join('|')));
  const absent=(stability?.pairs??[]).filter(p=>p.strength>=WEAK&&!drawn.has(`${p.a}|${p.b}`)&&!drawn.has(`${p.b}|${p.a}`));
  $('#absent-heading').hidden=$('#absent-pairs').hidden=!absent.length;
  $('#absent-pairs').innerHTML=absent.map(p=>`<span>${featureLabel(p.a)} – ${featureLabel(p.b)} · ${pct0(p.strength)}</span>`).join('');
  document.querySelectorAll('.stability-legend').forEach(el=>el.hidden=!stability);
  $('#stability-note').hidden=!stability;
  if(stability)$('#stability-note').textContent=`Arrow stability: the ${experiment.selection.bn} recipe was relearned on ${stability.repeats.toLocaleString()} resamples of the training wines (measurement groups drawn with replacement). Solid arrows appeared in at least ${pct0(STABLE)} of those graphs, the threshold the original R project used to keep an arrow; dotted arrows appeared in fewer than half. A direction share near 50% in a tooltip means the data does not determine which way the arrow points. This is a diagnostic of the frozen model; predictions are unchanged.`;
}
function showView(name) {
  if(!['predict','compare','network','study'].includes(name))name='predict';
  document.querySelectorAll('.view').forEach(view=>view.hidden=view.id!==name);
  document.querySelectorAll('.tab').forEach(button=>{const active=button.dataset.view===name;button.classList.toggle('active',active);if(active)button.setAttribute('aria-current','page');else button.removeAttribute('aria-current');});
  history.replaceState(null,'',`#${name}`);
}
function registerAgentTools() {
  if(!document.modelContext?.registerTool)return;
  const controller=new AbortController();
  window.addEventListener('pagehide',()=>controller.abort(),{once:true});
  const tool={name:'set_wine_measurements',title:'Set wine measurements',description:'Replace the visible wine measurements and return the network’s updated quality probabilities, plus the prediction reference when all measurements are given. Omitted or null measurements are treated as unknown.',inputSchema:{type:'object',properties:{measurements:{type:'object',properties:Object.fromEntries(experiment.network.features.map(n=>[n,{type:['number','null']}])) ,additionalProperties:false}},required:['measurements'],additionalProperties:false},annotations:{readOnlyHint:false,untrustedContentHint:false},execute(input){if(!input||Object.keys(input).some(k=>k!=='measurements'))throw new Error('Expected measurements only.');const validated=numericEvidence(input.measurements);evidence=validated;currentSample=null;$('#sample').value='';showView('predict');updatePrediction();const reference=referencePrediction();return{measurements:{...evidence},quality_probabilities:Object.fromEntries(probabilities.map((p,i)=>[String(i+3),p])),model:experiment.selection.bn,reference_model:experiment.reference_model?.name??null,reference_probabilities:reference?Object.fromEntries(reference.map((p,i)=>[String(i+3),p])):null};}};
  try{Promise.resolve(document.modelContext.registerTool(tool,{signal:controller.signal})).catch(()=>{});}catch{}
}
async function start() {
  try {
    if(['localhost','127.0.0.1'].includes(location.hostname)) {
      $('.source-link').href='results.json';$('.source-link').textContent='Download experiment results ↓';
    }
    const response=await fetch('results.json');if(!response.ok)throw new Error('Experiment results could not be loaded.');
    experiment=await response.json();const selected=experiment.models.find(m=>m.selected_bn);inspectedModel=selected.name;
    prior=query(experiment.network,{});evidence=Object.fromEntries(['alcohol','volatile_acidity','sulphates'].map(n=>[n,experiment.ranges[n].median]));
    $('#dataset-summary').innerHTML=`<strong>${experiment.dataset.rows.toLocaleString()}</strong> wines<br>${experiment.split.train_rows.toLocaleString()} training · ${experiment.split.test_rows} held out`;
    $('#model-name').textContent=selected.name;
    $('#accuracy-interval').textContent=`${pct(selected.test.accuracy)} on ${experiment.split.test_rows} unseen wines (95% interval ${pct(selected.accuracy_interval[0])}–${pct(selected.accuracy_interval[1])}). Answers with any subset of measurements.`;
    const reference=experiment.models.find(m=>m.selected_overall);
    $('#reference-name').textContent=reference.name;
    $('#reference-accuracy').textContent=`${pct(reference.test.accuracy)} on the same wines, selected by validation macro-F1. Needs all 11 measurements.`;
    $('#selection-note').textContent=`Validation macro-F1 selected ${experiment.selection.overall} as the prediction reference and ${experiment.selection.bn} as the reasoning network, before the holdout was evaluated. The network is kept for what it can do with partial evidence, not for its score.`;
    buildControls();updatePrediction();renderModelTable();
    for(const model of experiment.models){const option=document.createElement('option');option.value=model.name;option.textContent=model.name;option.selected=model.name===inspectedModel;$('#inspect-model').append(option);}
    $('#inspect-model').addEventListener('change',event=>{inspectedModel=event.target.value;renderModelDetail();});
    $('#metric-split').addEventListener('change',renderModelTable);renderModelDetail();renderNetwork();
    const max=Math.max(...Object.values(experiment.dataset.class_counts));
    $('#class-chart').innerHTML=Object.entries(experiment.dataset.class_counts).map(([q,n])=>`<div class="class-row ${q==='5'||q==='6'?'majority':''}"><span>Score ${q}</span><div class="track"><div class="fill" style="width:${n/max*100}%"></div></div><span>${n}</span></div>`).join('');
    $('#group-description').textContent=`${experiment.dataset.duplicate_rows} repeated rows are retained, but all identical predictor rows stay in the same group. ${experiment.dataset.unique_predictor_groups} unique measurement groups.`;
    $('#split-description').textContent=`${experiment.split.train_rows} training rows and ${experiment.split.test_rows} holdout rows, using the first stratified group fold with seed ${experiment.split.seed}.`;
    registerAgentTools();showView(location.hash.slice(1));
  } catch(error) {$('#load-error').hidden=false;$('#load-error').textContent=`Unable to load the explorer: ${error.message} If running locally, start it with python scripts/serve.py rather than opening the HTML file directly.`;$('#dataset-summary').textContent='Results unavailable';}
}
document.querySelectorAll('.tab').forEach(button=>button.addEventListener('click',()=>showView(button.dataset.view)));
window.addEventListener('hashchange',()=>showView(location.hash.slice(1)));
start();
