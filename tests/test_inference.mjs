import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {query, discretize} from '../app/inference.mjs';
const experiment=JSON.parse(readFileSync(new URL('../app/results.json',import.meta.url)));
let checked=0;
for(const {evidence,probabilities} of experiment.reference_queries) {
  const actual=query(experiment.network,evidence);
  assert.equal(actual.length,6);
  assert.ok(Math.abs(actual.reduce((a,b)=>a+b,0)-1)<1e-12);
  actual.forEach((p,i)=>assert.ok(Math.abs(p-probabilities[i])<1e-10,`Inference mismatch at query ${checked}, class ${i+3}`));
  checked++;
}
assert.throws(()=>query(experiment.network,{quality:7}),/Unknown/);
assert.throws(()=>query(experiment.network,{alcohol:NaN}),/finite/);
assert.throws(()=>query(experiment.network,{alcohol:'12'}),/finite/);
for(const [name,cuts] of Object.entries(experiment.network.cuts)) {
  assert.equal(discretize(experiment.network,{[name]:cuts[0]})[name],1);
  assert.equal(discretize(experiment.network,{[name]:cuts[1]})[name],2);
}
console.log(`Exact browser inference matches Python for ${checked} queries, including every bin boundary.`);
