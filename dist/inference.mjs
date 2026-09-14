// Exact variable elimination over Python-exported conditional probability tables.
// Every table uses explicit variable order and C-order flattening.
export const sizeOf = name => name === 'quality' ? 6 : 3;
const total = sizes => sizes.reduce((a, b) => a * b, 1);
function decode(index, sizes) {
  const out = Array(sizes.length);
  for (let i = sizes.length - 1; i >= 0; i--) { out[i] = index % sizes[i]; index = Math.floor(index / sizes[i]); }
  return out;
}
function offset(values, sizes) { return values.reduce((n, value, i) => n * sizes[i] + value, 0); }
function restrict(factor, evidence) {
  const keep = factor.variables.map((v, i) => [v, i]).filter(([v]) => !(v in evidence));
  const variables = keep.map(([v]) => v), sizes = keep.map(([, i]) => factor.sizes[i]);
  const values = new Float64Array(total(sizes));
  for (let i = 0; i < factor.values.length; i++) {
    const state = decode(i, factor.sizes);
    if (factor.variables.some((v, j) => v in evidence && evidence[v] !== state[j])) continue;
    values[offset(keep.map(([, j]) => state[j]), sizes)] = factor.values[i];
  }
  return {variables, sizes, values};
}
function multiply(factors) {
  const variables = [...new Set(factors.flatMap(f => f.variables))];
  const sizes = variables.map(sizeOf), values = new Float64Array(total(sizes));
  const positions = factors.map(f => f.variables.map(v => variables.indexOf(v)));
  for (let i = 0; i < values.length; i++) {
    const state = decode(i, sizes);
    values[i] = factors.reduce((value, f, k) => value * f.values[
      offset(positions[k].map(j => state[j]), f.sizes)], 1);
  }
  return {variables, sizes, values};
}
function sumOut(factor, variable) {
  const axis = factor.variables.indexOf(variable);
  const variables = factor.variables.filter((_, i) => i !== axis);
  const sizes = factor.sizes.filter((_, i) => i !== axis);
  const values = new Float64Array(total(sizes));
  for (let i = 0; i < factor.values.length; i++) {
    const state = decode(i, factor.sizes).filter((_, j) => j !== axis);
    values[offset(state, sizes)] += factor.values[i];
  }
  return {variables, sizes, values};
}
export function discretize(network, measurements) {
  const evidence = {};
  for (const [name, value] of Object.entries(measurements)) {
    if (!network.features.includes(name)) throw new Error(`Unknown measurement: ${name}`);
    if (value === null) continue;
    if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error('Measurements must be finite numbers.');
    evidence[name] = network.cuts[name].filter(cut => value >= cut).length;
  }
  return evidence;
}
export function query(network, measurements = {}) {
  const evidence = discretize(network, measurements);
  let factors = network.factors.map(f => restrict(f, evidence));
  let hidden = network.features.filter(v => !(v in evidence));
  while (hidden.length) {
    const cost = v => total([...new Set(factors.filter(f => f.variables.includes(v)).flatMap(f => f.variables))].map(sizeOf));
    hidden.sort((a, b) => cost(a) - cost(b));
    const variable = hidden.shift();
    const relevant = factors.filter(f => f.variables.includes(variable));
    factors = factors.filter(f => !f.variables.includes(variable));
    if (relevant.length) factors.push(sumOut(multiply(relevant), variable));
  }
  const result = multiply(factors);
  if (result.variables.length !== 1 || result.variables[0] !== 'quality') throw new Error('Invalid inference result.');
  const norm = result.values.reduce((a, b) => a + b, 0);
  if (!(norm > 0)) throw new Error('This evidence has zero probability in the model.');
  return Array.from(result.values, v => v / norm);
}
