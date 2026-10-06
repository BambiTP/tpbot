// nav/model.js must give the same outputs as PyTorch (model.json carries a reference input/output).
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { NavModel } = require('../nav/model');

// the newest runs/*/model.json, or one given on the command line
function newest() {
  const runs = path.join(__dirname, '..', 'runs');
  if (!fs.existsSync(runs)) return null;
  const files = fs.readdirSync(runs).map((r) => path.join(runs, r, 'model.json')).filter((f) => fs.existsSync(f));
  return files.sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs)[0] || null;
}
const file = process.argv[2] || newest();
if (!file || !fs.existsSync(file)) { console.log('skip: no', file, '(train first, or pass a model.json)'); process.exit(0); }
const json = JSON.parse(fs.readFileSync(file, 'utf8'));
const m = new NavModel(json);
const ref = json.reference;
const grid = new Uint8Array(Buffer.from(ref.grid, 'base64'));
const { logits, value } = m.forward(grid, Float32Array.from(ref.scalars));
let worst = Math.abs(value - ref.value);
ref.logits.forEach((l, i) => { worst = Math.max(worst, Math.abs(l - logits[i])); });
const t0 = Date.now(); for (let i = 0; i < 200; i++) m.forward(grid, Float32Array.from(ref.scalars));
console.log('JS vs PyTorch worst difference:', worst.toExponential(2), '| forward pass', ((Date.now() - t0) / 200).toFixed(2), 'ms');
assert(worst < 1e-3);
console.log('ok');
