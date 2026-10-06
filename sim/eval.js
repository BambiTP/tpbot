// eval.js - how good is a trained model? Plays fixed test episodes (the same every time) greedily
// with the JS runtime (nav/model.js), once network-only and once with the strike planner.
//
//   node sim/eval.js --run first [--episodes 300] [--append]     # uses runs/first/model.json
//   node sim/eval.js --model path/to/model.json
//
// --append adds a line to runs/<run>/eval.jsonl (the trainer does this; the UI charts it).
const fs = require('fs');
const path = require('path');
const G = require('../nav/geometry');
const { Env } = require('./env');
const { NavModel } = require('../nav/model');
const { plan } = require('../nav/planner');

const args = {};
for (let i = 2; i < process.argv.length; i++) {
  const a = process.argv[i];
  if (a.startsWith('--')) { const next = process.argv[i + 1]; if (next === undefined || next.startsWith('--')) args[a.slice(2)] = true; else { args[a.slice(2)] = next; i++; } }
}
const runDir = args.run ? path.join(__dirname, '..', 'runs', args.run) : null;
const modelPath = args.model || (runDir && path.join(runDir, 'model.json'));
if (!modelPath || !fs.existsSync(modelPath)) { console.error('no model.json yet (' + modelPath + ')'); process.exit(1); }
const json = JSON.parse(fs.readFileSync(modelPath, 'utf8'));
const model = new NavModel(json);
const episodes = Number(args.episodes || 300);

function play(usePlanner) {
  const env = new Env({ seed: 987654 }); // fixed test set
  const grid = new Uint8Array(G.GRID), scal = new Float32Array(G.SCALARS);
  const out = { episodes, hit: 0, graze: 0, pop: 0, timeout: 0, lost: 0, miss: [], time: [], pathSpeed: [], plannerTicks: 0, ticks: 0, ms: 0 };
  for (let e = 0; e < episodes; e++) {
    env.reset();
    for (;;) {
      const t0 = process.hrtime.bigint();
      let a = null;
      if (usePlanner) { const p = plan(env.me(), env.target, { tiles: env.room.tiles, team: 1 }); if (p) { a = p.action; out.plannerTicks++; } }
      if (a === null) { env.observe(grid, 0, scal, 0); a = model.act(grid, scal); }
      out.ms += Number(process.hrtime.bigint() - t0) / 1e6;
      out.ticks++;
      const { done, info } = env.step(a);
      if (!done) continue;
      out[info.result]++;
      if (info.b !== undefined) out.miss.push(info.b * 100);
      if (info.result === 'hit') { out.time.push(info.time); out.pathSpeed.push(info.dist0 / Math.max(info.time, 1e-3)); }
      break;
    }
  }
  const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);
  const med = (a) => (a.length ? a.slice().sort((x, y) => x - y)[a.length >> 1] : null);
  return {
    hit_rate: out.hit / episodes, graze_rate: out.graze / episodes, pop_rate: out.pop / episodes, timeout_rate: out.timeout / episodes,
    miss_px_median: med(out.miss), miss_px_mean: mean(out.miss), hit_time: mean(out.time), path_speed: mean(out.pathSpeed),
    planner_share: out.plannerTicks / out.ticks, ms_per_tick: out.ms / out.ticks,
  };
}

const row = { time: Date.now() / 1000, step: json.step, update: json.update, network: play(false), assisted: play(true) };
console.log(JSON.stringify(row, null, 1));
if (args.append && runDir) fs.appendFileSync(path.join(runDir, 'eval.jsonl'), JSON.stringify(row) + '\n');
