// server.js - a batch of training environments behind stdin/stdout, driven by train/vecenv.py.
//
//   node sim/server.js --envs 32 --seed 1 [--replay-dir runs/x/replays --replay-every 4] [--assist]
//
// Protocol (binary). In: 'R' (reset all) | 'S' + one action byte per env (step all) | 'Q' (quit).
// Out, after each R or S: u32 length, then grid (envs * GRID uint8), scalars (envs * SCALARS
// float32), rewards (envs float32), dones (envs uint8), u32 json length, json: list of finished
// episodes [{ env, result, b, time, ... }]. Finished environments are reset at once, so the
// observation of a done env is the first one of its next episode.
const fs = require('fs');
const path = require('path');
const G = require('../nav/geometry');
const { Env } = require('./env');
const { plan } = require('../nav/planner');

const args = {};
for (let i = 2; i < process.argv.length; i++) {
  const a = process.argv[i];
  if (a.startsWith('--')) { const next = process.argv[i + 1]; if (next === undefined || next.startsWith('--')) args[a.slice(2)] = true; else { args[a.slice(2)] = next; i++; } }
}
const N = Number(args.envs || 16), seed = Number(args.seed || 1);
const replayDir = args['replay-dir'] || null, replayEvery = Math.max(1, Number(args['replay-every'] || 4));
const assist = !!args.assist;
const envs = [];
for (let i = 0; i < N; i++) envs.push(new Env({ seed: seed * 1000 + i, generatedShare: args['generated-share'] !== undefined ? Number(args['generated-share']) : 0.5 }));
if (replayDir) fs.mkdirSync(replayDir, { recursive: true });

const grid = new Uint8Array(N * G.GRID);
const scal = new Float32Array(N * G.SCALARS);
const rew = new Float32Array(N);
const done = new Uint8Array(N);

function startEpisode(i) {
  const e = envs[i];
  e.count = (e.count || 0) + 1;
  e.record = !!replayDir && i === 0 && e.count % replayEvery === 1 % replayEvery;
  e.reset();
}

function saveReplay(rep) {
  const name = 'replay-' + Date.now() + '.json';
  fs.writeFileSync(path.join(replayDir, name), JSON.stringify(rep));
  const files = fs.readdirSync(replayDir).filter((f) => /^replay-\d+\.json$/.test(f)).sort();
  for (const f of files.slice(0, Math.max(0, files.length - 40))) { try { fs.unlinkSync(path.join(replayDir, f)); } catch (e) { /* gone */ } }
}

function respond(finished) {
  for (let i = 0; i < N; i++) envs[i].observe(grid, i * G.GRID, scal, i * G.SCALARS);
  const json = Buffer.from(JSON.stringify(finished));
  const parts = [Buffer.from(grid.buffer), Buffer.from(scal.buffer), Buffer.from(rew.buffer), Buffer.from(done.buffer)];
  const jl = Buffer.alloc(4); jl.writeUInt32LE(json.length);
  const body = parts.reduce((a, b) => a + b.length, 0) + 4 + json.length;
  const head = Buffer.alloc(4); head.writeUInt32LE(body);
  process.stdout.write(Buffer.concat([head, ...parts, jl, json]));
}

function stepAll(actions) {
  const finished = [];
  for (let i = 0; i < N; i++) {
    const e = envs[i];
    let a = actions[i];
    let assisted = false;
    if (assist) {
      const p = plan(e.me(), e.target, { tiles: e.room.tiles, team: 1 });
      if (p) { a = p.action; assisted = true; }
    }
    const r = e.step(a);
    rew[i] = r.reward; done[i] = r.done ? 1 : 0;
    if (assisted) e.assistedTicks = (e.assistedTicks || 0) + 1;
    if (r.done) {
      r.info.env = i;
      if (assist) { r.info.assisted = e.assistedTicks || 0; e.assistedTicks = 0; }
      finished.push(r.info);
      if (e.replay) { e.replay.result = r.info; saveReplay(e.replay); }
      startEpisode(i);
    }
  }
  return finished;
}

let buf = Buffer.alloc(0);
process.stdin.on('data', (chunk) => {
  buf = Buffer.concat([buf, chunk]);
  for (;;) {
    if (!buf.length) return;
    const cmd = String.fromCharCode(buf[0]);
    if (cmd === 'Q') process.exit(0);
    if (cmd === 'R') {
      buf = buf.subarray(1);
      for (let i = 0; i < N; i++) startEpisode(i);
      rew.fill(0); done.fill(0);
      respond([]);
    } else if (cmd === 'S') {
      if (buf.length < 1 + N) return;
      const actions = buf.subarray(1, 1 + N);
      const finished = stepAll(actions);
      buf = buf.subarray(1 + N);
      respond(finished);
    } else throw new Error('bad command ' + buf[0]);
  }
});
process.stdin.on('end', () => process.exit(0));
