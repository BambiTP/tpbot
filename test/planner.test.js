// The strike planner, run in the real engine on open floor: still/moving points and a moving
// ball within reach must be hit dead on (measured by the engine run, not the planner's guess).
const assert = require('assert');
const { SimRoom, PH } = require('../sim/simroom');
const G = require('../nav/geometry');
const { plan } = require('../nav/planner');
const { rng } = require('../sim/maps');

const W = 40, H = 30, tiles = [];
for (let x = 0; x < W; x++) { tiles[x] = []; for (let y = 0; y < H; y++) tiles[x][y] = (x === 0 || y === 0 || x === W - 1 || y === H - 1) ? 1 : 2; }
const room = new SimRoom({ tiles, info: {}, switches: {}, fields: {}, portals: {}, marsballs: [], spawnPoints: { red: [{ x: 5, y: 5 }], blue: [{ x: 8, y: 8 }] } });
const me = room.addBall(1), other = room.addBall(2);
const r = rng(11);
const damp = 1 - G.DT * PH.LINEAR_DAMPING;
const res = { hit: 0, miss: 0, none: 0 }, misses = [];
let worstB = 0, ticks = 0, planned = 0, strikes = 0;
const T0 = Date.now();
for (let trial = 0; trial < 150; trial++) {
  const kind = trial % 3; // 0 still point, 1 moving point, 2 ball
  // the situation the planner takes over in: already rolling roughly toward the target
  const ax = 8, ay = 6, d = r.range(0.8, 2), b = r() * 6.283;
  const sp = r.range(1.5, 3), a = b + r.range(-0.5, 0.5);
  room.place(me, ax, ay, Math.cos(a) * sp, Math.sin(a) * sp);
  const t = { x: ax + Math.cos(b) * d, y: ay + Math.sin(b) * d, vx: 0, vy: 0, r: 0, kx: 0, ky: 0 };
  if (kind === 1) { t.vx = r.range(-1, 1); t.vy = r.range(-1, 1); }
  let keys = [0, 0];
  if (kind === 2) {
    keys = G.ACTIONS[r.int(9)];
    room.place(other, t.x, t.y, r.range(-1, 1), r.range(-1, 1));
    other.keys.left = keys[0] < 0; other.keys.right = keys[0] > 0; other.keys.up = keys[1] < 0; other.keys.down = keys[1] > 0;
  } else room.place(other, 30, 20, 0, 0);
  let result = null;
  for (let i = 0; i < 120 && !result; i++) {
    const p = me.body.GetPosition(), v = me.body.GetLinearVelocity();
    if (kind === 2) { const op = other.body.GetPosition(), ov = other.body.GetLinearVelocity(); Object.assign(t, { x: op.x, y: op.y, vx: ov.x, vy: ov.y, r: G.R, kx: keys[0], ky: keys[1], ac: other.ac, ms: other.ms }); }
    ticks++;
    const pl = plan({ x: p.x, y: p.y, vx: v.x, vy: v.y }, t, { tiles: room.tiles, team: 1 });
    // no strike plan yet: head straight for the target (stands in for the network)
    let act = pl && pl.action;
    if (!pl) { const ang = Math.round(Math.atan2(t.y - p.y, t.x - p.x) / (Math.PI / 4)); act = { 0: 3, 1: 4, 2: 5, 3: 6, 4: 7, '-4': 7, '-3': 8, '-2': 1, '-1': 2 }[ang]; } else planned++;
    const [kx, ky] = G.ACTIONS[act];
    me.keys.left = kx < 0; me.keys.right = kx > 0; me.keys.up = ky < 0; me.keys.down = ky > 0;
    const a0 = { x: p.x, y: p.y, vx: v.x, vy: v.y }, t0 = { ...t };
    room.tick1();
    if (kind === 1) { t.x += t.vx * G.DT; t.y += t.vy * G.DT; }
    const tdx = kind === 2 ? t0.vx * damp * G.DT : t.x - t0.x, tdy = kind === 2 ? t0.vy * damp * G.DT : t.y - t0.y;
    const c = G.contact(a0.x, a0.y, a0.vx * damp * G.DT, a0.vy * damp * G.DT, t0.x, t0.y, tdx, tdy, G.R + t0.r);
    if (c) {
      result = c.b <= G.HIT_TOLERANCE ? 'hit' : 'miss';
      // the planner's own strikes must all be dead on
      if (pl) { strikes++; worstB = Math.max(worstB, c.b); if (result === 'miss') misses.push([kind, c.b]); }
    }
  }
  res[result || 'none']++; (res.byKind = res.byKind || {})[kind + (result || 'none')] = (res.byKind[kind + (result || 'none')] || 0) + 1;
}
console.log(res, '|', strikes, 'planner strikes, worst miss', (worstB * 100).toFixed(2), 'px |', ((Date.now() - T0) / ticks).toFixed(2), 'ms per tick');
assert(strikes >= 80 && misses.length === 0, 'every planner strike must be dead on');
console.log('ok');
