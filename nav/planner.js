// planner.js - exact short-range strike planner. In open floor the engine's movement is simple and
// deterministic, so this tries key plans ("hold A for n ticks, then hold B") up to a second ahead
// on a copy of that physics and returns the first key of the fastest plan whose leading point
// lands dead on the target. Replanned every tick. Plans that would touch anything but plain floor
// or team tiles (walls, spikes, boosts, bombs, portals, gates...) are skipped: the network handles
// those. Isomorphic: Node or browser (globalThis.TPNavPlanner).
(function () {
const G = typeof module !== 'undefined' && module.exports ? require('./geometry') : globalThis.TPNavGeometry;
const { TILE, R, DT, ACTIONS } = G;
const DAMP = 1 - DT * 0.5;           // Box2D linear damping 0.5, per 1/60 s step
const AC = 0.025, TEAM_AC = 0.012, MS = 2.5, TEAM_MS = 5;
const HOLDS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 19, 22, 26, 30, 35, 40, 46, 52];

// what plans may roll over: 0 = no, 1 = plain floor, 2 = team tile (faster)
function surface(t, team) {
  const c = G.tileClass(t, team);
  if (c === 'floor') return 1;
  if (c === 'teamTile') return 2;
  return 0;
}

// surface codes for every tile (0 no, 1 floor, 2 team tile), index x*H+y
function surfaces(tiles, team, W, H) {
  const s = new Uint8Array(W * H);
  for (let x = 0; x < W; x++) for (let y = 0; y < H; y++) s[x * H + y] = surface(tiles[x][y], team);
  return s;
}

// One tick of the engine for a ball with keys (kx, ky): tile check at the start of the tick sets
// ac/ms, then Box2D damps and moves, then the keys add acceleration (engine/game.js step()).
// s: { x, y, vx, vy }; surf: surfaces(). Returns false when the ball touches a tile plans may not use.
function tick(s, kx, ky, tiles, team, W, H, accel, topspeed, surf) {
  const h = TILE / 2;
  surf = surf || surfaces(tiles, team, W, H);
  const x0 = Math.floor((s.x - R + h) / TILE), x1 = Math.floor((s.x + R + h) / TILE);
  const y0 = Math.floor((s.y - R + h) / TILE), y1 = Math.floor((s.y + R + h) / TILE);
  let team1 = false;
  for (let x = x0; x <= x1; x++) for (let y = y0; y <= y1; y++) {
    if (x < 0 || y < 0 || x >= W || y >= H) return false;
    const k = surf[x * H + y];
    if (k === 1) continue; // plain floor: nothing to check
    const cx = x * TILE, cy = y * TILE;
    const dx = s.x - Math.max(cx - h, Math.min(s.x, cx + h)), dy = s.y - Math.max(cy - h, Math.min(s.y, cy + h));
    const e2 = dx * dx + dy * dy;
    if (e2 >= R * R) continue; // not touching this tile
    if (!k) return false;
    if (e2 < (R - 0.01) * (R - 0.01)) team1 = true;
  }
  const ac = (AC + (team1 ? TEAM_AC : 0)) * accel, ms = (team1 ? TEAM_MS : MS) * topspeed;
  s.vx *= DAMP; s.vy *= DAMP;
  s.x += s.vx * DT; s.y += s.vy * DT;
  if (kx < 0 && s.vx > -ms) s.vx -= ac;
  if (kx > 0 && s.vx < ms) s.vx += ac;
  if (ky < 0 && s.vy > -ms) s.vy -= ac;
  if (ky > 0 && s.vy < ms) s.vy += ac;
  return true;
}

// a target's own motion: a point keeps its velocity; a ball (r > 0) also damps and, when its keys
// are known (kx, ky), accelerates like any ball
function tickTarget(t) {
  if (t.r > 0) {
    t.vx *= DAMP; t.vy *= DAMP;
    t.x += t.vx * DT; t.y += t.vy * DT;
    const ms = t.ms || MS, ac = t.ac || AC;
    if (t.kx < 0 && t.vx > -ms) t.vx -= ac;
    if (t.kx > 0 && t.vx < ms) t.vx += ac;
    if (t.ky < 0 && t.vy > -ms) t.vy -= ac;
    if (t.ky > 0 && t.vy < ms) t.vy += ac;
  } else { t.x += t.vx * DT; t.y += t.vy * DT; }
}

// me: { x, y, vx, vy } (state right after a tick, as the engine reports it); target: { x, y, vx, vy,
// r, kx?, ky? }; world: { tiles, team, accel?, topspeed? }. opts: { horizon (ticks), tolerance (m, the
// most it may miss by), tight (m, preferred: the fastest plan this accurate wins when there is one) }.
// Returns null or { action, ticks (until contact), b (predicted miss distance) }.
function plan(me, target, world, opts = {}) {
  const horizon = opts.horizon || 60, tol = opts.tolerance ?? G.HIT_TOLERANCE, tight = Math.min(tol, opts.tight ?? 0.003);
  const tiles = world.tiles, team = world.team || 1, W = tiles.length, H = tiles[0].length;
  const accel = world.accel || 1, topspeed = world.topspeed || 1;
  // too far to reach within the horizon even flat out?
  const reachV = Math.max(Math.hypot(me.vx, me.vy), 5) + Math.hypot(target.vx, target.vy);
  if (Math.hypot(target.x - me.x, target.y - me.y) - R - target.r > reachV * horizon * DT) return null;
  // the target's path, once
  const tp = [{ x: target.x, y: target.y, vx: target.vx, vy: target.vy }];
  const tt = { ...target };
  for (let i = 0; i < horizon; i++) { tickTarget(tt); tp.push({ x: tt.x, y: tt.y, vx: tt.vx, vy: tt.vy }); }
  const rsum = R + target.r;
  const surf = surfaces(tiles, team, W, H);
  // best: fastest plan within `tight` (sub-pixel); loose: fastest within `tol`, used when no tight one exists
  let best = null, loose = null;
  const s = { x: 0, y: 0, vx: 0, vy: 0 };
  // the first leg's states are shared by every second leg
  for (let a1 = 0; a1 < 9; a1++) {
    const first = [{ x: me.x, y: me.y, vx: me.vx, vy: me.vy, hit: null }];
    Object.assign(s, me);
    let alive = true;
    for (let i = 1; i <= horizon; i++) {
      if (!alive) { first.push(null); continue; }
      const p0 = tp[i - 1];
      const c = G.contact(s.x, s.y, s.vx * DAMP * DT, s.vy * DAMP * DT, p0.x, p0.y, (tp[i].x - p0.x), (tp[i].y - p0.y), rsum);
      if (c) { first.push({ hit: { t: i - 1 + c.s, b: c.b } }); alive = false; continue; }
      alive = tick(s, ACTIONS[a1][0], ACTIONS[a1][1], tiles, team, W, H, accel, topspeed, surf);
      first.push(alive ? { x: s.x, y: s.y, vx: s.vx, vy: s.vy, hit: null } : null);
    }
    // plans: a1 held all the way, or a1 for n ticks then a2
    const consider = (hit) => {
      if (!hit || hit.b > tol) return;
      const better = (cur) => !cur || hit.t < cur.ticks - 1e-9 || (Math.abs(hit.t - cur.ticks) < 1e-9 && hit.b < cur.b);
      if (hit.b <= tight && better(best)) best = { action: a1, ticks: hit.t, b: hit.b };
      if (better(loose)) loose = { action: a1, ticks: hit.t, b: hit.b };
    };
    for (let i = 1; i <= horizon; i++) if (first[i] && first[i].hit) { consider(first[i].hit); break; }
    for (const n of HOLDS) {
      if (n >= horizon || !first[n] || first[n].hit) continue;
      for (let a2 = 0; a2 < 9; a2++) {
        if (a2 === a1) continue;
        Object.assign(s, first[n]);
        for (let i = n + 1; i <= horizon; i++) {
          if (best && i - 1 >= best.ticks) break; // can't beat the best plan any more
          const p0 = tp[i - 1];
          const c = G.contact(s.x, s.y, s.vx * DAMP * DT, s.vy * DAMP * DT, p0.x, p0.y, (tp[i].x - p0.x), (tp[i].y - p0.y), rsum);
          if (c) { consider({ t: i - 1 + c.s, b: c.b }); break; }
          if (!tick(s, ACTIONS[a2][0], ACTIONS[a2][1], tiles, team, W, H, accel, topspeed, surf)) break;
        }
      }
    }
  }
  return best || loose;
}

const api = { plan, tick, tickTarget, surfaces, DAMP };
if (typeof module !== 'undefined' && module.exports) module.exports = api;
else globalThis.TPNavPlanner = api;
})();
