// geometry.js - shared by training (sim/) and the in-game runtime: tile classes, the path
// distance field, the observation the network sees, and the "dead-on" contact test.
// Isomorphic: Node (require) or browser / web worker (globalThis.TPNavGeometry).
(function () {
const TILE = 0.4, R = 0.19, DT = 1 / 60;
const VIEW_W = 33, VIEW_H = 21;   // tiles: the 1280x800 viewport is 32x20 tiles; odd so the ball's tile is the centre
const HALF_W = 16, HALF_H = 10;
const CHANNELS = ['solid', 'diag1', 'diag2', 'diag3', 'diag4', 'spike', 'gate', 'boost', 'boostCool',
  'bomb', 'bombCool', 'teamTile', 'portal', 'well', 'target', 'dist'];
const C = CHANNELS.length;
const GRID = C * VIEW_H * VIEW_W;
const SCALARS = 15;
const ACTIONS = [ // [dx, dy] key directions: 0 = no keys
  [0, 0], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1],
];
const HIT_TOLERANCE = 0.02; // m (2 px): the target must be within this of the ball's line of travel

const num = (t) => (typeof t === 'string' ? parseFloat(t) : t);

// what a tile means for a ball of `team` (1 red, 2 blue)
function tileClass(t, team) {
  const v = num(t), f = Math.floor(v);
  if (v === 0 || v === 1) return 'solid';
  if (f === 1) return 'diag' + Math.round((v - 1) * 10); // 1.1 .. 1.4
  if (v === 7) return 'spike';
  if (v === 9.1 || (v === 9.2 && team === 2) || (v === 9.3 && team === 1)) return 'gate';
  if (f === 5 || (f === 14 && team === 1) || (f === 15 && team === 2)) return t === f ? 'boost' : 'boostCool';
  if (f === 10) return t === 10 ? 'bomb' : 'bombCool';
  if ((v === 11 && team === 1) || (v === 12 && team === 2) || v === 23) return 'teamTile';
  if ((f === 13 || (f === 24 && team === 1) || (f === 25 && team === 2)) && t === f) return 'portal';
  if (v === 22) return 'well';
  return 'floor';
}

// can the path search go through this tile
function walkable(cls) { return !(cls === 'solid' || cls === 'spike' || cls === 'gate' || cls.startsWith('diag')); }

// Path distance (in tiles) from every tile to the target tile, 8-connected without cutting
// corners. tiles: column-major [x][y]. Returns Float32Array W*H (index x*H+y), Infinity = unreachable.
// walkable mask (Uint8Array W*H, index x*H+y); cache it while the tiles don't change
function walkMask(tiles, team) {
  const W = tiles.length, H = tiles[0].length, ok = new Uint8Array(W * H);
  for (let x = 0; x < W; x++) for (let y = 0; y < H; y++) ok[x * H + y] = walkable(tileClass(tiles[x][y], team)) ? 1 : 0;
  return ok;
}

// Dial's algorithm with integer step costs 5 (straight) and 7 (diagonal): fast and within 1% of
// true distances. mask: optional walkMask (computed when missing).
const NB = [[1, 0, 5], [-1, 0, 5], [0, 1, 5], [0, -1, 5], [1, 1, 7], [1, -1, 7], [-1, 1, 7], [-1, -1, 7]];
function distanceField(tiles, team, tx, ty, mask) {
  const W = tiles.length, H = tiles[0].length;
  const ok = mask || walkMask(tiles, team);
  const n = W * H, di = new Int32Array(n).fill(0x7fffffff);
  const d = new Float32Array(n).fill(Infinity);
  if (tx < 0 || ty < 0 || tx >= W || ty >= H) return d;
  const buckets = [[tx * H + ty]];
  di[tx * H + ty] = 0;
  for (let c = 0, left = 1; left > 0; c++) {
    const b = buckets[c];
    if (!b) continue;
    buckets[c] = null;
    for (let i = 0; i < b.length; i++) {
      left--;
      const k = b[i];
      if (di[k] !== c) continue;
      const x = (k / H) | 0, y = k - x * H;
      for (let j = 0; j < 8; j++) {
        const dx = NB[j][0], dy = NB[j][1], nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
        const nk = nx * H + ny;
        if (!ok[nk]) continue;
        if (dx && dy && (!ok[nx * H + y] || !ok[x * H + ny])) continue;
        const nd = c + NB[j][2];
        if (nd < di[nk]) { di[nk] = nd; (buckets[nd] || (buckets[nd] = [])).push(nk); left++; }
      }
    }
  }
  for (let k = 0; k < n; k++) if (di[k] !== 0x7fffffff) d[k] = di[k] / 5;
  return d;
}

// path distance (tiles) at a point in metres, interpolated between tile centres
function fieldAt(field, W, H, x, y) {
  const gx = x / TILE, gy = y / TILE, x0 = Math.floor(gx), y0 = Math.floor(gy), fx = gx - x0, fy = gy - y0;
  let s = 0, w = 0, best = Infinity;
  for (const [ix, iy, wt] of [[x0, y0, (1 - fx) * (1 - fy)], [x0 + 1, y0, fx * (1 - fy)], [x0, y0 + 1, (1 - fx) * fy], [x0 + 1, y0 + 1, fx * fy]]) {
    if (ix < 0 || iy < 0 || ix >= W || iy >= H) continue;
    const v = field[ix * H + iy];
    if (v === Infinity) continue;
    s += v * wt; w += wt; best = Math.min(best, v);
  }
  if (w < 1e-6) return Infinity;
  return w > 0.5 ? s / w : best + 1;
}

// The network's input. view: { tiles, team, cooldown(x, y) -> seconds left, coolTotal(x, y) -> seconds,
//   field (distanceField of the target) }. me: { x, y, vx, vy, ac, ms } (metres, m/s).
// target: { x, y, vx, vy, r, kx?, ky? } (r: 0 for a point, R for a ball; kx, ky: its held keys). Writes grid (Uint8Array GRID) and
// scalars (Float32Array SCALARS) at the given offsets.
function observe(view, me, target, grid, gOff, scal, sOff) {
  const tiles = view.tiles, W = tiles.length, H = tiles[0].length, team = view.team;
  const cx = Math.round(me.x / TILE), cy = Math.round(me.y / TILE);
  const plane = VIEW_W * VIEW_H;
  grid.fill(0, gOff, gOff + GRID);
  const ch = {};
  CHANNELS.forEach((n, i) => { ch[n] = gOff + i * plane; });
  for (let j = 0; j < VIEW_H; j++) for (let i = 0; i < VIEW_W; i++) {
    const x = cx - HALF_W + i, y = cy - HALF_H + j, cell = j * VIEW_W + i;
    if (x < 0 || y < 0 || x >= W || y >= H) { grid[ch.solid + cell] = 255; grid[ch.dist + cell] = 255; continue; }
    const t = tiles[x][y], cls = tileClass(t, team);
    if (cls === 'boostCool' || cls === 'bombCool') {
      const left = view.cooldown(x, y), total = view.coolTotal(x, y) || 1;
      grid[ch[cls] + cell] = Math.max(1, Math.round(255 * Math.min(1, left / total)));
    } else if (cls !== 'floor') grid[ch[cls] + cell] = 255;
    const d = view.field[x * H + y];
    grid[ch.dist + cell] = d === Infinity ? 255 : Math.min(254, Math.round(d * 4));
  }
  // target: bilinear splat at its exact position
  const gx = target.x / TILE - (cx - HALF_W), gy = target.y / TILE - (cy - HALF_H);
  const x0 = Math.floor(gx), y0 = Math.floor(gy), fx = gx - x0, fy = gy - y0;
  for (const [ix, iy, wt] of [[x0, y0, (1 - fx) * (1 - fy)], [x0 + 1, y0, fx * (1 - fy)], [x0, y0 + 1, (1 - fx) * fy], [x0 + 1, y0 + 1, fx * fy]]) {
    if (ix < 0 || iy < 0 || ix >= VIEW_W || iy >= VIEW_H) continue;
    grid[ch.target + iy * VIEW_W + ix] = Math.round(255 * wt);
  }
  const dx = target.x - me.x, dy = target.y - me.y;
  const here = fieldAt(view.field, W, H, me.x, me.y);
  const s = scal, o = sOff;
  s[o] = me.x / TILE - cx; s[o + 1] = me.y / TILE - cy;
  s[o + 2] = me.vx / 5; s[o + 3] = me.vy / 5;
  s[o + 4] = (me.ac - 0.025) / 0.012; s[o + 5] = (me.ms - 2.5) / 2.5;
  s[o + 6] = dx / 6.4; s[o + 7] = dy / 4;
  s[o + 8] = target.vx / 5; s[o + 9] = target.vy / 5;
  s[o + 10] = target.r / R;
  s[o + 11] = here === Infinity ? 2 : Math.min(2, here / 32);
  s[o + 12] = Math.hypot(dx, dy) / 8;
  s[o + 13] = target.kx || 0; s[o + 14] = target.ky || 0; // a ball target's held keys (-1, 0, 1)
}

// First contact during one tick, moving in straight lines: ball at (ax, ay) moving (adx, ady) this
// tick, target at (tx, ty) moving (tdx, tdy), contact at centre distance rsum. Returns null or
// { s: fraction of the tick, b: miss distance of the target from the ball's line of travel
// (relative motion); 0 = dead on }.
function contact(ax, ay, adx, ady, tx, ty, tdx, tdy, rsum) {
  const px = ax - tx, py = ay - ty, ux = adx - tdx, uy = ady - tdy;
  const uu = ux * ux + uy * uy, pp = px * px + py * py;
  const ul = Math.sqrt(uu);
  const b = ul > 1e-9 ? Math.abs(ux * py - uy * px) / ul : Math.sqrt(pp);
  if (pp <= rsum * rsum) return { s: 0, b: Math.min(b, rsum) };
  if (uu < 1e-12) return null;
  const pu = px * ux + py * uy;
  if (pu >= 0) return null; // moving apart
  const disc = pu * pu - uu * (pp - rsum * rsum);
  if (disc < 0) return null;
  const s = (-pu - Math.sqrt(disc)) / uu;
  if (s < 0 || s > 1) return null;
  return { s, b };
}

const api = { TILE, R, DT, VIEW_W, VIEW_H, HALF_W, HALF_H, CHANNELS, C, GRID, SCALARS, ACTIONS, HIT_TOLERANCE,
  tileClass, walkable, walkMask, distanceField, fieldAt, observe, contact };
if (typeof module !== 'undefined' && module.exports) module.exports = api;
else globalThis.TPNavGeometry = api;
})();
