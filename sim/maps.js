// maps.js - maps to train on: the real maps in maps/ (PNG + JSON) and random generated ones.
const fs = require('fs');
const path = require('path');
const { PNG } = require('pngjs');
const { loadMap, trimPng } = require('../engine/mapLoader');

const MAP_DIR = path.join(__dirname, '..', 'maps');

function realMapKeys() {
  return fs.readdirSync(MAP_DIR).filter((f) => f.endsWith('.png')).map((f) => f.slice(0, -4))
    .filter((k) => fs.existsSync(path.join(MAP_DIR, k + '.json'))).sort();
}

const cache = {};
function realMap(key) {
  if (!cache[key]) {
    const img = PNG.sync.read(trimPng(fs.readFileSync(path.join(MAP_DIR, key + '.png'))));
    const json = JSON.parse(fs.readFileSync(path.join(MAP_DIR, key + '.json'), 'utf8'));
    cache[key] = { img, json };
  }
  const { img, json } = cache[key];
  const map = loadMap(img, json); // fresh tiles every time: rooms edit them
  map.key = key;
  return map;
}

// small seeded RNG (mulberry32)
function rng(seed) {
  let a = seed >>> 0;
  const f = () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  f.int = (n) => Math.floor(f() * n);
  f.range = (a, b) => a + f() * (b - a);
  return f;
}

// A random arena: border walls, wall blocks and bars (some with 45 degree corners), spikes,
// boosts, team boosts, bombs, team tiles. Sizes around a viewport or two.
function generatedMap(r) {
  const W = 30 + r.int(30), H = 22 + r.int(20);
  const t = [];
  for (let x = 0; x < W; x++) { t[x] = []; for (let y = 0; y < H; y++) t[x][y] = (x === 0 || y === 0 || x === W - 1 || y === H - 1) ? 1 : 2; }
  const inside = (x, y) => x > 0 && y > 0 && x < W - 1 && y < H - 1;
  const area = W * H;
  const density = r.range(0.2, 1.4);
  // wall blocks and bars
  for (let i = 0, n = Math.round(area / 60 * density); i < n; i++) {
    const w = r() < 0.5 ? 1 + r.int(3) : 1 + r.int(8), h = r() < 0.5 ? 1 + r.int(3) : 1 + r.int(8);
    const x0 = 1 + r.int(W - 2), y0 = 1 + r.int(H - 2);
    for (let x = x0; x < x0 + w; x++) for (let y = y0; y < y0 + h; y++) if (inside(x, y)) t[x][y] = 1;
  }
  // 45 degree walls on outside corners of walls next to floor
  for (let x = 1; x < W - 1; x++) for (let y = 1; y < H - 1; y++) {
    if (t[x][y] !== 2 || r() > 0.35) continue;
    const L = t[x - 1][y] === 1, R = t[x + 1][y] === 1, U = t[x][y - 1] === 1, D = t[x][y + 1] === 1;
    if (L && D && !R && !U) t[x][y] = 1.1;
    else if (L && U && !R && !D) t[x][y] = 1.2;
    else if (R && U && !L && !D) t[x][y] = 1.3;
    else if (R && D && !L && !U) t[x][y] = 1.4;
  }
  const floors = () => { const out = []; for (let x = 1; x < W - 1; x++) for (let y = 1; y < H - 1; y++) if (t[x][y] === 2) out.push([x, y]); return out; };
  const put = (v, n) => { const f = floors(); for (let i = 0; i < n && f.length; i++) { const j = r.int(f.length); const [x, y] = f[j]; f.splice(j, 1); t[x][y] = v; } };
  put(7, Math.round(area / 50 * r.range(0, 1.5)));    // spikes
  put(5, Math.round(area / 120 * r.range(0, 2)));     // boosts
  put(14, Math.round(area / 300 * r.range(0, 2)));    // red boosts (the agent is red)
  put(15, Math.round(area / 400 * r.range(0, 1)));    // blue boosts (do nothing for red)
  put(10, Math.round(area / 150 * r.range(0, 2)));    // bombs
  // team tile patches (red speeds the agent up; yellow everyone)
  for (let i = 0, n = r() < 0.4 ? r.int(4) : 0; i < n; i++) {
    const v = r() < 0.7 ? 11 : r() < 0.5 ? 12 : 23;
    const x0 = 1 + r.int(W - 2), y0 = 1 + r.int(H - 2), w = 2 + r.int(6), h = 2 + r.int(6);
    for (let x = x0; x < x0 + w; x++) for (let y = y0; y < y0 + h; y++) if (inside(x, y) && t[x][y] === 2) t[x][y] = v;
  }
  const f = floors();
  const sp = f.length ? f[r.int(f.length)] : [1, 1];
  const sp2 = f.length ? f[r.int(f.length)] : [1, 1];
  return {
    tiles: t, key: 'generated', info: { name: 'Generated', author: 'tpbot' },
    switches: {}, fields: {}, portals: {}, marsballs: [], spawnPoints: { red: [{ x: sp[0], y: sp[1], radius: 0 }], blue: [{ x: sp2[0], y: sp2[1], radius: 0 }] },
    gravity: false,
  };
}

module.exports = { realMapKeys, realMap, generatedMap, rng };
