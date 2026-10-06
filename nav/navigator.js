// navigator.js - the piece a bot uses: "get my ball's leading point onto this target as fast as
// possible". Each tick, call act() with the game state and press the keys it returns.
//
//   const nav = new Navigator(modelJson);           // runs/<run>/model.json
//   nav.setMap(tiles, team);                        // the 'map' packet's tiles; again on mapupdate
//   const { keys } = nav.act(me, target, nowMs);    // me: { x, y, vx, vy, ac, ms } (metres, m/s)
//                                                   // target: { x, y, vx?, vy?, r?, kx?, ky? }
//
// The network plans the route (walls, spikes, boosts, bombs). When an exact dead-on strike is
// possible within a second of open floor, the strike planner takes over (nav/planner.js).
// Isomorphic: Node or browser (load geometry.js, planner.js, model.js first).
(function () {
const node = typeof module !== 'undefined' && module.exports;
const G = node ? require('./geometry') : globalThis.TPNavGeometry;
const { plan } = node ? require('./planner') : globalThis.TPNavPlanner;
const { NavModel } = node ? require('./model') : globalThis.TPNavModel;

const RESPAWN = { boost: 10, bomb: 30 }; // seconds (group defaults speedPadRespawnTime / dynamiteRespawnTime)

class Navigator {
  // opts: { planner: true, boostRespawn: 10, bombRespawn: 30 }
  constructor(modelJson, opts = {}) {
    this.model = new NavModel(modelJson);
    this.usePlanner = opts.planner !== false;
    this.respawn = { boost: opts.boostRespawn || RESPAWN.boost, bomb: opts.bombRespawn || RESPAWN.bomb };
    this.grid = new Uint8Array(G.GRID);
    this.scal = new Float32Array(G.SCALARS);
    this.usedAt = {}; // "x,y" -> ms when a boost/bomb was seen used
    this.now = 0;
  }

  // tiles: column-major [x][y] as in the 'map' packet. Call again (or tileChanged) on mapupdate.
  setMap(tiles, team) {
    this.tiles = tiles; this.team = team;
    this.mask = G.walkMask(tiles, team);
    this.fieldKey = null;
  }

  // a 'mapupdate' entry; nowMs: when it arrived
  tileChanged(x, y, v, nowMs) {
    const was = this.tiles[x][y];
    this.tiles[x][y] = v;
    const c = G.tileClass(v, this.team), wc = G.tileClass(was, this.team);
    if ((c === 'boostCool' || c === 'bombCool') && wc !== c) this.usedAt[x + ',' + y] = nowMs;
    if (G.walkable(c) !== G.walkable(wc)) { this.mask = G.walkMask(this.tiles, this.team); this.fieldKey = null; }
  }

  view() {
    return {
      tiles: this.tiles, team: this.team, field: this.field,
      cooldown: (x, y) => {
        const kind = Math.floor(parseFloat(this.tiles[x][y])) === 10 ? 'bomb' : 'boost';
        const at = this.usedAt[x + ',' + y];
        // a used tile we never saw being used: guess half way
        return at === undefined ? this.respawn[kind] / 2 : Math.max(0, this.respawn[kind] - (this.now - at) / 1000);
      },
      coolTotal: (x, y) => (Math.floor(parseFloat(this.tiles[x][y])) === 10 ? this.respawn.bomb : this.respawn.boost),
    };
  }

  // Returns { action (0..8), keys: { up, down, left, right }, by: 'planner' | 'network' }
  act(me, target, nowMs = Date.now()) {
    this.now = nowMs;
    const t = Object.assign({ vx: 0, vy: 0, r: 0, kx: 0, ky: 0 }, target);
    const cx = Math.round(t.x / G.TILE), cy = Math.round(t.y / G.TILE), key = cx + ',' + cy;
    if (key !== this.fieldKey) { this.field = G.distanceField(this.tiles, this.team, cx, cy, this.mask); this.fieldKey = key; }
    let action = null, by = 'network';
    if (this.usePlanner) {
      const p = plan(me, t, { tiles: this.tiles, team: this.team });
      if (p) { action = p.action; by = 'planner'; }
    }
    if (action === null) {
      G.observe(this.view(), me, t, this.grid, 0, this.scal, 0);
      action = this.model.act(this.grid, this.scal);
    }
    const [dx, dy] = G.ACTIONS[action];
    return { action, by, keys: { left: dx < 0, right: dx > 0, up: dy < 0, down: dy > 0 } };
  }
}

const api = { Navigator };
if (node) module.exports = api;
else globalThis.TPNavigator = api;
})();
