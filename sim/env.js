// env.js - one training episode at a time: drop the red ball somewhere on a map with a target
// somewhere in its viewport (a still point, a moving point, or a moving blue ball) and score how
// fast and how dead-on its leading point hits it.
//
// A hit: the first moment the ball touches the target (centre distance R for a point, 2R for a
// ball). It is dead on when the target is within HIT_TOLERANCE of the ball's line of travel
// (relative to the target), i.e. the leading point of the ball lands on it.
const G = require('../nav/geometry');
const { SimRoom, T, PH } = require('./simroom');
const { realMapKeys, realMap, generatedMap, rng } = require('./maps');

const MAX_TICKS = 600;           // 10 s
const EPISODES_PER_ROOM = 16;
const HIT_BONUS = 4;             // reward for a perfect hit, in seconds saved
const HIT_SIGMA = 0.03;          // m: the bonus falls off with the miss distance like exp(-(b/sigma)^2)
const POP_PENALTY = 4;
const GAMMA = 0.995;             // for the potential-based shaping (must match the trainer's gamma)
const START_OK = new Set(['floor', 'teamTile']);

class Env {
  // opts: { seed, generatedShare (0..1), maps (keys), kinds: { point, movingPoint, ball } weights }
  constructor(opts = {}) {
    this.r = rng(opts.seed || 1);
    this.generatedShare = opts.generatedShare ?? 0.5;
    this.mapKeys = opts.maps || realMapKeys();
    this.kinds = opts.kinds || { point: 0.5, movingPoint: 0.2, ball: 0.3 };
    this.room = null;
    this.episodesInRoom = 0;
    this.record = false;
    this.replay = null;
  }

  newRoom() {
    if (this.room) this.room.close();
    let map;
    if (!this.mapKeys.length || this.r() < this.generatedShare) map = generatedMap(this.r);
    else map = realMap(this.mapKeys[this.r.int(this.mapKeys.length)]);
    const room = new SimRoom(map, {});
    // a team with no spawn would crash addClient: give it any floor tile
    const floors = [];
    for (let x = 0; x < room.W; x++) for (let y = 0; y < room.H; y++) if (START_OK.has(G.tileClass(room.tiles[x][y], 1))) floors.push({ x, y });
    if (!floors.length) return this.newRoom();
    for (const team of [1, 2]) if (!room.spawnTiles[team].length) room.spawnTiles[team].push({ ...floors[0], radius: 0, weight: 1 });
    this.agent = room.addBall(1);
    this.other = room.addBall(2);
    this.hideOther();
    // some boosts and bombs start used, at a random point of their respawn
    for (let x = 0; x < room.W; x++) for (let y = 0; y < room.H; y++) {
      const t = room.tiles[x][y], f = Math.floor(t);
      if (typeof t !== 'number' || this.r() > 0.35) continue;
      if (f === T.BOOST || f === T.RED_BOOST || f === T.BLUE_BOOST) room.coolTile(x, y, this.r() * room.settings.speedPadRespawnTime);
      else if (t === T.BOMB) room.coolTile(x, y, this.r() * room.settings.dynamiteRespawnTime);
    }
    this.room = room;
    this.floors = floors;
    this.episodesInRoom = 0;
    this.view = {
      tiles: room.tiles, team: 1, field: null,
      cooldown: (x, y) => room.cooldown(x, y),
      coolTotal: (x, y) => (room.respawnTotal[x + ',' + y] || 10000) / 1000,
    };
  }

  hideOther() {
    const o = this.other;
    o.spawnGen = (o.spawnGen || 0) + 1;
    o.dead = true;
    o.body.SetActive(false);
    o.body.SetPosition(new (require('../engine/box2d').Common.Math.b2Vec2)(-100, -100));
  }

  reset() {
    if (!this.room || this.episodesInRoom >= EPISODES_PER_ROOM || this.r() < 0.02) this.newRoom();
    this.episodesInRoom++;
    const r = this.r, room = this.room;
    for (let tries = 0; ; tries++) {
      if (tries > 40) { this.newRoom(); tries = 0; }
      const s = this.floors[r.int(this.floors.length)];
      // where can the ball get to from here (also a check that it isn't boxed in)
      const reach = G.distanceField(room.tiles, 1, s.x, s.y, this.mask());
      const cands = [];
      for (let dx = -G.HALF_W; dx <= G.HALF_W; dx++) for (let dy = -G.HALF_H; dy <= G.HALF_H; dy++) {
        const x = s.x + dx, y = s.y + dy;
        if (x < 0 || y < 0 || x >= room.W || y >= room.H || Math.abs(dx) + Math.abs(dy) < 2) continue;
        if (reach[x * room.H + y] === Infinity || !START_OK.has(G.tileClass(room.tiles[x][y], 1))) continue;
        cands.push({ x, y });
      }
      if (!cands.length) continue;
      const tc = cands[r.int(cands.length)];
      this.start(s, tc);
      return;
    }
  }

  start(s, tc) {
    const r = this.r, room = this.room;
    const ax = s.x * G.TILE + r.range(-0.08, 0.08), ay = s.y * G.TILE + r.range(-0.08, 0.08);
    let speed = r() < 0.3 ? 0 : r() < 0.85 ? r.range(0, 2.5) : r.range(2.5, 6);
    const ang = r() * Math.PI * 2;
    room.place(this.agent, ax, ay, Math.cos(ang) * speed, Math.sin(ang) * speed);
    this.hideOther();
    const w = this.kinds, total = w.point + w.movingPoint + w.ball;
    const pick = r() * total;
    this.kind = pick < w.point ? 'point' : pick < w.point + w.movingPoint ? 'movingPoint' : 'ball';
    const px = tc.x * G.TILE + r.range(-0.18, 0.18), py = tc.y * G.TILE + r.range(-0.18, 0.18);
    this.target = { x: px, y: py, vx: 0, vy: 0, r: 0 };
    if (this.kind === 'movingPoint') {
      speed = r.range(0.3, 3); const a2 = r() * Math.PI * 2;
      this.target.vx = Math.cos(a2) * speed; this.target.vy = Math.sin(a2) * speed;
    } else if (this.kind === 'ball') {
      speed = r() < 0.3 ? 0 : r.range(0, 3); const a2 = r() * Math.PI * 2;
      room.place(this.other, tc.x * G.TILE, tc.y * G.TILE, Math.cos(a2) * speed, Math.sin(a2) * speed);
      this.otherAction = r() < 0.3 ? 0 : 1 + r.int(8);
      this.syncBallTarget();
    }
    this.fieldCell = null;
    this.updateField();
    this.ticks = 0;
    this.phi = this.potential();
    this.dist0 = G.fieldAt(this.view.field, room.W, room.H, ax, ay) * G.TILE;
    this.straight0 = Math.hypot(this.target.x - ax, this.target.y - ay);
    this.ret = 0;
    if (this.record) {
      room.tileLog = [];
      this.replay = {
        map: room.map.key || 'generated', W: room.W, H: room.H, tiles: room.tiles.map((c) => c.map((t) => String(t))),
        kind: this.kind, targetR: this.target.r, frames: [], tileLog: [],
      };
      this.recordFrame(-1);
    } else { room.tileLog = null; this.replay = null; }
  }

  syncBallTarget() {
    const b = this.other.body, p = b.GetPosition(), v = b.GetLinearVelocity();
    this.target.x = p.x; this.target.y = p.y; this.target.vx = v.x; this.target.vy = v.y; this.target.r = G.R;
    [this.target.kx, this.target.ky] = G.ACTIONS[this.otherAction];
    this.target.ac = this.other.ac; this.target.ms = this.other.ms;
  }

  updateField() {
    this.mask();
    const cx = Math.round(this.target.x / G.TILE), cy = Math.round(this.target.y / G.TILE);
    const key = cx + ',' + cy;
    if (key === this.fieldCell) return;
    this.fieldCell = key;
    this.view.field = G.distanceField(this.room.tiles, 1, cx, cy, this.mask());
  }

  mask() {
    const v = this.room.walkVersion || 0;
    if (this.maskRoom !== this.room || this.maskVersion !== v) {
      this.maskRoom = this.room; this.maskVersion = v; this.fieldCell = null;
      this.walk = G.walkMask(this.room.tiles, 1);
    }
    return this.walk;
  }

  potential() {
    const p = this.agent.body.GetPosition();
    const d = G.fieldAt(this.view.field, this.room.W, this.room.H, p.x, p.y);
    const m = d === Infinity ? 40 : d * G.TILE;
    return -m / 2.5; // seconds at top speed
  }

  me() {
    const b = this.agent.body, p = b.GetPosition(), v = b.GetLinearVelocity();
    return { x: p.x, y: p.y, vx: v.x, vy: v.y, ac: this.agent.ac, ms: this.agent.ms };
  }

  observe(grid, gOff, scal, sOff) { G.observe(this.view, this.me(), this.target, grid, gOff, scal, sOff); }

  setKeys(p, a) {
    const [dx, dy] = G.ACTIONS[a];
    p.keys.left = dx < 0; p.keys.right = dx > 0; p.keys.up = dy < 0; p.keys.down = dy > 0;
  }

  // returns { reward, done, info }
  step(action) {
    const room = this.room, r = this.r;
    this.setKeys(this.agent, action);
    if (this.kind === 'ball') {
      if (r() < 1 / 30) this.otherAction = r() < 0.2 ? 0 : 1 + r.int(8);
      this.setKeys(this.other, this.otherAction);
      [this.target.kx, this.target.ky] = G.ACTIONS[this.otherAction];
    }
    const a0 = this.me(), t0 = { ...this.target };
    room.tick1();
    this.ticks++;
    const damp = 1 - G.DT * PH.LINEAR_DAMPING;
    if (this.kind === 'movingPoint') {
      // straight line, bouncing off anything the ball couldn't go through
      let nx = this.target.x + this.target.vx * G.DT, ny = this.target.y + this.target.vy * G.DT;
      const blocked = (x, y) => { const ix = Math.round(x / G.TILE), iy = Math.round(y / G.TILE); return ix < 0 || iy < 0 || ix >= room.W || iy >= room.H || !G.walkable(G.tileClass(room.tiles[ix][iy], 1)); };
      if (blocked(nx, this.target.y)) { this.target.vx = -this.target.vx; nx = this.target.x; }
      if (blocked(this.target.x, ny)) { this.target.vy = -this.target.vy; ny = this.target.y; }
      this.target.x = nx; this.target.y = ny;
    } else if (this.kind === 'ball') {
      if (this.other.dead) this.hideOther();
      else this.syncBallTarget();
    }
    let reward = -G.DT, done = false, info = null;
    if (this.agent.dead) {
      reward -= POP_PENALTY; done = true; info = { result: 'pop' };
    } else {
      // contact this tick: straight-line motion from the pre-step state (a ball-on-ball hit has
      // already bounced in the post-step positions)
      const rsum = G.R + t0.r;
      // a ball target's step also bounced; a point target moved exactly this much
      const tdx = this.kind === 'ball' ? t0.vx * damp * G.DT : this.target.x - t0.x;
      const tdy = this.kind === 'ball' ? t0.vy * damp * G.DT : this.target.y - t0.y;
      const c = G.contact(a0.x, a0.y, a0.vx * damp * G.DT, a0.vy * damp * G.DT, t0.x, t0.y, tdx, tdy, rsum);
      const now = this.me();
      let hit = c;
      if (!hit && Math.hypot(now.x - this.target.x, now.y - this.target.y) <= rsum + 0.002) {
        hit = G.contact(a0.x, a0.y, now.x - a0.x, now.y - a0.y, t0.x, t0.y, this.target.x - t0.x, this.target.y - t0.y, rsum) || { s: 1, b: rsum };
      }
      if (hit) {
        done = true;
        const good = hit.b <= G.HIT_TOLERANCE;
        reward += HIT_BONUS * Math.exp(-((hit.b / HIT_SIGMA) ** 2));
        info = { result: good ? 'hit' : 'graze', b: hit.b, time: (this.ticks - 1 + hit.s) * G.DT, speed: Math.hypot(a0.vx - t0.vx, a0.vy - t0.vy) };
      } else if (this.kind === 'ball' && this.other.dead) {
        done = true; info = { result: 'lost' }; // the target popped on a spike: nobody's fault
      } else if (this.ticks >= MAX_TICKS) {
        done = true; info = { result: 'timeout' };
      }
    }
    if (!done) {
      this.updateField();
      const phi = this.potential();
      reward += GAMMA * phi - this.phi;
      this.phi = phi;
    } else reward += -this.phi; // terminal potential is 0
    this.ret += reward;
    if (this.replay) this.recordFrame(action);
    if (done) {
      Object.assign(info, { kind: this.kind, ticks: this.ticks, ret: this.ret, dist0: this.dist0, straight0: this.straight0, map: room.map.key || 'generated' });
      if (this.replay) { this.replay.result = info; this.replay.tileLog = room.tileLog.map((e) => [e[0], e[1], String(e[2])]); }
    }
    return { reward, done, info };
  }

  recordFrame(action) {
    const m = this.me(), t = this.target;
    const r3 = (v) => Math.round(v * 1000) / 1000;
    this.replay.frames.push([r3(m.x), r3(m.y), r3(m.vx), r3(m.vy), r3(t.x), r3(t.y), action, this.room.tileLog.length]);
  }
}

module.exports = { Env, MAX_TICKS, GAMMA, HIT_BONUS };
