// simroom.js - the real game engine (engine/game.js) run headless for training: a virtual clock
// that advances 1/60 s per step, virtual timers instead of setTimeout, and no packets.
// The physics, boosts, bombs, spikes, gates and portals are the engine's own code, untouched.
const { GameRoom, T } = require('../engine/game');
const { PHYSICS: PH, STATES } = require('../engine/constants');
const Box2D = require('../engine/box2d');
const V = Box2D.Common.Math.b2Vec2;

const TICK_MS = 1000 / 60;
const noClient = () => ({ emit() {}, disconnect() {} });

class SimRoom extends GameRoom {
  constructor(map, settings = {}) {
    const clock = { t: 0 };
    super({
      id: 'sim', map, isPrivate: true, now: () => clock.t,
      settings: Object.assign({ disableAllPups: true, noAfkKick: true, time: 1e6 }, settings),
    });
    this.clock = clock;
    this.timerQueue = [];   // { at, fn, seq }, kept sorted by (at, seq)
    this.timerSeq = 0;
    this.readyAt = {};      // "x,y" -> virtual ms when a used boost/bomb/powerup is back
    this.respawnTotal = {}; // "x,y" -> its full respawn time
    this.tileLog = null;    // set to an array to record tile changes (replays)
    // straight into play: no countdown
    this.state = STATES.ACTIVE;
    this.stateEndsAt = Infinity;
    this.startedPlayAt = 0;
  }

  // ---- no network ----
  send() {}
  broadcast() {}
  flushDirty() { if (this.dirty) this.dirty.clear(); }
  snapshot() {}
  afkCheck() {}

  // ---- virtual time ----
  later(ms, fn) {
    const h = { at: this.clock.t + ms, fn, seq: this.timerSeq++, dead: false };
    let i = this.timerQueue.length;
    while (i > 0 && this.timerQueue[i - 1].at > h.at) i--;
    this.timerQueue.splice(i, 0, h);
    return h;
  }

  timedRespawn(x, y, total, finalTile, warnTile) {
    const k = x + ',' + y;
    this.readyAt[k] = this.clock.t + total;
    this.respawnTotal[k] = total;
    super.timedRespawn(x, y, total, finalTile, warnTile);
  }

  setTile(x, y, v, quiet) {
    if (Math.floor(parseFloat(v)) === T.GATE_OFF || Math.floor(parseFloat(this.tiles[x][y])) === T.GATE_OFF) this.walkVersion = (this.walkVersion || 0) + 1;
    super.setTile(x, y, v, quiet);
    if (this.tileLog) this.tileLog.push([x, y, v]);
  }

  // one 1/60 s game tick
  tick1() {
    this.clock.t += TICK_MS;
    const q = this.timerQueue;
    while (q.length && q[0].at <= this.clock.t) { const h = q.shift(); if (!h.dead && !this.closed) h.fn(); }
    this.step();
    this.pendingTiles.length = 0;
  }

  // seconds until a used boost/bomb at (x, y) is back (0 when ready)
  cooldown(x, y) {
    const r = this.readyAt[x + ',' + y];
    return r ? Math.max(0, r - this.clock.t) / 1000 : 0;
  }

  addBall(team) {
    const c = noClient();
    this.addClient(c, { publicId: 'sim' + this.nextPlayerId, name: 'Sim', auth: null }, { team });
    return this.players[c.playerId];
  }

  // place a ball (alive) at (x, y) metres with velocity (vx, vy)
  place(p, x, y, vx = 0, vy = 0) {
    p.spawnGen = (p.spawnGen || 0) + 1; // cancels a pending respawn
    p.dead = false;
    p.body.SetActive(true);
    p.body.SetPosition(new V(x, y));
    p.body.SetLinearVelocity(new V(vx, vy));
    p.body.SetAngularVelocity(0);
    p.body.SetAwake(true);
    p.onPickups = new Set();
    p.portalPending = null;
    p.arrivedOnPortal = null;
    for (const k in p.keys) p.keys[k] = false;
  }

  // put a boost/bomb tile into its "used" state with `remaining` ms left
  coolTile(x, y, remaining) {
    const base = this.tiles[x][y];
    if (typeof base !== 'number') return;
    const f = Math.floor(base);
    if (f === T.BOOST || f === T.RED_BOOST || f === T.BLUE_BOOST) {
      const empty = Number((base + 0.1).toFixed(1));
      this.setTile(x, y, String(empty), true);
      this.timedRespawn(x, y, remaining, base, (i) => Number(empty.toFixed(1) + String(i).padStart(2, '0')));
      this.respawnTotal[x + ',' + y] = this.settings.speedPadRespawnTime;
    } else if (base === T.BOMB) {
      this.setTile(x, y, '10.1', true);
      this.timedRespawn(x, y, remaining, T.BOMB, (i) => Number('10.1' + String(i).padStart(2, '0')));
      this.respawnTotal[x + ',' + y] = this.settings.dynamiteRespawnTime;
    }
  }
}

module.exports = { SimRoom, TICK_MS, PH, T, V };
