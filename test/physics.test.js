// The planner's copy of the movement physics must match the engine tick for tick in open floor.
const assert = require('assert');
const { SimRoom } = require('../sim/simroom');
const P = require('../nav/planner');
const G = require('../nav/geometry');
const { rng } = require('../sim/maps');

const W = 40, H = 30;
const tiles = [];
for (let x = 0; x < W; x++) { tiles[x] = []; for (let y = 0; y < H; y++) tiles[x][y] = (x === 0 || y === 0 || x === W - 1 || y === H - 1) ? 1 : (x > 25 && y > 18 ? 11 : 2); }
const map = { tiles, info: { name: 'open' }, switches: {}, fields: {}, portals: {}, marsballs: [], spawnPoints: { red: [{ x: 5, y: 5 }], blue: [{ x: 8, y: 8 }] } };
const room = new SimRoom(map);
const ball = room.addBall(1);
const r = rng(7);
let worst = 0;
for (let trial = 0; trial < 200; trial++) {
  const x = r.range(6, 9) , y = r.range(6, 9);
  const vx = r.range(-6, 6), vy = r.range(-6, 6);
  room.place(ball, x, y, vx, vy);
  const s = { x, y, vx, vy };
  let a = r.int(9);
  for (let i = 0; i < 90; i++) {
    if (r() < 0.1) a = r.int(9);
    const [kx, ky] = G.ACTIONS[a];
    ball.keys.left = kx < 0; ball.keys.right = kx > 0; ball.keys.up = ky < 0; ball.keys.down = ky > 0;
    room.tick1();
    if (!P.tick(s, kx, ky, room.tiles, 1, W, H, 1, 1)) break;
    const p = ball.body.GetPosition(), v = ball.body.GetLinearVelocity();
    const e = Math.max(Math.abs(p.x - s.x), Math.abs(p.y - s.y), Math.abs(v.x - s.vx) / 60, Math.abs(v.y - s.vy) / 60);
    worst = Math.max(worst, e);
    if (Math.hypot(p.x - 8, p.y - 8) > 3) break; // stay in open floor, away from the walls
  }
}
console.log('planner vs engine, worst error over 90 ticks:', worst.toExponential(2), 'm');
assert(worst < 1e-4, 'planner physics drifted from the engine');
console.log('ok');
