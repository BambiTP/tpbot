# tpbot

A TagPro ball that gets its **leading point** (the spot on its rim facing the way it's moving)
**dead on** a target anywhere in its viewport, as fast as possible. The target can be a still
point, a moving point, or a moving ball. Uses for it: lining the ball up in front of a boost,
crossing the map in intermission, or hitting an enemy ball head on.

There are two parts:

- **A neural network** that plans the route: walls, spikes, gates, boosts, bombs, team tiles,
  portals. It's trained with PPO in Python against the **real game engine** (a copy of
  tagpro-local's `engine/`, run headless in Node), so it learns the game's actual physics.
- **An exact strike planner** (`nav/planner.js`) for the final stretch. In open floor the
  engine's movement is deterministic, so it tries key plans on a copy of that physics, up to
  a second ahead, and takes the fastest one that lands dead on. Its copy matches the engine
  with zero error (`test/physics.test.js`), and its strikes land within 2 px, usually under 1.

Both run in plain JavaScript (`nav/`), with no libraries, in Node or the browser.

## What counts as a hit

The first contact between the ball and the target (a point: centre distance 0.19 m; a ball:
0.38 m) is the end of an episode. It's a **hit** when the target is within **2 px** of the
ball's line of travel (relative to the target), i.e. the leading point landed on it.
Otherwise it's a **graze**. Training rewards a hit by up to 4 s worth, falling off with the
miss distance, and every tick costs 1/60 s, so the network learns to be both fast and exact.
Popping on a spike or gate costs 4 s. Episodes time out after 10 s.

## Setup (Linux, once)

These are for Ubuntu or Debian. You need Node.js 18+ and Python 3.10+.

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip curl
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs

git clone https://github.com/BambiTP/tpbot.git
cd tpbot
npm install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm test
```

`pip install -r requirements.txt` gets PyTorch with NVIDIA GPU support (used automatically when
there's a GPU). With no NVIDIA GPU, this smaller download works instead:
`pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install numpy`.

## Train

Terminal 1, from the `tpbot` folder:

```bash
source .venv/bin/activate
python train/ppo.py --run first
```

Terminal 2, the viewer:

```bash
node ui/server.js
```

Then open <http://localhost:8000>. It shows hit rate, miss distance, time to hit, path speed, pops,
hit rate by target type, eval results, and replays of recent episodes on the map. It refreshes
every 5 seconds.

- **Stop:** Ctrl+C. **Resume:** run the same command again (it continues from `runs/first/ckpt.pt`).
- **A fresh run:** use a new name, e.g. `--run second`.
- **Speed:** it uses one simulator process per CPU core minus one (`--workers`), 32 games each
  (`--envs`). Watch "steps/s" in the viewer. A few hundred million steps (`--steps`, default 3e8)
  is a realistic amount of training. Expect it to take a while, from hours to days depending on
  the machine.
- **Viewer from another computer:** `node ui/server.js --host 0.0.0.0`, then open
  `http://<this machine's IP>:8000`.

Everything for a run goes in `runs/<name>/`: `metrics.jsonl`, `eval.jsonl`, `ckpt.pt`,
`model.json` (the trained weights for JavaScript, rewritten every 20 updates), `replays/`.

## Check a trained model

```bash
node sim/eval.js --run first --episodes 300
node test/model.test.js runs/first/model.json
```

`sim/eval.js` plays 300 fixed test episodes greedily, network-only and network + planner.
`test/model.test.js` checks the JavaScript network gives the same output as PyTorch.

## Use it in a bot

```js
const { Navigator } = require('./nav/navigator');      // browser: load geometry.js, planner.js, model.js, navigator.js
const nav = new Navigator(require('./runs/first/model.json'));
nav.setMap(mapPacket.tiles, myTeam);                   // the 'map' packet (column-major tiles[x][y])
nav.tileChanged(x, y, value, Date.now());              // for each 'mapupdate' entry

// every game tick (1/60 s), positions in metres (rx, ry) and velocities in m/s (lx, ly):
const { keys } = nav.act(
  { x: me.rx, y: me.ry, vx: me.lx, vy: me.ly, ac: me.ac, ms: me.ms },
  { x: tx, y: ty },                                    // a point; for a ball add vx, vy, r: 0.19, kx, ky (its held keys, -1/0/1)
);
// press keys.up / keys.down / keys.left / keys.right
```

It needs to run every tick. The current bots in tagpro-offline decide every 50 ms, which is too
coarse for exact hits. The network takes about 2–4 ms per tick in Node. The planner adds up to
about 3 ms, but only when a strike is in reach.

## Layout

| Path | What |
|---|---|
| `engine/` | The game engine, copied from tagpro-local (see `engine/SOURCE.md`) |
| `sim/simroom.js` | The engine headless: virtual clock and timers, no packets |
| `sim/maps.js` | Real maps (`maps/`) and random generated arenas |
| `sim/env.js` | One episode: placement, targets, hit test, rewards, replays |
| `sim/server.js` | A batch of episodes behind stdin/stdout for Python |
| `sim/eval.js` | Greedy evaluation of a `model.json` |
| `train/` | PPO trainer, network, environment pool (Python) |
| `nav/geometry.js` | What the network sees (a 33×21 tile viewport, 16 channels + 15 numbers) and the hit test |
| `nav/planner.js` | The exact strike planner |
| `nav/model.js` | The network's forward pass in JavaScript |
| `nav/navigator.js` | The API a bot uses |
| `ui/` | The training viewer |
