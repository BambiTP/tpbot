"""Training environments: a pool of Node processes (sim/server.js), each running a batch of
games of the real engine. All workers step in parallel."""
import json
import os
import shutil
import struct
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# must match nav/geometry.js
CHANNELS = 16
VIEW_H, VIEW_W = 21, 33
GRID = CHANNELS * VIEW_H * VIEW_W
SCALARS = 15
ACTIONS = 9


def _node():
    node = shutil.which("node")
    if not node:
        raise SystemExit("node (Node.js 18+) is not on PATH")
    return node


class Worker:
    def __init__(self, envs, seed, extra=()):
        self.n = envs
        cmd = [_node(), os.path.join(ROOT, "sim", "server.js"), "--envs", str(envs), "--seed", str(seed), *extra]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=ROOT, bufsize=0)

    def send(self, data: bytes):
        self.p.stdin.write(data)
        self.p.stdin.flush()

    def _read(self, n):
        out = bytearray()
        while len(out) < n:
            chunk = self.p.stdout.read(n - len(out))
            if not chunk:
                raise RuntimeError("simulator process exited (code %s)" % self.p.poll())
            out += chunk
        return bytes(out)

    def recv(self):
        (length,) = struct.unpack("<I", self._read(4))
        body = self._read(length)
        n, o = self.n, 0
        grid = np.frombuffer(body, np.uint8, n * GRID, o).reshape(n, CHANNELS, VIEW_H, VIEW_W); o += n * GRID
        scal = np.frombuffer(body, np.float32, n * SCALARS, o).reshape(n, SCALARS); o += n * SCALARS * 4
        rew = np.frombuffer(body, np.float32, n, o); o += n * 4
        done = np.frombuffer(body, np.uint8, n, o); o += n
        (jl,) = struct.unpack_from("<I", body, o); o += 4
        infos = json.loads(body[o:o + jl].decode())
        return grid, scal, rew, done, infos

    def close(self):
        try:
            self.send(b"Q")
        except Exception:
            pass
        try:
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


class VecEnv:
    """workers x envs_per_worker games. The first worker saves replays when replay_dir is set."""

    def __init__(self, workers, envs_per_worker, seed=1, replay_dir=None, replay_every=50, extra=()):
        self.workers = []
        for i in range(workers):
            args = list(extra)
            if replay_dir and i == 0:
                args += ["--replay-dir", replay_dir, "--replay-every", str(replay_every)]
            self.workers.append(Worker(envs_per_worker, seed * 100 + i, args))
        self.num_envs = workers * envs_per_worker

    def _gather(self):
        parts = [w.recv() for w in self.workers]
        grid = np.concatenate([p[0] for p in parts])
        scal = np.concatenate([p[1] for p in parts])
        rew = np.concatenate([p[2] for p in parts])
        done = np.concatenate([p[3] for p in parts])
        infos, base = [], 0
        for w, p in zip(self.workers, parts):
            for info in p[4]:
                info["env"] += base
                infos.append(info)
            base += w.n
        return grid, scal, rew, done, infos

    def reset(self):
        for w in self.workers:
            w.send(b"R")
        grid, scal, _, _, _ = self._gather()
        return grid, scal

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.uint8)
        o = 0
        for w in self.workers:
            w.send(b"S" + actions[o:o + w.n].tobytes())
            o += w.n
        return self._gather()

    def close(self):
        for w in self.workers:
            w.close()
