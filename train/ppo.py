"""Train the navigation network with PPO against the real engine.

    python train/ppo.py --run first            # start (or resume) runs/first
    python train/ppo.py --run first --steps 2e8 --workers 12
    python train/ppo.py --run first --viewer 100.x.y.z     # also serve the viewer on that address

Writes to runs/<run>/: metrics.jsonl (one line per update, for the UI), ckpt.pt (resume),
model.json (weights for nav/model.js, every --save-every updates), replays/ (episodes for the UI),
eval.jsonl (greedy play of the latest model.json, with and without the strike planner).
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Net, export_json, reference_io  # noqa: E402
from vecenv import ROOT, VecEnv  # noqa: E402

GAMMA = 0.995  # must match sim/env.js GAMMA (reward shaping)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="default", help="name: everything goes to runs/<run>/")
    p.add_argument("--steps", type=float, default=3e8, help="total environment steps (1/60 s each)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="simulator processes")
    p.add_argument("--envs", type=int, default=32, help="games per simulator process")
    p.add_argument("--rollout", type=int, default=128, help="steps per env per update")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lambda", dest="lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.01, help="entropy bonus at the start (decays to a tenth)")
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--save-every", type=int, default=20, help="updates between checkpoints / model.json")
    p.add_argument("--eval-every", type=int, default=100, help="updates between evals (0: never)")
    p.add_argument("--replay-every", type=int, default=4, help="episodes of game 0 between saved replays")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--viewer", metavar="HOST", help="also start the training viewer on this address (e.g. your Tailscale IP)")
    p.add_argument("--viewer-port", type=int, default=8000)
    return p.parse_args()


class Stats:
    def __init__(self):
        self.eps = []

    def add(self, infos):
        self.eps.extend(infos)

    def summary(self):
        e, out = self.eps, {"episodes": len(self.eps)}
        if not e:
            return out
        res = [x["result"] for x in e]
        for r in ("hit", "graze", "pop", "timeout", "lost"):
            out[r + "_rate"] = res.count(r) / len(e)
        contacts = [x for x in e if x["result"] in ("hit", "graze")]
        if contacts:
            b = np.array([x["b"] for x in contacts])
            out["miss_px_mean"] = float(b.mean() * 100)
            out["miss_px_median"] = float(np.median(b) * 100)
        hits = [x for x in e if x["result"] == "hit"]
        if hits:
            out["hit_time"] = float(np.mean([x["time"] for x in hits]))
            out["hit_speed"] = float(np.mean([x["speed"] for x in hits]))
            # average speed along the shortest path: higher is a faster route
            out["path_speed"] = float(np.mean([x["dist0"] / max(x["time"], 1e-3) for x in hits]))
        for kind in ("point", "movingPoint", "ball"):
            k = [x for x in e if x["kind"] == kind]
            if k:
                out["hit_rate_" + kind] = sum(x["result"] == "hit" for x in k) / len(k)
        out["return"] = float(np.mean([x["ret"] for x in e]))
        self.eps = []
        return out


def main():
    a = parse()
    run_dir = os.path.join(ROOT, "runs", a.run)
    os.makedirs(run_dir, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)
    step, update = 0, 0
    ckpt_path = os.path.join(run_dir, "ckpt.pt")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        step, update = ck["step"], ck["update"]
        print(f"resumed {a.run} at step {step:,} (update {update})")
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(a), f, indent=1)

    viewer = None
    if a.viewer:
        viewer = subprocess.Popen(["node", os.path.join(ROOT, "ui", "server.js"), "--host", a.viewer, "--port", str(a.viewer_port)], cwd=ROOT)
        print(f"viewer: http://{a.viewer}:{a.viewer_port}")
    env = VecEnv(a.workers, a.envs, seed=a.seed + update, replay_dir=os.path.join(run_dir, "replays"), replay_every=a.replay_every)
    N, T = env.num_envs, a.rollout
    print(f"{a.workers} simulators x {a.envs} games = {N} games, {N * T:,} steps per update, device {dev}")
    grid, scal = env.reset()
    grid_t = torch.from_numpy(grid.copy()).to(dev)
    scal_t = torch.from_numpy(scal.copy()).to(dev)

    b_grid = torch.zeros((T, N) + grid.shape[1:], dtype=torch.uint8, device=dev)
    b_scal = torch.zeros((T, N, scal.shape[1]), device=dev)
    b_act = torch.zeros((T, N), dtype=torch.long, device=dev)
    b_logp = torch.zeros((T, N), device=dev)
    b_val = torch.zeros((T, N), device=dev)
    b_rew = torch.zeros((T, N), device=dev)
    b_done = torch.zeros((T, N), device=dev)
    stats = Stats()
    metrics = open(os.path.join(run_dir, "metrics.jsonl"), "a")
    eval_proc = None
    t_start, s_start = time.time(), step

    while step < a.steps:
        frac = min(1.0, step / a.steps)
        lr = a.lr * (1 - 0.9 * frac)
        ent_coef = a.ent * (1 - 0.9 * frac)
        for g in opt.param_groups:
            g["lr"] = lr
        t0 = time.time()
        net.eval()
        for t in range(T):
            with torch.no_grad():
                logits, v = net(grid_t, scal_t)
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()
            b_grid[t] = grid_t
            b_scal[t] = scal_t
            b_act[t] = act
            b_logp[t] = dist.log_prob(act)
            b_val[t] = v
            grid, scal, rew, done, infos = env.step(act.cpu().numpy())
            stats.add(infos)
            b_rew[t] = torch.from_numpy(rew.copy()).to(dev)
            b_done[t] = torch.from_numpy(done.astype(np.float32)).to(dev)
            grid_t = torch.from_numpy(grid.copy()).to(dev)
            scal_t = torch.from_numpy(scal.copy()).to(dev)
        step += N * T
        t_sim = time.time() - t0

        # GAE; a done step's next state belongs to a new episode
        with torch.no_grad():
            _, next_v = net(grid_t, scal_t)
        adv = torch.zeros_like(b_rew)
        last = torch.zeros(N, device=dev)
        for t in reversed(range(T)):
            nv = next_v if t == T - 1 else b_val[t + 1]
            nonterm = 1 - b_done[t]
            delta = b_rew[t] + GAMMA * nv * nonterm - b_val[t]
            last = delta + GAMMA * a.lam * nonterm * last
            adv[t] = last
        ret = adv + b_val

        net.train()
        B = N * T
        fg, fs = b_grid.reshape((B,) + b_grid.shape[2:]), b_scal.reshape(B, -1)
        fa, flp, fadv, fret, fval = b_act.reshape(B), b_logp.reshape(B), adv.reshape(B), ret.reshape(B), b_val.reshape(B)
        mb = B // a.minibatches
        info = {"pl": 0.0, "vl": 0.0, "ent": 0.0, "kl": 0.0, "clip": 0.0}
        n_mb = 0
        for _ in range(a.epochs):
            perm = torch.randperm(B, device=dev)
            for i in range(0, B, mb):
                idx = perm[i:i + mb]
                logits, v = net(fg[idx], fs[idx])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(fa[idx])
                ratio = (lp - flp[idx]).exp()
                ad = fadv[idx]
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                pl = -torch.min(ratio * ad, ratio.clamp(1 - a.clip, 1 + a.clip) * ad).mean()
                v_clip = fval[idx] + (v - fval[idx]).clamp(-a.clip * 5, a.clip * 5)
                vl = 0.5 * torch.max((v - fret[idx]) ** 2, (v_clip - fret[idx]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pl + a.vf * vl - ent_coef * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    info["pl"] += pl.item(); info["vl"] += vl.item(); info["ent"] += ent.item()
                    info["kl"] += ((ratio - 1) - (lp - flp[idx])).mean().item()
                    info["clip"] += ((ratio - 1).abs() > a.clip).float().mean().item()
                n_mb += 1
        update += 1
        for k in info:
            info[k] /= n_mb

        elapsed = time.time() - t_start
        row = {"update": update, "step": step, "time": time.time(), "fps": (step - s_start) / max(elapsed, 1e-6),
               "sim_s": t_sim, "lr": lr, "entropy": info["ent"], "policy_loss": info["pl"], "value_loss": info["vl"],
               "approx_kl": info["kl"], "clip_frac": info["clip"], **stats.summary()}
        metrics.write(json.dumps(row) + "\n")
        metrics.flush()
        print(f"upd {update:5d} step {step:>12,} fps {row['fps']:7.0f} | hit {row.get('hit_rate', 0):.2f} "
              f"graze {row.get('graze_rate', 0):.2f} pop {row.get('pop_rate', 0):.2f} timeout {row.get('timeout_rate', 0):.2f} "
              f"| miss {row.get('miss_px_median', float('nan')):.1f}px time {row.get('hit_time', float('nan')):.2f}s "
              f"| ent {info['ent']:.2f}", flush=True)

        if update % a.save_every == 0 or step >= a.steps:
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "step": step, "update": update}, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
            out = export_json(net)
            out.update({"step": step, "update": update, "reference": reference_io(net)})
            with open(os.path.join(run_dir, "model.json.tmp"), "w") as f:
                json.dump(out, f)
            os.replace(os.path.join(run_dir, "model.json.tmp"), os.path.join(run_dir, "model.json"))
        if a.eval_every and update % a.eval_every == 0 and (eval_proc is None or eval_proc.poll() is not None):
            # in the background: greedy play of the latest model.json, appended to eval.jsonl
            eval_proc = subprocess.Popen(["node", os.path.join(ROOT, "sim", "eval.js"), "--run", a.run, "--episodes", "100",
                                          "--append"], cwd=ROOT, stdout=subprocess.DEVNULL)
    env.close()
    if viewer:
        viewer.terminate()


if __name__ == "__main__":
    main()
