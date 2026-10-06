// model.js - runs the trained network (runs/<run>/model.json from train/ppo.py) in plain
// JavaScript: no libraries, Node or browser (globalThis.TPNavModel). Same forward pass as
// train/model.py: two stride-2 3x3 convolutions, then two dense layers, then policy logits.
(function () {
function decode(b64) {
  let bytes;
  if (typeof Buffer !== 'undefined') bytes = Buffer.from(b64, 'base64');
  else { const s = atob(b64); bytes = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i); }
  const out = new Float32Array(bytes.length / 4);
  new Uint8Array(out.buffer).set(bytes);
  return out;
}

// The inputs are mostly zeros (one-hot tile channels, ReLU outputs), so both layer kinds work as
// "for each non-zero input, add its row of the transposed weights": contiguous and skips zeros.

// x: Float32Array [cin, h, w] -> [cout, ho, wo], then ReLU. L.wt: weights as [cin*k*k, cout].
function conv(x, cin, h, w, L) {
  const [cout, , k] = L.shape, s = L.stride, p = L.pad;
  const ho = Math.floor((h + 2 * p - k) / s) + 1, wo = Math.floor((w + 2 * p - k) / s) + 1;
  const acc = new Float32Array(cout), Wt = L.wt, B = L.bf;
  const y = new Float32Array(cout * ho * wo), plane = ho * wo;
  for (let oy = 0; oy < ho; oy++) for (let ox = 0; ox < wo; ox++) {
    acc.set(B);
    for (let c = 0; c < cin; c++) {
      const xc = c * h * w;
      for (let ky = 0; ky < k; ky++) {
        const iy = oy * s - p + ky;
        if (iy < 0 || iy >= h) continue;
        for (let kx = 0; kx < k; kx++) {
          const ix = ox * s - p + kx;
          if (ix < 0 || ix >= w) continue;
          const v = x[xc + iy * w + ix];
          if (v === 0) continue;
          const r = ((c * k + ky) * k + kx) * cout;
          for (let o = 0; o < cout; o++) acc[o] += v * Wt[r + o];
        }
      }
    }
    const pos = oy * wo + ox;
    for (let o = 0; o < cout; o++) y[o * plane + pos] = acc[o] > 0 ? acc[o] : 0;
  }
  return { y, h: ho, w: wo };
}

// L.wt: weights as [nin, nout]
function dense(x, L, relu) {
  const [nout, nin] = L.shape, Wt = L.wt, y = Float32Array.from(L.bf);
  for (let i = 0; i < nin; i++) {
    const v = x[i];
    if (v === 0) continue;
    const r = i * nout;
    for (let o = 0; o < nout; o++) y[o] += v * Wt[r + o];
  }
  if (relu) for (let o = 0; o < nout; o++) if (y[o] < 0) y[o] = 0;
  return y;
}

// [out, rest...] -> [rest, out]
function transpose(w, nout) {
  const nin = w.length / nout, t = new Float32Array(w.length);
  for (let o = 0; o < nout; o++) for (let i = 0; i < nin; i++) t[i * nout + o] = w[o * nin + i];
  return t;
}

class NavModel {
  constructor(json) {
    if (json.format !== 'tpbot-nav-1') throw new Error('not a tpbot model.json');
    this.json = json;
    this.L = {};
    for (const l of json.layers) this.L[l.name] = Object.assign({}, l, { wt: transpose(decode(l.w), l.shape[0]), bf: decode(l.b) });
    this.C = json.channels; [this.H, this.W] = json.view; this.S = json.scalars;
  }

  // grid: Uint8Array C*H*W (0..255), scalars: Float32Array S. Returns { logits, value }.
  forward(grid, scalars, gOff = 0, sOff = 0) {
    const n = this.C * this.H * this.W, x = new Float32Array(n);
    for (let i = 0; i < n; i++) x[i] = grid[gOff + i] / 255;
    let a = conv(x, this.C, this.H, this.W, this.L.conv1);
    a = conv(a.y, this.L.conv1.shape[0], a.h, a.w, this.L.conv2);
    const flat = new Float32Array(a.y.length + this.S);
    flat.set(a.y); for (let i = 0; i < this.S; i++) flat[a.y.length + i] = scalars[sOff + i];
    const h1 = dense(flat, this.L.fc1, true), h2 = dense(h1, this.L.fc2, true);
    return { logits: dense(h2, this.L.pi, false), value: dense(h2, this.L.v, false)[0] };
  }

  // the most likely action (0..8, see geometry.ACTIONS)
  act(grid, scalars, gOff = 0, sOff = 0) {
    const { logits } = this.forward(grid, scalars, gOff, sOff);
    let best = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }
}

const api = { NavModel };
if (typeof module !== 'undefined' && module.exports) module.exports = api;
else globalThis.TPNavModel = api;
})();
