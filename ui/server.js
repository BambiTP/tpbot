// server.js - the training viewer: charts of runs/<run>/metrics.jsonl and eval.jsonl, and
// replays of recent episodes. No dependencies.
//
//   node ui/server.js [--port 8000] [--host 127.0.0.1]     then open http://localhost:8000
const http = require('http');
const fs = require('fs');
const path = require('path');

const args = {};
for (let i = 2; i < process.argv.length; i += 2) args[process.argv[i].replace(/^--/, '')] = process.argv[i + 1];
const PORT = Number(args.port || 8000), HOST = args.host || '127.0.0.1';
const RUNS = path.join(__dirname, '..', 'runs');
const NAME = /^[\w.-]+$/;

function readJsonl(file, maxPoints) {
  if (!fs.existsSync(file)) return [];
  const rows = [];
  for (const line of fs.readFileSync(file, 'utf8').split('\n')) { if (line.trim()) { try { rows.push(JSON.parse(line)); } catch (e) { /* partial line */ } } }
  if (!maxPoints || rows.length <= maxPoints) return rows;
  // average neighbouring updates so long runs stay readable
  const k = Math.ceil(rows.length / maxPoints), out = [];
  for (let i = 0; i < rows.length; i += k) {
    const chunk = rows.slice(i, i + k), o = { ...chunk[chunk.length - 1] };
    for (const key of Object.keys(o)) {
      if (typeof o[key] !== 'number' || key === 'step' || key === 'update' || key === 'time') continue;
      const vals = chunk.map((r) => r[key]).filter((v) => typeof v === 'number');
      o[key] = vals.reduce((a, b) => a + b, 0) / vals.length;
    }
    out.push(o);
  }
  return out;
}

function runInfo(name) {
  const dir = path.join(RUNS, name);
  const m = path.join(dir, 'metrics.jsonl');
  let last = null;
  if (fs.existsSync(m)) {
    const lines = fs.readFileSync(m, 'utf8').trim().split('\n');
    try { last = JSON.parse(lines[lines.length - 1]); } catch (e) { last = null; }
  }
  return { name, last, modified: fs.existsSync(m) ? fs.statSync(m).mtimeMs : fs.statSync(dir).mtimeMs };
}

function send(res, code, body, type = 'application/json') {
  res.writeHead(code, { 'Content-Type': type, 'Cache-Control': 'no-store' });
  res.end(typeof body === 'string' || Buffer.isBuffer(body) ? body : JSON.stringify(body));
}

http.createServer((req, res) => {
  const url = new URL(req.url, 'http://x');
  const run = url.searchParams.get('run');
  if (run !== null && !NAME.test(run)) return send(res, 400, { error: 'bad run' });
  const dir = run && path.join(RUNS, run);
  try {
    switch (url.pathname) {
      case '/': return send(res, 200, fs.readFileSync(path.join(__dirname, 'index.html')), 'text/html; charset=utf-8');
      case '/api/runs': {
        const names = fs.existsSync(RUNS) ? fs.readdirSync(RUNS).filter((n) => NAME.test(n) && fs.statSync(path.join(RUNS, n)).isDirectory()) : [];
        return send(res, 200, names.map(runInfo).sort((a, b) => b.modified - a.modified));
      }
      case '/api/metrics': return send(res, 200, readJsonl(path.join(dir, 'metrics.jsonl'), 1500));
      case '/api/eval': return send(res, 200, readJsonl(path.join(dir, 'eval.jsonl')));
      case '/api/replays': {
        const rd = path.join(dir, 'replays');
        const files = fs.existsSync(rd) ? fs.readdirSync(rd).filter((f) => /^replay-\d+\.json$/.test(f)).sort().reverse() : [];
        return send(res, 200, files);
      }
      case '/api/replay': {
        const f = url.searchParams.get('file') || '';
        if (!/^replay-\d+\.json$/.test(f)) return send(res, 400, { error: 'bad file' });
        return send(res, 200, fs.readFileSync(path.join(dir, 'replays', f)));
      }
      default: return send(res, 404, { error: 'not found' });
    }
  } catch (e) { return send(res, 500, { error: String(e.message || e) }); }
}).listen(PORT, HOST, () => console.log(`training viewer: http://${HOST === '0.0.0.0' ? 'localhost' : HOST}:${PORT}`));
