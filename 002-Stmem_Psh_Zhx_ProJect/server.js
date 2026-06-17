const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const PORT = 8080;
const BATCH_HOST = '127.0.0.1';
const BATCH_PORT = 8000;
// 远端 FastGS host（origin/psh batch_service）：跑 3DGS 训练
// 形式：host:port，例如 192.168.1.20:8000；未设置则 /fastgs/* 返 503
const FASTGS_HOST_FULL = process.env.STMEM_FASTGS_HOST || '';
const [FASTGS_HOST, FASTGS_PORT] = FASTGS_HOST_FULL.includes(':')
  ? FASTGS_HOST_FULL.split(':')
  : [FASTGS_HOST_FULL, '8000'];
const WEB_DIR = path.join(__dirname, 'src', 'web');
const CERTS_DIR = path.join(__dirname, 'certs');
const DATA_DIR = process.env.STMEM_DATA_DIR || path.join(__dirname, 'data');

const MIME = {
  '.html': 'text/html', '.css': 'text/css', '.js': 'application/javascript',
  '.json': 'application/json', '.png': 'image/png', '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml', '.ico': 'image/x-icon', '.wasm': 'application/wasm',
  '.woff2': 'font/woff2', '.mp4': 'video/mp4', '.webm': 'video/webm',
  '.ply': 'application/octet-stream', '.splat': 'application/octet-stream',
};

function json(res, code, obj) {
  res.writeHead(code, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(obj));
}

function getQuery(req) {
  const u = new URL(req.url, 'http://localhost');
  const q = {};
  for (const [k, v] of u.searchParams) q[k] = v;
  return q;
}

function serveStatic(req, res) {
  let p = new URL(req.url, 'http://localhost').pathname;
  try { p = decodeURIComponent(p); } catch (_) {}
  if (p === '/') p = '/index.html';
  const fp = path.join(WEB_DIR, p);
  if (!fp.startsWith(WEB_DIR)) { res.writeHead(403); return res.end(); }
  const ext = path.extname(fp);
  fs.readFile(fp, (err, data) => {
    if (err) { res.writeHead(404); return res.end('Not found'); }
    res.writeHead(200, { 'Content-Type': MIME[ext] || 'application/octet-stream', 'Cache-Control': 'no-cache' });
    res.end(data);
  });
}

function proxyToBatch(req, res) {
  const opts = {
    hostname: BATCH_HOST, port: BATCH_PORT,
    path: req.url, method: req.method,
    headers: { ...req.headers, host: `${BATCH_HOST}:${BATCH_PORT}` },
    timeout: 300000,
  };
  const preq = http.request(opts, (pres) => {
    res.writeHead(pres.statusCode, pres.headers);
    pres.pipe(res);
  });
  preq.on('error', () => { res.writeHead(502); res.end('Batch service unavailable'); });
  req.pipe(preq);
}

// /fastgs/{id}/{frame|finish|ply} → 远端 origin/psh batch_service 的 /batch/{id}/{fastgs_frame|finish_inference|fastgs_ply}
// 走环境变量 STMEM_FASTGS_HOST=host:port（未设置则 503）
function proxyToFastgs(req, res) {
  if (!FASTGS_HOST) {
    res.writeHead(503, { 'Content-Type': 'application/json' });
    return res.end('{"error":"STMEM_FASTGS_HOST not set"}');
  }
  // /fastgs/{id}/frame  → /batch/{id}/fastgs_frame
  // /fastgs/{id}/finish → /batch/{id}/finish_inference
  // /fastgs/{id}/ply    → /batch/{id}/fastgs_ply
  const u = new URL(req.url, 'http://x');
  const parts = u.pathname.split('/');  // ['', 'fastgs', '{id}', '{op}']
  const batchId = parts[2];
  const op = parts[3];
  if (!batchId || !op) {
    res.writeHead(400); return res.end('Bad fastgs path');
  }
  const opMap = { frame: 'fastgs_frame', finish: 'finish_inference', ply: 'fastgs_ply', logs: 'logs', status: 'status' };
  const remoteOp = opMap[op];
  if (!remoteOp) { res.writeHead(404); return res.end('Unknown fastgs op'); }
  const remotePath = '/batch/' + batchId + '/' + remoteOp + (u.search || '');
  const opts = {
    hostname: FASTGS_HOST, port: Number(FASTGS_PORT),
    path: remotePath, method: req.method,
    headers: { ...req.headers, host: `${FASTGS_HOST}:${FASTGS_PORT}` },
    timeout: 600000,  // 训练触发可能慢
  };
  const preq = http.request(opts, (pres) => {
    res.writeHead(pres.statusCode, pres.headers);
    pres.pipe(res);
  });
  preq.on('error', (e) => {
    res.writeHead(502, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'FastGS host unreachable', detail: String(e.message || e) }));
  });
  req.pipe(preq);
}

function serveFrame(req, res) {
  const fname = new URL(req.url, 'http://localhost').pathname.split('/').pop();
  if (fs.existsSync(DATA_DIR)) {
    const dirs = fs.readdirSync(DATA_DIR).filter(d => d.startsWith('batch_')).sort().reverse();
    for (const dir of dirs) {
      for (const sub of ['frames', 'rgb']) {
        const fp = path.join(DATA_DIR, dir, sub, fname);
        if (fs.existsSync(fp)) {
          const ct = fname.endsWith('.jpg') ? 'image/jpeg' : 'image/png';
          res.writeHead(200, { 'Content-Type': ct, 'Cache-Control': 'no-cache' });
          return fs.createReadStream(fp).pipe(res);
        }
      }
    }
  }
  res.writeHead(404); res.end('Frame not found');
}

function handleRequest(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  if (req.method === 'OPTIONS') {
    res.writeHead(204, { 'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS', 'Access-Control-Allow-Headers': 'Content-Type' });
    return res.end();
  }
  const p = new URL(req.url, 'http://localhost').pathname;

  if (p.startsWith('/batch/')) return proxyToBatch(req, res);
  if (p.startsWith('/vlm/')) return proxyToBatch(req, res);
  if (p.startsWith('/fastgs/')) return proxyToFastgs(req, res);

  // Dashboard-compatible API endpoints
  if (p === '/api/status') return json(res, 200, { config: {}, isCapturing: false, faceApiReady: false, sourceName: 'orin' });
  if (p === '/api/frames') return json(res, 200, []);
  if (p === '/api/latest-frame') { res.writeHead(204); return res.end(); }
  if (p.startsWith('/frames/')) return serveFrame(req, res);

  // ========== GS 高斯点云 API ==========

  // POST /api/gaussian-splats/generate?batch_id=xxx&frames_dir=xxx
  if (p === '/api/gaussian-splats/generate') {
    const q = getQuery(req);
    const batchId = q.batch_id || ('batch_' + Date.now());
    const framesDir = q.frames_dir || path.join(DATA_DIR, 'frames');
    const gsDir = path.join(DATA_DIR, 'gaussian-splats');
    const scriptPath = path.join(__dirname, 'scripts', 'generate-gs.sh');

    if (!fs.existsSync(scriptPath)) {
      return json(res, 500, { error: 'generate-gs.sh not found', path: scriptPath });
    }

    const child = spawn('bash', [scriptPath, batchId, framesDir, gsDir], {
      detached: true,
      stdio: 'ignore',
    });
    child.unref();

    return json(res, 202, {
      batch_id: batchId,
      status_url: '/api/gaussian-splats/status?batch_id=' + batchId,
      ply_url: '/api/gaussian-splats/' + batchId + '.ply',
      message: 'GS generation started',
    });
  }

  // GET /api/gaussian-splats/status?batch_id=xxx
  if (p === '/api/gaussian-splats/status') {
    const q = getQuery(req);
    const batchId = q.batch_id;
    if (!batchId) return json(res, 400, { error: 'Missing batch_id' });
    const statusPath = path.join(DATA_DIR, 'gaussian-splats', batchId + '_status.json');
    if (fs.existsSync(statusPath)) {
      res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-cache' });
      return fs.createReadStream(statusPath).pipe(res);
    }
    return json(res, 404, { error: 'Status not found: ' + batchId });
  }

  // GET /api/gaussian-splats/list
  if (p === '/api/gaussian-splats/list') {
    const gsDir = path.join(DATA_DIR, 'gaussian-splats');
    if (!fs.existsSync(gsDir)) return json(res, 200, []);
    const files = fs.readdirSync(gsDir)
      .filter(f => f.endsWith('.ply'))
      .map(f => ({
        name: f,
        url: '/api/gaussian-splats/' + f,
        size: fs.statSync(path.join(gsDir, f)).size,
      }));
    return json(res, 200, files);
  }

  // GET /api/gaussian-splats/demo
  if (p === '/api/gaussian-splats/demo') {
    const demoPath = path.join(DATA_DIR, 'gaussian-splats', 'lingbot.ply');
    if (fs.existsSync(demoPath)) {
      res.writeHead(200, { 'Content-Type': 'application/octet-stream' });
      return fs.createReadStream(demoPath).pipe(res);
    }
    return json(res, 404, { error: 'Demo not found' });
  }

  // GET /api/gaussian-splats/:filename (must be after specific routes)
  if (p.startsWith('/api/gaussian-splats/')) {
    const fname = p.split('/').pop();
    if (!fname || ['demo', 'list', 'status', 'generate'].includes(fname))
      return json(res, 404, { error: 'Not found' });
    const fp = path.join(DATA_DIR, 'gaussian-splats', fname);
    if (fs.existsSync(fp)) {
      res.writeHead(200, { 'Content-Type': 'application/octet-stream' });
      return fs.createReadStream(fp).pipe(res);
    }
    return json(res, 404, { error: 'Not found' });
  }

  serveStatic(req, res);
}

const ssl = {
  key: fs.readFileSync(path.join(CERTS_DIR, 'key.pem')),
  cert: fs.readFileSync(path.join(CERTS_DIR, 'cert.pem')),
};
https.createServer(ssl, handleRequest).listen(PORT, () => {
  console.log(`Dashboard on https://0.0.0.0:${PORT}`);
});
