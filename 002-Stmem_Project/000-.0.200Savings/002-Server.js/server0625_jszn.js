const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');

const PORT = 8080;
const BATCH_HOST = '127.0.0.1';
const BATCH_PORT = 8000;
const WEB_DIR = path.join(__dirname, 'src', 'web');
const CERTS_DIR = path.join(__dirname, 'certs');
const DATA_DIR = process.env.STMEM_DATA_DIR || path.join(__dirname, 'data');

const MIME = {
  '.html': 'text/html', '.css': 'text/css', '.js': 'application/javascript',
  '.json': 'application/json', '.png': 'image/png', '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml', '.ico': 'image/x-icon', '.wasm': 'application/wasm',
  '.woff2': 'font/woff2', '.mp4': 'video/mp4', '.webm': 'video/webm',
};

function json(res, code, obj) {
  res.writeHead(code, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(obj));
}

function serveStatic(req, res) {
  let p = new URL(req.url, 'http://localhost').pathname;
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

// Serve frame images from data directory
function serveFrame(req, res) {
  const fname = new URL(req.url, 'http://localhost').pathname.split('/').pop();
  // Search recent batch dirs for the frame
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

  // Dashboard-compatible API endpoints
  if (p === '/api/status') return json(res, 200, { config: {}, isCapturing: false, faceApiReady: false, sourceName: 'orin' });
  if (p === '/api/frames') return json(res, 200, []);
  if (p === '/api/latest-frame') { res.writeHead(204); return res.end(); }
  if (p.startsWith('/frames/')) return serveFrame(req, res);

  serveStatic(req, res);
}

const ssl = {
  key: fs.readFileSync(path.join(CERTS_DIR, 'key.pem')),
  cert: fs.readFileSync(path.join(CERTS_DIR, 'cert.pem')),
};
https.createServer(ssl, handleRequest).listen(PORT, () => {
  console.log(`Dashboard on https://0.0.0.0:${PORT}`);
});
