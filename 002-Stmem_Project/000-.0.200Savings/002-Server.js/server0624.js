const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');
const { spawn, execSync } = require('child_process');

const DATA_DIR = process.env.STMEM_DATA_DIR || path.join(__dirname, 'data');

// ── 记忆聚合层（SQLite + batch 数据） ──
let _memdb = null;
function memdb() {
  if (_memdb) return _memdb;
  const dbPath = path.join(DATA_DIR, 'memory.sqlite');
  if (fs.existsSync(dbPath)) {
    try { _memdb = require('better-sqlite3')(dbPath); } catch (_) {}
  }
  return _memdb;
}

function scanBatchDirs() {
  // 返回所有包含真实帧文件的 batch 目录路径
  const dirs = [];
  if (!fs.existsSync(DATA_DIR)) return dirs;
  for (const d of fs.readdirSync(DATA_DIR)) {
    const dp = path.join(DATA_DIR, d);
    // 跳过符号链接（如 latest -> jszn）
    if (fs.lstatSync(dp).isSymbolicLink()) continue;
    const st = fs.statSync(dp);
    if (!st.isDirectory()) continue;
    // 统计 frames 数量（只有实际帧文件才算）
    const framesDir = path.join(dp, 'frames');
    let frameCount = 0;
    if (fs.existsSync(framesDir)) {
      try { frameCount = fs.readdirSync(framesDir).filter(f => f.match(/\.(jpg|jpeg|png)$/i)).length; } catch (_) {}
    }
    if (frameCount > 0) dirs.push(dp);
  }
  return dirs;
}

function readJsonSafe(fp) {
  try { return JSON.parse(fs.readFileSync(fp, 'utf8')); } catch (_) { return null; }
}

function aggregateBatchStats() {
  let totalFrames = 0, processedFrames = 0, totalPoints = 0, batchCount = 0;
  for (const d of scanBatchDirs()) {
    const s = readJsonSafe(path.join(d, 'status.json'));
    if (s) {
      batchCount++;
      totalFrames += s.total_frames || 0;
      processedFrames += s.processed_frames || 0;
      totalPoints += s.total_points || 0;
    }
  }
  // 也统计记忆 DB
  const db = memdb();
  let memoryCount = 0, faceCount = 0, archivedCount = 0;
  if (db) {
    try {
      memoryCount = db.prepare('SELECT COUNT(*) as c FROM memories').get().c;
      faceCount = db.prepare('SELECT COUNT(*) as c FROM faces').get().c;
      archivedCount = db.prepare('SELECT COUNT(*) as c FROM archived_memories').get().c;
    } catch (_) {}
  }
  return { totalFrames, processedFrames, totalPoints, batchCount, memoryCount, faceCount, archivedCount };
}

function gsList() {
  const gsDir = path.join(DATA_DIR, 'gaussian-splats');
  if (!fs.existsSync(gsDir)) return [];
  return fs.readdirSync(gsDir).filter(f => f.endsWith('.ply')).map(f => ({
    name: f, url: '/api/gaussian-splats/' + f,
    size: fs.statSync(path.join(gsDir, f)).size,
  }));
}

// ── 中英文类名翻译表 ──
const CLS_ZH = {
  'chair': '椅子', 'office chair': '办公椅', 'table': '桌子', 'tv': '电视',
  'potted plant': '盆栽', 'flower': '花', 'basket': '篮子', 'clothes': '衣物',
  'cabinet': '柜子', 'box': '盒子', 'book': '书', 'remote control': '遥控器',
  'cup': '杯子', 'water bottle': '水瓶', 'pen': '笔', 'computer tower': '电脑主机',
  'cart': '推车', 'case of water bottles': '水瓶箱', 'desk': '书桌', 'fan': '风扇',
  'keyboard': '键盘', 'mouse': '鼠标', 'laptop': '笔记本电脑', 'plant': '植物',
  'bag': '包', 'umbrella': '雨伞', 'storage bin': '收纳箱', 'bottle': '瓶子',
  'tissue pack': '纸巾包', 'marker': '马克笔', 'red marker': '红马克笔', 'pencil': '铅笔',
  'film roll': '胶卷', 'thermos': '保温杯', 'power strip': '排插', 'scissors': '剪刀',
  'cat': '猫', 'cellphone': '手机', 'tree': '树', 'person': '人', 'dog': '狗',
  'phone': '手机', 'door': '门', 'object': '物体', 'fish': '鱼',
  'outlet': '插座', 'power adapter': '电源适配器', 'hat': '帽子', 'laptop': '笔记本电脑', 'tv': '电视', 'monitor': '显示器', 'can': '易拉罐', 'desk': '书桌', 'keyboard': '键盘',
};
function clsZh(name) { return CLS_ZH[name] || name; }

const PORT = 8080;
const BATCH_HOST = '127.0.0.1';
const BATCH_PORT = 8000;
let _gsGenerating = {};  // { batch_id: true } — prevent duplicate GS spawns per process
// 远端 FastGS host（origin/psh batch_service）：跑 3DGS 训练
// 形式：host:port，例如 192.168.1.20:8000；未设置则 /fastgs/* 返 503
const FASTGS_HOST_FULL = process.env.STMEM_FASTGS_HOST || '';
const [FASTGS_HOST, FASTGS_PORT] = FASTGS_HOST_FULL.includes(':')
  ? FASTGS_HOST_FULL.split(':')
  : [FASTGS_HOST_FULL, '8000'];
const WEB_DIR = path.join(__dirname, 'src', 'web');
const CERTS_DIR = path.join(__dirname, 'certs');

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

function serveCrop(req, res) {
  // Serve object crop images from datazhx
  const fname = new URL(req.url, 'http://localhost').pathname.split('/').pop();
  let cropPath = null; const batchDirs = fs.readdirSync(DATA_DIR).filter(x => !x.startsWith('.') && fs.lstatSync(path.join(DATA_DIR,x)).isDirectory()); for (const bd of batchDirs) { const tp = path.join(DATA_DIR, bd, 'stream', 'objects_img_crop', fname); if (fs.existsSync(tp)) { cropPath = tp; break; } }
  if (!fs.existsSync(cropPath)) cropPath = path.join(DATA_DIR, d, 'stream', 'objects_img_crop', fname);
  if (fs.existsSync(cropPath)) {
    const ct = fname.endsWith('.jpg') ? 'image/jpeg' : 'image/png';
    res.writeHead(200, { 'Content-Type': ct, 'Cache-Control': 'no-cache' });
    return fs.createReadStream(cropPath).pipe(res);
  }
  res.writeHead(404); res.end('Crop not found');
}

function serveFrame(req, res) {
  const fname = new URL(req.url, 'http://localhost').pathname.split('/').pop();
  if (fs.existsSync(DATA_DIR)) {
    const dirs = fs.readdirSync(DATA_DIR).filter(d => d.startsWith('batch_') || d === d).sort().reverse();
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

  // GET /batch/{id}/dgsg_objects → 代理到 Python DGSG 实时缓存
  const dgsgObjMatch = p.match(/^\/batch\/([^/]+)\/dgsg_objects$/);
  if (dgsgObjMatch && req.method === 'GET') {
    return proxyToBatch(req, res);
  }

  if (p.startsWith('/batch/')) return proxyToBatch(req, res);
  if (p.startsWith('/vlm/')) return proxyToBatch(req, res);
  if (p.startsWith('/fastgs/')) return proxyToFastgs(req, res);

  // Dashboard-compatible API endpoints
  if (p === '/api/status') return json(res, 200, { config: {}, isCapturing: false, faceApiReady: false, sourceName: 'orin' });
  if (p === '/api/frames') return json(res, 200, []);
  if (p === '/api/latest-frame') { res.writeHead(204); return res.end(); }
  if (p.startsWith('/crops/')) { return serveCrop(req, res); }
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

    // 防止重复启动同一批次的 GS 生成
    const statusPath = path.join(gsDir, batchId + '_status.json');
    if (_gsGenerating[batchId] || (fs.existsSync(statusPath) && JSON.parse(fs.readFileSync(statusPath, 'utf8')).status === 'generating')) {
      return json(res, 202, {
        batch_id: batchId,
        status: 'already_running',
        status_url: '/api/gaussian-splats/status?batch_id=' + batchId,
        message: 'GS generation already in progress',
      });
    }
    _gsGenerating[batchId] = true;

    const child = spawn('bash', [scriptPath, batchId, framesDir, gsDir], {
      detached: true,
      stdio: 'ignore',
    });
    child.unref();
    child.on('exit', () => { delete _gsGenerating[batchId]; });

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
      const stat = fs.statSync(demoPath);
      res.writeHead(200, {
        'Content-Type': 'application/octet-stream',
        'Content-Length': stat.size,
      });
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
      const stat = fs.statSync(fp);
      res.writeHead(200, {
        'Content-Type': 'application/octet-stream',
        'Content-Length': stat.size,
      });
      return fs.createReadStream(fp).pipe(res);
    }
    return json(res, 404, { error: 'Not found' });
  }

  // ========== 记忆聚合 API (New_Stmem.html 数据源) ==========

  // ── GET /api/memory/summary ── 概览页统计
  if (p === '/api/memory/summary') {
    const s = aggregateBatchStats();
    const gs = gsList();
    return json(res, 200, {
      totalDuration: { hours: (s.processedFrames / 3600).toFixed(1), delta: null },
      keyframeCount:  { count: s.processedFrames, annotated: s.memoryCount, delta: s.memoryCount > 0 ? `VLM 已标注 ${Math.round(s.memoryCount / Math.max(1, s.processedFrames) * 100)}%` : null },
      spaceCount:     { count: s.batchCount, delta: s.batchCount > 0 ? `+${s.batchCount} 个空间` : null },
      eventCount:     { count: s.memoryCount + s.archivedCount, delta: null },
      pointsTotal:    { count: s.totalPoints, unit: '点' },
      facesCount:     s.faceCount,
      gaussianSplats: gs.length,
    });
  }

  // ── GET /api/memory/search?q=xxx ── 自然语言检索记忆
  if (p === '/api/memory/search') {
    const q = getQuery(req).q || '';
    const db = memdb();
    const cards = [];
    if (db && q) {
      try {
        const rows = db.prepare(
          "SELECT frame_path, description, capture_time, model FROM memories WHERE description LIKE ? OR frame_path LIKE ? LIMIT 20"
        ).all('%' + q + '%', '%' + q + '%');
        for (const r of rows) {
          cards.push({
            place: clsZh((r.frame_path || '').replace(/^.*?\//, '')),
            time: r.capture_time || '',
            dur: '',
            match: '--',
            desc: r.description || '',
            bg: 'linear-gradient(135deg,#5B6B52,#33402c)',
          });
          pi++;
        }
      } catch (_) {}
    }
    // Fallback: 返回 batch 目录场景数据
    if (!cards.length) {
      for (const d of scanBatchDirs()) {
        const sg = readJsonSafe(path.join(d, 'scene_graph.json'));
        if (sg && sg.nodes) {
          for (const n of sg.nodes) {
            if (q && n.description && n.description.includes(q))
              cards.push({
                place: n.category || n.name || d,
                time: '',
                dur: '',
                match: '--',
                desc: n.description || '',
                bg: 'linear-gradient(135deg,rgba(126,143,196,0.55),rgba(154,111,176,0.7))',
              });
          }
        }
      }
    }
    const html = cards.length
      ? `已检索到 <b>${cards.length} 段</b> 与「<b>${q}</b>」相关的记忆`
      : `暂未找到与「<b>${q}</b>」相关的记忆`;
    return json(res, 200, { html, cards: cards.slice(0, 10), query: q });
  }

  // ── GET /api/memory/anchors?batch_id=xxx ── 3D 漫游锚点
  if (p === '/api/memory/anchors') {
    const batchId = getQuery(req).batch_id;
    const anchors = [];
    const dirs = batchId
      ? [path.join(DATA_DIR, batchId)]
      : scanBatchDirs();
    for (const d of dirs) {
      if (!fs.existsSync(d)) continue;
      // DGSG objects 作为锚点
      const dobj = readJsonSafe(path.join(d, 'dgsg_objects.json'));
      if (dobj && dobj.objects) {
        for (const o of dobj.objects) {
          anchors.push({
            name: o.class_name || ('object_' + o.idx),
            time: '',
            dur: '',
            frames: 0,
            mood: '',
            moodColor: '#5E7A18',
            caption: o.description || '',
            tags: [o.class_name || ''],
            pos: o.center_3d || [0, 1, 0],
          });
        }
      }
      // Fallback: scene_graph nodes
      if (!anchors.length) {
        const sg = readJsonSafe(path.join(d, 'scene_graph.json'));
        if (sg && sg.nodes) {
          for (const n of sg.nodes) {
            anchors.push({
              name: n.category || n.name || ('node_' + n.idx),
              time: '',
              dur: '',
              frames: 0,
              mood: '',
              moodColor: '#C7700E',
              caption: n.description || '',
              tags: [n.category || ''],
              pos: n.center_3d || n.position || [0, 1, 0],
            });
          }
        }
      }
    }
    return json(res, 200, { anchors, count: anchors.length });
  }

  // ── GET /api/memory/clips?type=event|place|person|time ── 分类片段
  if (p === '/api/memory/clips') {
    const type = getQuery(req).type || 'event';
    const db = memdb();
    const items = [];
    if (db) {
      try {
        // 获取每条记忆及其裁剪图索引
        const rows = db.prepare(
          "SELECT frame_path, description, capture_time, inference_time as crop_id FROM memories ORDER BY capture_time DESC LIMIT 200"
        ).all();
        const groups = {};
        const palettes = [
          { bgSoft: 'rgba(255,176,124,0.12)', tag: { color: '#C7700E', bg: '#FCEBD3' } },
          { bgSoft: 'rgba(126,143,196,0.12)', tag: { color: '#4A5A8A', bg: '#E8EBF5' } },
          { bgSoft: 'rgba(255,210,122,0.12)', tag: { color: '#C7700E', bg: '#FCF1DD' } },
          { bgSoft: 'rgba(111,161,94,0.12)', tag: { color: '#5E7A18', bg: '#EEF7D6' } },
          { bgSoft: 'rgba(212,112,138,0.12)', tag: { color: '#8A3A55', bg: '#FBE8EE' } },
          { bgSoft: 'rgba(91,107,82,0.12)', tag: { color: '#3A4A33', bg: '#E8EBE3' } },
          { bgSoft: 'rgba(62,155,150,0.12)', tag: { color: '#1A5A56', bg: '#E0F2F1' } },
          { bgSoft: 'rgba(240,166,182,0.12)', tag: { color: '#8A3A55', bg: '#FBE8EE' } },
        ];
        let pi = 0;
        for (const r of rows) {
          const key = clsZh((r.frame_path || 'unknown').replace(/^.*?\//, ''));
          const cropId = r.crop_id || 0;
          if (!groups[key]) groups[key] = { place: key, count: 0, desc: r.description, time: r.capture_time, cropId: cropId };
          groups[key].count++;
        }
        for (const k of Object.keys(groups)) {
          const g = groups[k];
          const p = palettes[pi % palettes.length];
          items.push({
            place: g.place,
            time: g.time || '',
            count: g.count,
            duration: '--',
            desc: g.desc || '',
            bg: 'url(/crops/' + g.cropId + '.jpg) center/cover',
            bgSoft: p.bgSoft,
            badge: g.count > 1 ? g.count + ' 段' : '',
            title: g.place,
            sub: (function(ts){ try { var s=String(ts); var t=s.slice(0,10)+'T'+s.slice(11,19)+'Z'; var d=new Date(t); if(!isNaN(d.getTime())) return (d.getMonth()+1)+'月'+d.getDate()+'日 '; } catch(_) {} return ''; })(g.time) + (g.desc || '').slice(0, 50),
            initial: g.place[0] || '?',
            isRect: true,
            isRound: false,
            tags: [Object.assign({ label: g.place }, p.tag)],
          });
          pi++;
        }
      } catch (_) {}
    }
    return json(res, 200, { type, items: items.slice(0, 30), count: items.length });
  }

  // ── GET /api/memory/report ── 回忆报告
  if (p === '/api/memory/report') {
    const s = aggregateBatchStats();
    const db = memdb();
    let topPlaces = [], people = [], keywords = [], moodBars = [];
    if (db) {
      try {
        // 地点/物体聚合（去掉 datazhx/ 前缀）
        const placeRows = db.prepare(
          "SELECT frame_path, COUNT(*) as cnt FROM memories GROUP BY frame_path ORDER BY cnt DESC LIMIT 8"
        ).all();
        const maxCnt = placeRows.length ? placeRows[0].cnt : 1;
        const pal = ['#7E8FC4', '#6FA15E', '#F2A03D', '#3E9B96', '#FFB07C', '#F97C6E', '#D4708A', '#5B6B52'];
        topPlaces = placeRows.map((r, i) => ({
          name: clsZh((r.frame_path || '').replace(/^.*?\//, '')),
          count: r.cnt + ' 段',
          bar: `width:${Math.round(r.cnt / maxCnt * 100)}%; background:${pal[i % pal.length]}; border-radius:999px;`,
        }));
        // 人物
        const faceRows = db.prepare("SELECT name, count FROM faces ORDER BY count DESC LIMIT 4").all();
        const faceBgs = [
          'linear-gradient(135deg,#FFD27A,#F2A03D)',
          'linear-gradient(135deg,#7E8FC4,#9A6FB0)',
          'linear-gradient(135deg,#F0A6B6,#D4708A)',
          'linear-gradient(135deg,#FFB07C,#F97C6E)',
        ];
        people = faceRows.map((r, i) => ({
          initial: (r.name || '-')[0],
          name: r.name || '--',
          sub: '',
          count: r.count + ' 次',
          bg: faceBgs[i % faceBgs.length],
        }));
        // 关键词（从 VLM 识别出的物体类别聚合）
        const kwRows = db.prepare("SELECT frame_path, COUNT(*) as cnt FROM memories WHERE frame_path != '' GROUP BY frame_path ORDER BY cnt DESC LIMIT 15").all();
        const maxKw = kwRows.length ? kwRows[0].cnt : 1;
        keywords = kwRows.map((r, i) => {
          const word = clsZh((r.frame_path || '').replace(/^datazhx\//, ''));
          const n = r.cnt;
          return { text: word, style: `font-size:${Math.min(24, 12 + n * 2)}px; color:#33312b; border-radius:999px; padding:4px 12px;` };
        });
        // 心情条（简化：每天一段随机色）
        moodBars = Array.from({ length: 30 }, (_, i) => {
          const colors = ['#6FA15E', '#F2A03D', '#7E8FC4', '#FFB07C', '#D4708A'];
          const c = colors[i % colors.length];
          const h = 0.3 + Math.random() * 0.7;
          return { style: `flex:1; height:${Math.round(h * 100)}%; background:${c}; border-radius:5px 5px 2px 2px;` };
        });
      } catch (_) {}
    }
    return json(res, 200, {
      stats: [
        { value: String(s.batchCount || '--'), label: '重建空间', color: '#1c1d19' },
        { value: (s.processedFrames > 0 ? (s.processedFrames / 3600).toFixed(1) + 'h' : '--'), label: '记忆总时长', color: '#1c1d19' },
        { value: s.batchCount > 0 ? '+' + s.batchCount : '--', label: '新重建空间', color: '#5E7A18' },
        { value: String(s.memoryCount || '--'), label: 'VLM 标注数', color: '#C7700E' },
      ],
      moodBars,
      keywords: keywords.length ? keywords : [{ text: '等待数据...', style: 'font-size:16px; color:#8A8578;' }],
      topPlaces: topPlaces.length ? topPlaces : [{ name: '等待数据...', count: '--', bar: 'width:0%' }],
      people: people.length ? people : [{ initial: '-', name: '等待数据...', sub: '', count: '--', bg: 'linear-gradient(135deg,#ccc,#aaa)' }],
      // 时间线 days
      days: Array.from({ length: 31 }, (_, i) => {
        const colors = ['#6FA15E', '#F2A03D', '#7E8FC4', '#FFB07C', '#D4708A'];
        return { dot: colors[i % colors.length], label: (i + 1) + '日', value: '--' };
      }),
    });
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
http.createServer(handleRequest).listen(8081, () => {
  console.log(`Dashboard HTTP on http://0.0.0.0:8081`);
});
