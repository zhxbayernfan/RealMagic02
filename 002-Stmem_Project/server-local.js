const express = require('express');
const multer = require('multer');
const sqlite3 = require('better-sqlite3');
const cors = require('cors');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 8100;
const FRAMES_DIR = path.join(__dirname, '001-Data', 'frames');
const DB_PATH = path.join(__dirname, '001-Data', 'memory.sqlite');

app.use(cors());
app.use(express.static(__dirname));

const db = sqlite3(DB_PATH);
db.exec(`CREATE TABLE IF NOT EXISTS memories (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  filename TEXT,
  description TEXT,
  model TEXT,
  capture_time TEXT,
  file_size INTEGER
)`);

const upload = multer({ dest: FRAMES_DIR });

// Stats endpoint
app.get('/api/stats', (req, res) => {
  const total = db.prepare('SELECT COUNT(*) as c FROM memories').get().c;
  const labeled = db.prepare("SELECT COUNT(*) as c FROM memories WHERE description != '' AND description IS NOT NULL AND description != '无结果'").get().c;
  const latest = db.prepare('SELECT capture_time FROM memories ORDER BY capture_time DESC LIMIT 1').get();
  const files = fs.existsSync(FRAMES_DIR) ? fs.readdirSync(FRAMES_DIR).filter(f => /\.(jpg|jpeg|png)$/i.test(f)) : [];
  let totalSize = 0;
  files.forEach(f => { try { totalSize += fs.statSync(path.join(FRAMES_DIR, f)).size } catch(_) {} });
  res.json({
    total, labeled,
    latestTime: latest ? latest.capture_time : null,
    storageMB: Math.round(totalSize / 1024 / 1024 * 10) / 10
  });
});

// List all frames
app.get('/api/frames', (req, res) => {
  const rows = db.prepare('SELECT id, filename, description, model, capture_time FROM memories ORDER BY id DESC').all();
  const files = new Set(fs.existsSync(FRAMES_DIR) ? fs.readdirSync(FRAMES_DIR) : []);
  const result = rows.filter(r => files.has(r.filename)).map(r => ({
    id: r.id, filename: r.filename,
    description: r.description || '',
    model: r.model || '',
    time: r.capture_time || ''
  }));
  res.json(result);
});

// Get single frame image
app.get('/api/frame/:filename', (req, res) => {
  const fp = path.join(FRAMES_DIR, req.params.filename);
  if (!fp.startsWith(FRAMES_DIR)) return res.status(403).send('Forbidden');
  if (fs.existsSync(fp)) res.sendFile(fp);
  else res.status(404).send('Not found');
});

// Upload frame
app.post('/api/upload', upload.single('image'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No file' });
  const ext = path.extname(req.file.originalname) || '.jpg';
  const count = db.prepare('SELECT COUNT(*) as c FROM memories').get().c;
  const filename = 'frame_' + String(count + 1).padStart(3, '0') + ext;
  const dest = path.join(FRAMES_DIR, filename);
  fs.renameSync(req.file.path, dest);
  const now = new Date().toISOString();
  db.prepare('INSERT INTO memories (filename, description, model, capture_time, file_size) VALUES (?, ?, ?, ?, ?)').run(filename, '', '', now, req.file.size);
  const id = db.prepare('SELECT last_insert_rowid() as id').get().id;
  res.json({ id, filename });
});

// Update description
app.post('/api/update/:id', express.json(), (req, res) => {
  db.prepare('UPDATE memories SET description = ?, model = ? WHERE id = ?').run(req.body.description || '', req.body.model || '', req.params.id);
  res.json({ ok: true });
});

// Delete frame
app.delete('/api/frame/:id', (req, res) => {
  const row = db.prepare('SELECT filename FROM memories WHERE id = ?').get(req.params.id);
  if (!row) return res.status(404).json({ error: 'Not found' });
  const fp = path.join(FRAMES_DIR, row.filename);
  if (fs.existsSync(fp)) fs.unlinkSync(fp);
  db.prepare('DELETE FROM memories WHERE id = ?').run(req.params.id);
  res.json({ ok: true });
});

app.listen(PORT, () => {
  console.log('02-002 Local Server on http://localhost:' + PORT);
  console.log('Frames dir:', FRAMES_DIR);
  console.log('DB path:', DB_PATH);
});
