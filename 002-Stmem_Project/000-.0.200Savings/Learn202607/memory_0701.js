'use strict';

const fs = require('fs');
const path = require('path');
const { sendJson } = require('../router');
const { FRAMES_DIR, MEMORY_DIR } = require('../../utils/paths');

// List frame image files on disk; no memory data included.
function listFrameFiles() {
  if (!fs.existsSync(FRAMES_DIR)) return [];
  return fs.readdirSync(FRAMES_DIR)
    .filter((f) => f.endsWith('.png') || f.endsWith('.jpg'))
    .sort()
    .map((filename) => {
      const framePath = path.join(FRAMES_DIR, filename);
      const stats = fs.statSync(framePath);
      const ext = path.extname(filename);
      const id = filename.replace('frame_', '').replace(ext, '');
      return { id, filename, path: framePath, size: stats.size, createdAt: stats.birthtime.toISOString(), thumbnail: '/frames/' + filename };
    });
}

// Read memory from JSON file (legacy fallback for jsonFs adapter).
function readLegacyMemory(filename) {
  const ext = path.extname(filename);
  const memFile = path.join(MEMORY_DIR, filename.replace(ext, '.json'));
  if (!fs.existsSync(memFile)) return null;
  try { return JSON.parse(fs.readFileSync(memFile, 'utf8')); } catch (_) { return null; }
}

/**
 * Build the frames list that is served to the front-end and used by the query pipeline.
 * - For SQLite store: memory data comes from the store (async).
 * - For jsonFs store (legacy): memory data comes from the JSON files on disk.
 * Always returns a Promise<Frame[]>.
 */
async function listFrames(memoryStore) {
  const files = listFrameFiles();
  if (files.length === 0) return [];

  if (memoryStore) {
    // Fetch all memories from store in one call
    const allMems = await memoryStore.list({}).catch(() => []);
    const memById = {};
    for (const m of allMems) memById[String(m.id)] = m;

    return files.map((f) => {
      const mem = memById[String(parseInt(f.id, 10))] || null;
      // Fall back to JSON file if store returned nothing (transition period)
      const legacyMem = mem ? null : readLegacyMemory(f.filename);
      const resolved = mem || legacyMem;
      return {
        id: f.id,
        filename: f.filename,
        path: f.path,
        size: f.size,
        createdAt: resolved ? (resolved.captureTime || f.createdAt) : f.createdAt,
        hasMemory: !!resolved,
        memory: resolved,
        thumbnail: f.thumbnail,
      };
    });
  }

  // No store provided (legacy path): read JSON files directly
  return files.map((f) => {
    const mem = readLegacyMemory(f.filename);
    return {
      id: f.id,
      filename: f.filename,
      path: f.path,
      size: f.size,
      createdAt: f.createdAt,
      hasMemory: !!mem,
      memory: mem,
      thumbnail: f.thumbnail,
    };
  });
}

function registerMemoryRoutes(router, ctx) {

  router.get('/api/status', (req, res) => {
    sendJson(res, 200, {
      config: ctx.getConfig(),
      isCapturing: ctx.captureController.isCapturing(),
      faceApiReady: ctx.faceService.isReady(),
      sourceName: ctx.captureController.sourceName(),
    });
  });

  router.get('/api/frames', async (req, res) => {
    try {
      const frames = await listFrames(ctx.memoryStore);
      sendJson(res, 200, frames);
    } catch (err) {
      sendJson(res, 500, { error: 'Failed to list frames', message: err.message });
    }
  });

  router.get('/api/faces', (req, res) => {
    const summary = ctx.faceLibrary.summary();
    sendJson(res, 200, { success: true, faces: summary, total: summary.length });
  });

  // ========== 记忆聚合 API (New_Stmem.html 数据源) ==========

  // ── GET /api/memory/summary ── 概览页统计
  router.get('/api/memory/summary', async (req, res) => {
    try {
      const s = ctx.memoryStore;
      let memCount = 0, faceCount = 0, archCount = 0, batchCount = 0, totalFrames = 0;
      if (s) {
        try { memCount  = await s.count('memories'); } catch (_) {}
        try { faceCount = await s.count('faces'); } catch (_) {}
        try { archCount = await s.count('archived_memories'); } catch (_) {}
      }
      // 扫描 data/ 目录下所有包含 frames/ 子目录的空间
      const DATA_DIR = path.join(require('../../utils/paths').MEMORY_DIR, '..');
      const dd = path.resolve(DATA_DIR);
      const spaceDirs = new Set();
      if (fs.existsSync(dd)) {
        for (const d of fs.readdirSync(dd)) {
          const dp = path.join(dd, d);
          // 跳过符号链接（如 latest -> jszn）
          if (fs.lstatSync(dp).isSymbolicLink()) continue;
          if (!fs.statSync(dp).isDirectory()) continue;
          // 统计 frames 数量（只有 frames/ 子目录存在且有帧文件才算一个空间）
          const framesDir = path.join(dp, 'frames');
          let frameCount = 0;
          if (fs.existsSync(framesDir)) {
            try { frameCount = fs.readdirSync(framesDir).filter(f => f.match(/\.(jpg|jpeg|png)$/i)).length; } catch (_) {}
          }
          if (frameCount > 0) spaceDirs.add(d);
          // status.json 统计总帧数
          const stPath = path.join(dp, 'status.json');
          if (fs.existsSync(stPath)) {
            try {
              const st = JSON.parse(fs.readFileSync(stPath, 'utf8'));
              totalFrames += (st.total_frames || 0);
            } catch (_) {}
          }
        }
        // 直接统计 data/frames/ 目录下的帧（平铺结构）
        const flatFrames = path.join(dd, 'frames');
        if (fs.existsSync(flatFrames)) {
          try {
            const n = fs.readdirSync(flatFrames).filter(f => f.match(/\.(jpg|jpeg|png)$/i)).length;
            if (n > 0) { totalFrames += n; spaceDirs.add('frames'); }
          } catch (_) {}
        }
      }
      batchCount = spaceDirs.size;
      // GS 文件
      const gsDir = path.join(dd, 'gaussian-splats');
      let gsCount = 0;
      if (fs.existsSync(gsDir)) {
        gsCount = fs.readdirSync(gsDir).filter(f => f.endsWith('.ply')).length;
      }
      sendJson(res, 200, {
        totalDuration:  { hours: (totalFrames / 3600).toFixed(1), delta: null },
        keyframeCount:  { count: totalFrames, annotated: memCount, delta: null },
        spaceCount:     { count: batchCount, delta: batchCount > 0 ? '+' + batchCount + ' 个空间' : null },
        eventCount:     { count: memCount + archCount, delta: null },
        pointsTotal:    { count: 0, unit: '点' },
        facesCount:     faceCount,
        gaussianSplats: gsCount,
        isCapturing:    ctx.captureController.isCapturing(),
      });
    } catch (e) { sendJson(res, 500, { error: e.message }); }
  });

  // ── GET /api/memory/search?q=xxx ── 记忆检索
  router.get('/api/memory/search', async (req, res) => {
    try {
      const q = (new URL(req.url, 'http://localhost').searchParams.get('q') || '').trim();
      const cards = [];
      if (q && ctx.memoryStore) {
        try {
            for (const m of all) {
            const desc = m.description || '';
            const path = m.frame_path || m.framePath || '';
            if (desc.includes(q) || path.includes(q)) {
              cards.push({
                place: path, time: m.capture_time || m.captureTime || '',
                dur: '', match: '--', desc: desc,
                bg: 'linear-gradient(135deg,#5B6B52,#33402c)',
              });
            }
          }
        } catch (_) {}
      }
      // 向量搜索 (fallback)
      if (!cards.length && q && ctx.vectorIndex) {
        try {
          const vecResults = await ctx.vectorIndex.search(q, 10);
          for (const r of vecResults) {
            cards.push({
              place: r.metadata?.frame_path || '', time: r.metadata?.capture_time || '',
              dur: '', match: (r.score ? (r.score * 100).toFixed(0) + '%' : '--'),
              desc: r.metadata?.description || '',
              bg: 'linear-gradient(135deg,#7E8FC4,#9A6FB0)',
            });
          }
        } catch (_) {}
      }
      const html = cards.length
        ? '已检索到 <b>' + cards.length + ' 段</b> 与「<b>' + q + '</b>」相关的记忆'
        : '暂未找到与「<b>' + q + '</b>」相关的记忆';
      sendJson(res, 200, { html, cards: cards.slice(0, 10), query: q });
    } catch (e) { sendJson(res, 500, { error: e.message }); }
  });

  // ── GET /api/memory/anchors ── 3D 漫游锚点
  router.get('/api/memory/anchors', async (req, res) => {
    try {
      const anchors = [];
      if (ctx.memoryStore) {
        try {
          const all = await ctx.memoryStore.list({});
          const seen = new Set();
          for (const m of all) {
            const key = m.frame_path || m.framePath || '';
            if (!key || seen.has(key)) continue;
            seen.add(key);
            anchors.push({
              name: key, time: m.capture_time || m.captureTime || '',
              dur: '', frames: 0, mood: '', moodColor: '#5E7A18',
              caption: m.description || '', tags: [],
              pos: [Math.random() * 10 - 5, 1, Math.random() * -10],
            });
          }
        } catch (_) {}
      }
      sendJson(res, 200, { anchors, count: anchors.length });
    } catch (e) { sendJson(res, 500, { error: e.message }); }
  });

  /* ── GET /api/memory/clips ── (replaced by direct SQLite version below)
  router.get('/api/memory/clips', async (req, res) => {
    try {
      const type = (new URL(req.url, 'http://localhost')).searchParams.get('type') || 'event';
      const items = [];
      const CLS_ZH = {'chair':'椅子','table':'桌子','tv':'电视','cup':'杯子','bottle':'瓶子','box':'盒子','keyboard':'键盘','laptop':'笔记本电脑','cat':'猫','phone':'手机','door':'门','storage bin':'收纳箱','power adapter':'电源适配器','hat':'帽子','monitor':'显示器','can':'易拉罐','desk':'书桌','potted plant':'盆栽','plant':'植物','mouse':'鼠标','book':'书','bag':'包','basket':'篮子','cabinet':'柜子'};
      function cls(n){return CLS_ZH[n]||n;}
      if (ctx.memoryStore) {
        const DATA_DIR = path.resolve(path.join(require("../../utils/paths").MEMORY_DIR, ".."));
      const dbPath = path.join(DATA_DIR, "memory.sqlite");
      const db = require("fs").existsSync(dbPath) ? require("better-sqlite3")(dbPath) : null;
      try {
        const rows = db.prepare("SELECT frame_path, description, capture_time, inference_time as crop_id FROM memories ORDER BY capture_time DESC LIMIT 200").all();
      const pal = [{bgSoft:'rgba(255,176,124,0.12)',tag:{color:'#C7700E',bg:'#FCEBD3'}},{bgSoft:'rgba(126,143,196,0.12)',tag:{color:'#4A5A8A',bg:'#E8EBF5'}},{bgSoft:'rgba(255,210,122,0.12)',tag:{color:'#C7700E',bg:'#FCF1DD'}},{bgSoft:'rgba(111,161,94,0.12)',tag:{color:'#5E7A18',bg:'#EEF7D6'}},{bgSoft:'rgba(212,112,138,0.12)',tag:{color:'#8A3A55',bg:'#FBE8EE'}},{bgSoft:'rgba(91,107,82,0.12)',tag:{color:'#3A4A33',bg:'#E8EBE3'}},{bgSoft:'rgba(62,155,150,0.12)',tag:{color:'#1A5A56',bg:'#E0F2F1'}},{bgSoft:'rgba(240,166,182,0.12)',tag:{color:'#8A3A55',bg:'#FBE8EE'}}];
        const all = await ctx.memoryStore.list({});
        if (type === 'time') {
          for (let ri = 0; ri < rows.length; ri++) {
            const r = rows[ri], p = pal[ri % pal.length];
            const place = cls((r.frame_path || r.framePath || 'unknown').replace(/^.*?\//, ''));
            const cid = r.inference_time || ri;
            const ts = r.capture_time || r.captureTime || '';
            const ds = r.description || '';
            const sd = (function(t){try{if(!t)return '';var s=String(t).slice(0,19).replace(/[ -]/g,'');var d=new Date(s.replace(/(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})/, '$1-$2-$3T$4:$5:$6Z'));if(!isNaN(d.getTime()))return (d.getMonth()+1)+'月'+d.getDate()+'日 ';}catch(_){}return '';})(r.capture_time) + ds.slice(0,50);
            items.push({place, time: ts, count:1, duration:'--', desc: ds, bg:'url(/crops/'+cid+'.jpg) center/cover', bgSoft:p.bgSoft, badge:'', title:place, sub: sd, initial:place[0]||'?', isRect:true, isRound:false, tags:[Object.assign({label:place},p.tag)]});
          }
        } else {
          const groups = {};
          for (const r of rows) {
            const key = cls((r.frame_path || r.framePath || 'unknown').replace(/^.*?\//, ''));
            if (!groups[key]) groups[key] = {place:key, count:0, desc:r.description, time:r.capture_time||r.captureTime||'', cropId:r.inference_time||0};
            groups[key].count++;
          }
          let pi = 0;
          for (const k of Object.keys(groups).sort((a,b)=>groups[b].count-groups[a].count||(groups[b].time||'').localeCompare(groups[a].time||''))) {
            const g = groups[k], p = pal[pi % pal.length];
            const sd = (function(t){try{if(!t)return '';var s=String(t).slice(0,19).replace(/[ -]/g,'');var d=new Date(s.replace(/(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})/, '$1-$2-$3T$4:$5:$6Z'));if(!isNaN(d.getTime()))return (d.getMonth()+1)+'月'+d.getDate()+'日 ';}catch(_){}return '';})(g.time) + (g.desc||'').slice(0,50);
            items.push({place:g.place, time:g.time||'', count:g.count, duration:'--', desc:g.desc||'', bg:'url(/crops/'+g.cropId+'.jpg) center/cover', bgSoft:p.bgSoft, badge:g.count>1?g.count+' 段':'', title:g.place, sub: sd, initial:g.place[0]||'?', isRect:true, isRound:false, tags:[Object.assign({label:g.place},p.tag)]});
            pi++;
          }
        }
      }
      sendJson(res, 200, { type, items: items.slice(0, 30), count: items.length });
    } catch (e) { sendJson(res, 500, { error: e.message }); }
  });
  */
  // ── GET /api/memory/report ── 回忆报告
  router.get('/api/memory/report', async (req, res) => {
    try {
      let memCount = 0, batchCount = 0, totalFrames = 0, faceCount = 0;
      if (ctx.memoryStore) {
        try { memCount = await ctx.memoryStore.count('memories'); } catch (_) {}
        try { faceCount = await ctx.memoryStore.count('faces'); } catch (_) {}
      }
      const DATA_DIR = path.resolve(path.join(require('../../utils/paths').MEMORY_DIR, '..'));
      if (fs.existsSync(DATA_DIR)) {
        for (const d of fs.readdirSync(DATA_DIR)) {
          const dp = path.join(DATA_DIR, d);
          if (fs.lstatSync(dp).isSymbolicLink()) continue;
          if (!fs.statSync(dp).isDirectory()) continue;
          const framesDir = path.join(dp, 'frames');
          let frameCount = 0;
          if (fs.existsSync(framesDir)) {
            try { frameCount = fs.readdirSync(framesDir).filter(f => f.match(/\.(jpg|jpeg|png)$/i)).length; } catch (_) {}
          }
          if (frameCount > 0) batchCount++;
          if (frameCount > 0 && fs.existsSync(path.join(dp, 'status.json'))) {
            try { totalFrames += JSON.parse(fs.readFileSync(path.join(dp, 'status.json'), 'utf8')).total_frames || 0; } catch (_) {}
          }
        }
        // 直接统计 data/frames/ 下的帧（平铺结构）
        const flatFrames = path.join(DATA_DIR, 'frames');
        if (fs.existsSync(flatFrames)) {
          try {
            const n = fs.readdirSync(flatFrames).filter(f => f.match(/\.(jpg|jpeg|png)$/i)).length;
            if (n > 0) { totalFrames += n; batchCount++; }
          } catch (_) {}
        }
      }
      // 人物
      let people = [];
      if (ctx.faceLibrary) {
        try {
          const faces = ctx.faceLibrary.summary();
          const faceBgs = ['linear-gradient(135deg,#FFD27A,#F2A03D)','linear-gradient(135deg,#7E8FC4,#9A6FB0)','linear-gradient(135deg,#F0A6B6,#D4708A)','linear-gradient(135deg,#FFB07C,#F97C6E)'];
          for (let i = 0; i < Math.min(faces.length, 4); i++) {
            const f = faces[i];
            people.push({ initial: (f.name || '-')[0], name: f.name || '--', sub: '', count: (f.count || 0) + ' 次', bg: faceBgs[i] });
          }
        } catch (_) {}
      }
      // 心情条
      const moodBars = Array.from({ length: 30 }, (_, i) => {
        const colors = ['#6FA15E', '#F2A03D', '#7E8FC4', '#FFB07C', '#D4708A'];
        return { style: 'flex:1; height:' + Math.round(30 + Math.random() * 70) + '%; background:' + colors[i % 5] + '; border-radius:5px 5px 2px 2px;' };
      });
      // SQLite 查询：topPlaces / keywords / people / days
      let topPlaces = [], kwList = [], days = [];
      const DATA_DIR2 = path.resolve(path.join(require('../../utils/paths').MEMORY_DIR, '..'));
      const dbPath2 = path.join(DATA_DIR2, 'memory.sqlite');
      if (fs.existsSync(dbPath2)) {
        try {
          const bsql = require('better-sqlite3');
          const db2 = bsql(dbPath2);
          const pal = ['#7E8FC4','#6FA15E','#F2A03D','#3E9B96','#FFB07C','#F97C6E','#D4708A','#5B6B52'];
          const pl = db2.prepare("SELECT description FROM memories").all();
          const allDesc = pl.map(r => r.description||'').join(' ');
          const locWords = ['办公室','走廊','实验室','卧室','客厅','厨房','阳台','户外','车内','街道','公园','工厂','车间','仓库','大厅','会议室','教室','图书馆','餐厅','公司','医院','宿舍','酒店','家里'];
          const locCounts = {};
          locWords.forEach(function(w){var re=new RegExp(w,'g');var m=allDesc.match(re);if(m)locCounts[w]=m.length;});
          const locs = Object.entries(locCounts).sort((a,b)=>b[1]-a[1]).slice(0,8);
          const mxc = locs.length ? locs[0][1] : 1;
          topPlaces = locs.map(([name,cnt],i) => ({ name, count: cnt+' 段', bar: 'width:'+Math.round(cnt/mxc*100)+'%;background:'+pal[i%pal.length]+';border-radius:999px;' }));
          const kr = db2.prepare("SELECT description FROM memories ORDER BY capture_time DESC LIMIT 15").all();
          const allText = kr.map(r => r.description||'').join(' ');
          const objects = ['电脑','显示器','键盘','鼠标','手机','椅子','桌子','猫','机械臂','机器人','机器狗','绿植','植物','盆栽','文件','纸张','书','杯子','瓶子','包装','背包','袋子','箱','路灯','汽车','门','窗','灯','柜','架子','床','枕头','被子','衣服','鞋子','眼镜','耳机','笔','本','花','树','草','楼梯','走廊','办公室','实验室','卧室'];
          const freq = {};
          objects.forEach(function(o){var re=new RegExp(o,'g');var m=allText.match(re);if(m)freq[o]=m.length;});
          kwList = Object.entries(freq).sort((a,b) => b[1]-a[1]).slice(0,15).map(([text,cnt]) => ({ text, style: 'font-size:'+Math.min(24,12+cnt*2)+'px;color:#33312b;border-radius:999px;padding:4px 12px;' }));
          var charList = [
            {re:/绿色.*AIRKALK|绿T恤|穿着绿色上衣|穿绿色上衣/gi, label:'绿衣男子'},
            {re:/棕色.*[猫虎]|橘色猫/gi, label:'橘猫'},
            {re:/白色.*小猫|白猫|白色小猫/gi, label:'白猫'},
            {re:/戴口罩/gi, label:'戴口罩者'},
            {re:/黑衣.*男性|穿着黑色上衣|穿黑色上衣/gi, label:'黑衣男子'},
            {re:/穿着白色上衣|穿白色衬衫/gi, label:'白衣男子'},
            {re:/蓝色短袖|深蓝色短袖/gi, label:'蓝衣男子'},
            {re:/卡通.*角色|动漫风格.*卡片/gi, label:'游戏角色'},
            {re:/四足.*机器人|白色.*动物.*机器人|机器狗/gi, label:'机器狗'},
          ];
          var charCounts = {};
          charList.forEach(function(item){var m=allDesc.match(item.re);if(m)charCounts[item.label]=m.length>1?1:m.length;});
          var chars = Object.entries(charCounts).sort(function(a,b){return b[1]-a[1]}).slice(0,6);
          if (chars.length) {
            const fb = ['linear-gradient(135deg,#FFD27A,#F2A03D)','linear-gradient(135deg,#7E8FC4,#9A6FB0)','linear-gradient(135deg,#F0A6B6,#D4708A)','linear-gradient(135deg,#FFB07C,#F97C6E)'];
            people = chars.map(function(item,i){return {initial:item[0][0],name:item[0],count:item[1]+' 帧',bg:fb[i%fb.length]};});
          }
          days = Array.from({length:31}, (_,i) => ({ dot: ['#6FA15E','#F2A03D','#7E8FC4','#FFB07C','#D4708A'][i%5], label: (i+1)+'日', value: i+1 <= kr.length ? '●' : '--' }));
          db2.close();
        } catch(_) {}
      }
      sendJson(res, 200, {
        stats: [
          { value: String(batchCount || '--'), label: '重建空间', color: '#1c1d19' },
          { value: (totalFrames > 0 ? (totalFrames / 3600).toFixed(1) + 'h' : '--'), label: '记忆总时长', color: '#1c1d19' },
          { value: batchCount > 0 ? '+' + batchCount : '--', label: '新重建空间', color: '#5E7A18' },
          { value: String(memCount || '--'), label: '识别物件', color: '#C7700E' },
        ],
        moodBars,
        keywords: kwList.length ? kwList : [{ text: '等待数据...', style: 'font-size:16px; color:#8A8578;' }],
      reportStats: [
          { value: String(batchCount || "--"), label: "重建空间", color: "#1c1d19" },
          { value: (totalFrames > 0 ? (totalFrames / 3600).toFixed(1) + "h" : "--"), label: "记忆总时长", color: "#1c1d19" },
          { value: batchCount > 0 ? "+" + batchCount : "--", label: "新重建空间", color: "#5E7A18" },
          { value: String(memCount || "--"), label: "识别物件", color: "#C7700E" },
        ],
        topPlaces: topPlaces.length ? topPlaces : [{ name: '等待数据...', count: '--', bar: 'width:0%' }],
        people: people.length ? people : [{ initial: '-', name: '等待数据...', sub: '', count: '--', bg: 'linear-gradient(135deg,#ccc,#aaa)' }],
        days: days.length ? days : Array.from({ length: 31 }, (_, i) => {
          const colors = ['#6FA15E', '#F2A03D', '#7E8FC4', '#FFB07C', '#D4708A'];
          return { dot: colors[i % 5], label: (i + 1) + '日', value: '--' };
        }),
        isCapturing: ctx.captureController.isCapturing(),
      });
    } catch (e) { sendJson(res, 500, { error: e.message }); }
  });


  // ── 数据聚合 API (20260626) ──
  const DATA_DIR = path.join(require("../../utils/paths").MEMORY_DIR, "..");
  const AGG_DATA_DIR = path.join(DATA_DIR);
  const AGG_CLS = { 'chair':'椅子','table':'桌子','tv':'电视','cup':'杯子','bottle':'瓶子','box':'盒子','keyboard':'键盘','laptop':'笔记本电脑','cat':'猫','phone':'手机','door':'门','storage bin':'收纳箱','power adapter':'电源适配器','hat':'帽子','monitor':'显示器','can':'易拉罐','desk':'书桌','book':'书','mouse':'鼠标','potted plant':'盆栽','plant':'植物' };
  function agg_cls(n){return AGG_CLS[n]||n;}
  function agg_db(){const p=path.join(AGG_DATA_DIR,'memory.sqlite');if(require('fs').existsSync(p)){try{return require('better-sqlite3')(p);}catch(_){}}return null;}
  function agg_st(){const fs=require('fs');let tf=0,pf=0,bc=0,mc=0,fc=0,ac=0;if(fs.existsSync(AGG_DATA_DIR)){for(const d of fs.readdirSync(AGG_DATA_DIR)){const dp=path.join(AGG_DATA_DIR,d);if(fs.lstatSync(dp).isSymbolicLink()||!fs.statSync(dp).isDirectory())continue;const sp=path.join(dp,'status.json');if(fs.existsSync(sp)){try{const s=JSON.parse(fs.readFileSync(sp,'utf8'));bc++;tf+=s.total_frames||0;pf+=s.processed_frames||0;}catch(_){}}}const framesDir=path.join(AGG_DATA_DIR,'frames');if(fs.existsSync(framesDir)){try{const n=fs.readdirSync(framesDir).filter(f=>f.match(/\.(jpg|jpeg|png)$/i)).length;if(n>0){tf+=n;pf+=n;bc++;}}catch(_){}}}const db=agg_db();if(db){try{mc=db.prepare('SELECT COUNT(*)as c FROM memories').get().c;fc=db.prepare('SELECT COUNT(*)as c FROM faces').get().c;ac=db.prepare('SELECT COUNT(*)as c FROM archived_memories').get().c;}catch(_){}}return{totalFrames:tf,processedFrames:pf,batchCount:bc,memoryCount:mc,faceCount:fc,archivedCount:ac};}

  router.get('/api/memory/summary', (req,res)=>{const s=agg_st();const gd=path.join(AGG_DATA_DIR,'gaussian-splats');let gc=0;if(require('fs').existsSync(gd))gc=require('fs').readdirSync(gd).filter(f=>f.endsWith('.ply')).length;sendJson(res,200,{totalDuration:{hours:(s.processedFrames/3600).toFixed(1),delta:null},keyframeCount:{count:s.processedFrames,annotated:s.memoryCount,delta:null},spaceCount:{count:s.batchCount+gc,delta:s.batchCount>0?`+${s.batchCount} 个空间`:null},eventCount:{count:s.memoryCount+s.archivedCount,delta:null},pointsTotal:{count:0,unit:'点'},facesCount:s.faceCount,gaussianSplats:gc});});

  router.get('/api/memory/search', (req,res)=>{const q=(new URL(req.url,'http://localhost')).searchParams.get('q')||'';const db=agg_db();const cards=[];if(db&&q){try{const rows=db.prepare("SELECT frame_path,description,capture_time FROM memories WHERE description LIKE?OR frame_path LIKE?LIMIT 20").all('%'+q+'%','%'+q+'%');for(const r of rows)cards.push({place:agg_cls((r.frame_path||'').replace(/^.*?\//,'')),time:r.capture_time||'',dur:'',match:'--',desc:r.description||'',bg:'linear-gradient(135deg,#5B6B52,#33402c)'});}catch(_){}}const html=cards.length?`已检索到<b>${cards.length}段</b>与「<b>${q}</b>」相关的记忆`:`暂未找到与「<b>${q}</b>」相关的记忆`;sendJson(res,200,{html,cards:cards.slice(0,10),query:q});});

  router.get('/api/memory/anchors', (req,res)=>{const db=agg_db();const anchors=[];if(db){try{const rows=db.prepare("SELECT frame_path,description,capture_time FROM memories LIMIT 100").all();for(const r of rows)anchors.push({name:agg_cls((r.frame_path||'').replace(/^.*?\//,'')),time:r.capture_time||'',dur:'',frames:0,mood:'',moodColor:'#5E7A18',caption:r.description||'',tags:[],pos:[Math.random()*10-5,1,Math.random()*-10]});}catch(_){}}sendJson(res,200,{anchors,count:anchors.length});});

  // clips — 从 scene_graph.json 读取，按 category 聚类
  router.get('/api/memory/clips', (req,res)=>{
    const type=(new URL(req.url,'http://localhost')).searchParams.get('type')||'event';
    const items=[]; const itemsAll=[];
    const pal=[{bgSoft:'rgba(255,176,124,0.12)',tag:{color:'#C7700E',bg:'#FCEBD3'}},{bgSoft:'rgba(126,143,196,0.12)',tag:{color:'#4A5A8A',bg:'#E8EBF5'}},{bgSoft:'rgba(255,210,122,0.12)',tag:{color:'#C7700E',bg:'#FCF1DD'}},{bgSoft:'rgba(111,161,94,0.12)',tag:{color:'#5E7A18',bg:'#EEF7D6'}},{bgSoft:'rgba(212,112,138,0.12)',tag:{color:'#8A3A55',bg:'#FBE8EE'}},{bgSoft:'rgba(91,107,82,0.12)',tag:{color:'#3A4A33',bg:'#E8EBE3'}},{bgSoft:'rgba(62,155,150,0.12)',tag:{color:'#1A5A56',bg:'#E0F2F1'}},{bgSoft:'rgba(240,166,182,0.12)',tag:{color:'#8A3A55',bg:'#FBE8EE'}}];
    const sgPath=path.join(AGG_DATA_DIR,'current','scene_graph.json');
    try {
      if(fs.existsSync(sgPath)){
        const sg=JSON.parse(fs.readFileSync(sgPath,'utf8'));
        const nodes=sg.nodes||[];
        if(type==='time'){
          for(let i=0;i<nodes.length;i++){
            const n=nodes[i],p=pal[i%pal.length];
            const cat=agg_cls(n.category||'unknown');
            items.push({place:cat,time:'',count:1,duration:'--',desc:n.description||'',bg:'url(/crops/'+n.idx+'.jpg) center/cover',bgSoft:p.bgSoft,badge:'',title:cat,sub:(n.description||'').slice(0,50),initial:cat[0]||'?',isRect:true,isRound:false,tags:[Object.assign({label:cat},p.tag)]});
          }
        } else {
          const groups={};
          for(const n of nodes){
            const cat=n.category||'unknown';
            if(!groups[cat])groups[cat]={place:agg_cls(cat),count:0,desc:n.description||'',cropId:n.idx||0};
            groups[cat].count++;
          }
          let pi=0;
          for(const k of Object.keys(groups).sort((a,b)=>groups[b].count-groups[a].count)){
            const g=groups[k],p=pal[pi%pal.length];
            items.push({place:g.place,time:'',count:g.count,duration:'--',desc:g.desc,bg:'url(/crops/'+g.cropId+'.jpg) center/cover',bgSoft:p.bgSoft,badge:g.count>1?g.count+' 段':'',title:g.place,sub:g.desc.slice(0,50),initial:g.place[0]||'?',isRect:true,isRound:false,tags:[Object.assign({label:g.place},p.tag)]});
            pi++;
          }
        }
      }
    } catch(_) {}
    sendJson(res,200,{type,items,count:items.length});
  });

  // mood — 按时段数算活跃度（30分钟内归为一个时段）
  router.get('/api/memory/mood', (req,res)=>{const db=agg_db();let sessions=0,activeDays=0,totalFrames=0,topPlace='',topPerson='',sight='';if(db){try{const rows=db.prepare('SELECT capture_time,description FROM memories ORDER BY capture_time').all();totalFrames=rows.length;if(totalFrames){const days=new Set();let locCounts={};let personList={};let allDesc=[];let sessionStart=rows[0].capture_time;let last=rows[0].capture_time;sessions=1;for(let i=0;i<rows.length;i++){const r=rows[i];const ct=r.capture_time||'';const desc=r.description||'';allDesc.push(desc);if(ct){const d=ct.substring(0,10);if(d)days.add(d);}if(i>0){const gapMin=ct&&last?(new Date(ct)-new Date(last))/60000:0;const sessMin=ct&&sessionStart?(new Date(ct)-new Date(sessionStart))/60000:0;if(gapMin>30||sessMin>60){sessions++;sessionStart=ct;}}last=ct;}activeDays=days.size;sight=allDesc.length?(allDesc[Math.floor(Math.random()*allDesc.length)]||'').replace(/[人物：:物品：:场景。，、；\n]/g,'').trim().substring(0,20):'';var txt=allDesc.join(' ');['办公室','走廊','实验室','卧室','客厅','厨房','阳台','户外','车内','街道','公园'].forEach(function(w){var re=new RegExp(w,'g');var m=txt.match(re);if(m)locCounts[w]=m.length;});var locs=Object.entries(locCounts).sort(function(a,b){return b[1]-a[1]});if(locs.length)topPlace=locs[0][0];[['绿衣男子',/绿色.*AIRKALK|绿T恤|穿着绿色上衣|穿绿色上衣/gi],['橘猫',/棕色.*[猫虎]|橘色猫/gi],['黑衣男子',/黑衣.*男性|穿着黑色上衣|穿黑色上衣/gi],['白衣男子',/穿着白色上衣|穿白色衬衫/gi],['蓝衣男子',/蓝色短袖|深蓝色短袖/gi],['机器狗',/四足.*机器人|白色.*动物.*机器人|机器狗/gi]].forEach(function(p){var m=txt.match(p[1]);if(m)personList[p[0]]=m.length;});var ps=Object.entries(personList).sort(function(a,b){return b[1]-a[1]});if(ps.length)topPerson='一名'+ps[0][0];}}catch(_){}}let mood='';if(activeDays<1)mood='暂无记录';else if(sessions<3)mood='平静';else if(sessions<6)mood='还行';else if(sessions<10)mood='充实';else mood='忙碌';sendJson(res,200,{sessions,activeDays,mood,totalFrames,topPlace,topPerson,sight});});

  router.get('/api/memory/report', (req,res)=>{const s=agg_st();const db=agg_db();let topPlaces=[],people=[],keywords=[],moodBars=[];if(db){try{const pr=db.prepare("SELECT frame_path,COUNT(*)as cnt FROM memories GROUP BY frame_path ORDER BY cnt DESC LIMIT 8").all();const mc=pr.length?pr[0].cnt:1;const pal=['#7E8FC4','#6FA15E','#F2A03D','#3E9B96','#FFB07C','#F97C6E','#D4708A','#5B6B52'];topPlaces=pr.map((r,i)=>({name:agg_cls((r.frame_path||'').replace(/^.*?\//,'')),count:r.cnt+' 段',bar:`width:${Math.round(r.cnt/mc*100)}%;background:${pal[i%pal.length]};border-radius:999px;`}));const fr=db.prepare("SELECT name,count FROM faces ORDER BY count DESC LIMIT 4").all();const fb=['linear-gradient(135deg,#FFD27A,#F2A03D)','linear-gradient(135deg,#7E8FC4,#9A6FB0)','linear-gradient(135deg,#F0A6B6,#D4708A)','linear-gradient(135deg,#FFB07C,#F97C6E)'];people=fr.map((r,i)=>({initial:(r.name||'-')[0],name:r.name||'--',sub:'',count:r.count+' 次',bg:fb[i%fb.length]}));const kr=db.prepare("SELECT frame_path,COUNT(*)as cnt FROM memories GROUP BY frame_path ORDER BY cnt DESC LIMIT 15").all();keywords=kr.map((r,i)=>({text:agg_cls((r.frame_path||'').replace(/^.*?\//,'')),style:`font-size:${Math.min(24,12+r.cnt*2)}px;color:#33312b;border-radius:999px;padding:4px 12px;`}));moodBars=Array.from({length:30},(_,i)=>({style:`flex:1;height:${Math.round((0.3+Math.random()*0.7)*100)}%;background:${['#6FA15E','#F2A03D','#7E8FC4','#FFB07C','#D4708A'][i%5]};border-radius:5px 5px 2px 2px;`}));}catch(_){}}sendJson(res,200,{stats:[{value:String(s.batchCount||'--'),label:'重建空间',color:'#1c1d19'},{value:(s.processedFrames>0?(s.processedFrames/3600).toFixed(1)+'h':'--'),label:'记忆总时长',color:'#1c1d19'},{value:s.batchCount>0?'+ '+s.batchCount:'--',label:'新重建空间',color:'#5E7A18'},{value:String(s.memoryCount||'--'),label:'识别物件',color:'#C7700E'}],reportStats:[{value:String(s.batchCount||'--'),label:'重建空间',color:'#1c1d19'},{value:(s.processedFrames>0?(s.processedFrames/3600).toFixed(1)+'h':'--'),label:'记忆总时长',color:'#1c1d19'},{value:s.batchCount>0?'+ '+s.batchCount:'--',label:'新重建空间',color:'#5E7A18'},{value:String(s.memoryCount||'--'),label:'识别物件',color:'#C7700E'}],moodBars,keywords:keywords.length?keywords:[{text:'等待数据...',style:'font-size:16px;color:#8A8578;'}],topPlaces:topPlaces.length?topPlaces:[{name:'等待数据...',count:'--',bar:'width:0%'}],people:people.length?people:[{initial:'-',name:'等待数据...',sub:'',count:'--',bg:'linear-gradient(135deg,#ccc,#aaa)'}],days:Array.from({length:31},(_,i)=>({dot:['#6FA15E','#F2A03D','#7E8FC4','#FFB07C','#D4708A'][i%5],label:(i+1)+'日',value:'--'}))});});

  // crops — 同时支持 stream/ 下和非 stream 下的裁剪图（用正则匹配，router 不支持 :param 语法）
  router.get(/^\/crops\/(.+)$/, (req,res,info)=>{
    const fname=info.params[1]||new URL(req.url,'http://localhost').pathname.split('/').pop();
    const fs=require('fs');
    const dirs=fs.readdirSync(AGG_DATA_DIR).filter(d=>{const dp=path.join(AGG_DATA_DIR,d);return fs.statSync(dp).isDirectory()&&!d.startsWith('.');});
    for(const d of dirs){
      for(const sub of ['objects_img_crop','stream/objects_img_crop']){
        const cp=path.join(AGG_DATA_DIR,d,sub,fname);
        if(fs.existsSync(cp)){
          const ct=fname.endsWith('.jpg')?'image/jpeg':'image/png';
          res.writeHead(200,{'Content-Type':ct,'Cache-Control':'no-cache'});
          return fs.createReadStream(cp).pipe(res);
        }
      }
    }
    res.writeHead(404);res.end('Crop not found');
  });

  // frames — 静态帧文件服务
  router.any((p) => p.startsWith('/frames/'), (req, res, info) => {
    const fname = info.url.pathname.split('/').pop();
    const framePath = path.join(FRAMES_DIR, fname);
    if (fs.existsSync(framePath)) {
      const ct = fname.endsWith('.jpg') ? 'image/jpeg' : 'image/png';
      res.writeHead(200, { 'Content-Type': ct, 'Cache-Control': 'no-cache, no-store, must-revalidate', 'Pragma': 'no-cache' });
      fs.createReadStream(framePath).pipe(res);
      return;
    }
    res.writeHead(404); res.end('Not found');
  });

  // 单条记忆 wildcard（必须放在其他 /api/memory/* 路由之后）
  router.any((p) => p.startsWith('/api/memory/'), async (req, res, info) => {
    const id = info.url.pathname.split('/').pop().replace('.json', '');
    const memory = await ctx.memoryStore.get(id);
    if (memory) sendJson(res, 200, memory);
    else sendJson(res, 404, { error: 'Not found' });
  });

}
module.exports = { registerMemoryRoutes, listFrames };
