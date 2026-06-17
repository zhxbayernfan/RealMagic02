// Sentrix Monitor · VSLAM 重建子视图运行时（流式渲染版）
// 与 src/web/assets/spatial.js 一致的核心模式：
//   - prefetch queue + 并发上限（不堆积请求）
//   - 每帧拉 /frame/{idx}/camera，相机平滑跟随当前帧位姿
//   - NaN/Infinity 过滤、可调 downsample、共享 PointsMaterial
//   - status='completed' && fetch 追平 → 自动停 + 解锁 OrbitControls
//
// 协议：
//   POST /batch/{id}/start_inference
//   POST /batch/{id}/frames           multipart files=  (单帧 JPEG)
//   GET  /batch/{id}/status           { status, processed_frames, uploaded_frames, ... }
//   GET  /batch/{id}/frame/{idx}/point_cloud   gzip 二进制
//          [uint32 N][N*12B float32 xyz][N*12B float32 rgb01][N*4B float32 conf]
//   GET  /batch/{id}/frame/{idx}/camera        { camera:{focal,pp,R_c2w,t_c2w,image_w,image_h} }
//   POST /batch/{id}/finish_inference

import * as THREE from '/assets/lib/three.module.js';
import { OrbitControls } from '/assets/lib/controls/OrbitControls.js';

// ===== 可调参数（与主项目 spatial.js 对齐）=====
const FRAME_UPLOAD_FPS = 2;
const STATUS_POLL_MS   = 500;
const MAX_CONCURRENT   = 4;        // 同时 in-flight 的 fetch 数
const CONF_FILTER      = 0.5;      // ← 对齐 guiConfThreshold
const DOWNSAMPLE       = 8;        // ← 对齐 guiDownsample
const POINT_SIZE       = 0.00002;  // ← 对齐 guiPointSize
const FOLLOW_BACK      = 0.5;      // 相机后退距离（沿 -forward）
const FOLLOW_DOWN      = 0.3;      // 相机下偏（沿 -up）
const FOLLOW_AHEAD     = 2.0;      // lookAt 在相机前方
const FOLLOW_LERP_RATE = 6.0;      // 每秒衰减系数
const DGSG_POLL_MS     = 1500;     // finish 之后等 dgsg_status='done' 的轮询间隔

let initialized = false;

const ui = {
  canvas: null, stage: null, seg: null,
  cloudOverlay: null, gsWrap: null,
  ovPose: null, ovLoop: null, ovMode: null,
  sPoints: null, sCover: null, sRecog: null,
  btnStart: null, btnStop: null, btnFollow: null, btnFull: null, statusBar: null,
  toast: null,
};

const three = {
  scene: null, camera: null, renderer: null, controls: null,
  framePoints: [],            // [{ idx, points }]
  sharedMaterial: null,
  totalPoints: 0,
};

const session = {
  active: false,
  batchId: null,
  uploadTimer: null,
  pollTimer: null,
  dgsgTimer: null,
  inflight: 0,
  fetchAt: 0,                 // 下一帧待请求 idx
  processedFrames: 0,
  uploadedFrames: 0,
  status: 'idle',
  finished: false,            // finish_inference 已发出
  completed: false,           // 全部完成（含 dgsg done + 全局点云已加载）
  pendingCams: new Map(),
  followEnabled: true,
  followFrameIdx: -1,
  followSmoothPos: null,
  followSmoothLook: null,
  followTargetPos: null,
  followTargetLook: null,
  lastFollowMs: 0,
  getVideo: null,
  mode: 'cloud',
};

// ─────────────────────────────────────────────────────────────
export function install(opts) {
  if (initialized) return;
  initialized = true;
  session.getVideo = opts && opts.getVideo;

  ui.canvas  = document.getElementById('cloudCanvas');
  ui.stage   = ui.canvas && ui.canvas.parentElement;
  ui.seg     = document.getElementById('seeSeg');
  ui.ovPose  = document.getElementById('ov-pose');
  ui.ovLoop  = document.getElementById('ov-loop');
  ui.ovMode  = document.getElementById('ov-mode');
  ui.sPoints = document.getElementById('s-points');
  ui.sCover  = document.getElementById('s-cover');
  ui.sRecog  = document.getElementById('s-recog');

  if (!ui.canvas || !ui.stage) {
    console.warn('[sentrix-vslam] #cloudCanvas / see-stage 缺失');
    return;
  }

  attachStatusUI();
  attachButtons();
  bindModeSeg();
  initThree();

  setPose('待启动');
  setMode('点云 POINTS');
  setLoop('—');
}

// ─────────────────────────────────────────────────────────────
function attachStatusUI() {
  const bar = document.createElement('div');
  bar.style.cssText = `
    position:absolute; left:10px; bottom:10px; z-index:5;
    display:flex; gap:8px; align-items:center;
    font: 600 11px/1 -apple-system,BlinkMacSystemFont,sans-serif;
    pointer-events:auto;
  `;
  ui.statusBar = bar;
  ui.stage.appendChild(bar);

  ui.btnStart  = makeBtn('● 开始采集', '#B3FF38', '#1b2400');
  ui.btnStop   = makeBtn('■ 结束采集', '#2a2a2a', '#fff');
  ui.btnFollow = makeBtn('🎯 跟随相机', '#1f2a17', '#B3FF38');
  ui.btnFull = makeBtn('⛶ 全屏', '#1f2a17', '#B3FF38');
  ui.btnStop.disabled = true;
  ui.btnStop.style.opacity = '0.5';
  bar.appendChild(ui.btnStart);
  bar.appendChild(ui.btnStop);
  bar.appendChild(ui.btnFollow);
  bar.appendChild(ui.btnFull);

  ui.toast = document.createElement('div');
  ui.toast.style.cssText = `
    position:absolute; left:50%; top:50%; transform:translate(-50%,-50%);
    padding:8px 14px; border-radius:8px; background:rgba(0,0,0,0.7);
    color:#fff; font:500 12px/1.4 -apple-system,sans-serif;
    pointer-events:none; opacity:0; transition:opacity .25s; z-index:6;
  `;
  ui.stage.appendChild(ui.toast);

  // 加载/完成 覆盖面板：中央，更醒目
  ui.loadingPanel = document.createElement('div');
  ui.loadingPanel.style.cssText = `
    position:absolute; left:50%; top:50%; transform:translate(-50%,-50%);
    min-width:220px; padding:18px 26px; border-radius:12px;
    background:rgba(15,16,13,0.85); border:1px solid rgba(179,255,56,0.35);
    color:#fff; font:600 14px/1.5 -apple-system,sans-serif;
    text-align:center; pointer-events:none;
    opacity:0; transition:opacity .3s; z-index:7;
    backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);
  `;
  ui.loadingPanel.innerHTML = `
    <div style="display:flex;align-items:center;justify-content:center;gap:10px;margin-bottom:6px;">
      <span class="sx-spinner" style="
        width:14px;height:14px;border-radius:50%;
        border:2px solid rgba(179,255,56,0.25);border-top-color:#B3FF38;
        animation:sx-spin 1s linear infinite;display:inline-block;
      "></span>
      <span class="sx-loading-title" style="color:#B3FF38;letter-spacing:0.5px;">加载中</span>
    </div>
    <div class="sx-loading-sub" style="font-weight:400;font-size:12px;color:rgba(255,255,255,0.7);"></div>
  `;
  ui.stage.appendChild(ui.loadingPanel);
  // spinner 动画注入到 head（一次）
  if (!document.getElementById('sx-keyframes')) {
    const sty = document.createElement('style');
    sty.id = 'sx-keyframes';
    sty.textContent = '@keyframes sx-spin{to{transform:rotate(360deg)}}';
    document.head.appendChild(sty);
  }
}

function showLoading(title, sub, accent) {
  if (!ui.loadingPanel) return;
  const t = ui.loadingPanel.querySelector('.sx-loading-title');
  const s = ui.loadingPanel.querySelector('.sx-loading-sub');
  const sp = ui.loadingPanel.querySelector('.sx-spinner');
  if (t) {
    t.textContent = title || '加载中';
    t.style.color = accent || '#B3FF38';
  }
  if (s) s.textContent = sub || '';
  if (sp) sp.style.display = (accent === 'done') ? 'none' : 'inline-block';
  if (accent === 'done') {
    if (t) t.style.color = '#B3FF38';
    if (sp) sp.style.display = 'none';
  }
  ui.loadingPanel.style.opacity = '1';
}

function hideLoading(delayMs) {
  if (!ui.loadingPanel) return;
  setTimeout(() => { ui.loadingPanel.style.opacity = '0'; }, delayMs || 0);
}

function makeBtn(text, bg, color) {
  const b = document.createElement('button');
  b.textContent = text;
  b.style.cssText = `
    padding: 6px 12px; border-radius: 6px; border: 0;
    background: ${bg}; color: ${color};
    font: inherit; cursor: pointer; letter-spacing: .5px;
  `;
  return b;
}

function showToast(msg, ms) {
  ui.toast.textContent = msg;
  ui.toast.style.opacity = '1';
  clearTimeout(ui.toast._t);
  ui.toast._t = setTimeout(() => { ui.toast.style.opacity = '0'; }, ms || 1800);
}

function attachButtons() {
  ui.btnStart.addEventListener('click', () => {
    if (session.active) return;
    if (!session.getVideo || !session.getVideo()) {
      showToast('请先允许摄像头权限');
      return;
    }
    startCapture().catch((e) => {
      console.error('[sentrix-vslam] start failed', e);
      showToast('启动失败：' + (e.message || e));
      resetButtons(false);
    });
  });
  ui.btnStop.addEventListener('click', () => {
    if (!session.active && !gs.active) return;
    stopCapture().catch((e) => {
      console.error('[sentrix-vslam] stop failed', e);
      showToast('结束失败：' + (e.message || e));
    });
  });
  ui.btnFollow.addEventListener('click', () => {
    setFollowEnabled(!session.followEnabled);
  });
  ui.btnFull.addEventListener('click', toggleFullscreen);
  setFollowEnabled(true);
}

function resetButtons(active) {
  ui.btnStart.disabled = active;
  ui.btnStart.style.opacity = active ? '0.5' : '1';
  ui.btnStop.disabled = !active;
  ui.btnStop.style.opacity = active ? '1' : '0.5';
}

function setFollowEnabled(on) {
  session.followEnabled = on;
  ui.btnFollow.style.background = on ? '#B3FF38' : '#1f2a17';
  ui.btnFollow.style.color      = on ? '#1b2400' : '#B3FF38';
  if (three.controls) three.controls.enabled = !on;
  if (!on) {
    session.followSmoothPos = null;
    session.followSmoothLook = null;
  }
}

let _fullscreenActive = false;
let _fullscreenPrevStyles = null;
function toggleFullscreen() {
  // 尝试原生 Fullscreen API（如果可用且非 iframe）
  if (!_fullscreenActive && document.fullscreenEnabled && !(window.parent && window.parent !== window)) {
    ui.stage.requestFullscreen().catch(() => {});
    return;
  }

  _fullscreenActive = !_fullscreenActive;
  if (_fullscreenActive) {
    // expand see-stage to fill viewport
    _fullscreenPrevStyles = {
      position: ui.stage.style.position || '',
      inset: ui.stage.style.inset || '',
      zIndex: ui.stage.style.zIndex || '',
      borderRadius: ui.stage.style.borderRadius || '',
    };
    Object.assign(ui.stage.style, {
      position: 'fixed',
      inset: '0',
      zIndex: '9999',
      borderRadius: '0',
    });
    ui.btnFull.textContent = '✕ 退出全屏';
    ui.btnFull.style.background = '#B3FF38';
    ui.btnFull.style.color = '#1b2400';
  } else {
    Object.assign(ui.stage.style, _fullscreenPrevStyles || {});
    _fullscreenPrevStyles = null;
    ui.btnFull.textContent = '⛶ 全屏';
    ui.btnFull.style.background = '#1f2a17';
    ui.btnFull.style.color = '#B3FF38';
  }
  // 通知 renderer 尺寸变了
  setTimeout(() => {
    if (three.renderer) {
      const rr = ui.stage.getBoundingClientRect();
      three.renderer.setSize(rr.width, rr.height, false);
      if (three.camera) {
        three.camera.aspect = rr.width / rr.height;
        three.camera.updateProjectionMatrix();
      }
    }
  }, 100);
}

// 响应原生全屏事件更新按钮
function _onFsChange() {
  if (!document.fullscreenElement && _fullscreenActive) toggleFullscreen();
}
document.addEventListener('fullscreenchange', _onFsChange);

// ─────────────────────────────────────────────────────────────
function bindModeSeg() {
  if (!ui.seg) return;
  ui.seg.querySelectorAll('button').forEach((b) => {
    b.addEventListener('click', () => {
      const m = b.dataset.mode;
      session.mode = m;
      ui.seg.querySelectorAll('button').forEach((x) => x.classList.remove('on'));
      b.classList.add('on');
      if (m === 'cloud') {
        showRenderer('cloud');
        setMode('点云 POINTS');
      } else if (m === 'gauss') {
        showRenderer('gauss');
        setMode('3DGS');
        showToast('3DGS 模式：结束采集后用本地数据生成');
      } else if (m === 'depth') {
        showRenderer('cloud');
        setMode('深度（待实现）');
        showToast('深度模式待接入');
      }
    });
  });
}

function showRenderer(which) {
  // 用 visibility 而非 display，避免 hidden viewer 初始化时尺寸为 0
  if (ui.cloudOverlay) {
    ui.cloudOverlay.style.visibility = (which === 'gauss') ? 'hidden' : 'visible';
    ui.cloudOverlay.style.pointerEvents = (which === 'gauss') ? 'none' : 'auto';
  }
  if (ui.gsWrap) {
    ui.gsWrap.style.visibility = (which === 'gauss') ? 'visible' : 'hidden';
    ui.gsWrap.style.pointerEvents = (which === 'gauss') ? 'auto' : 'none';
  }
  if (which === 'gauss' && !ui.gsWrap) showToast('3DGS 尚未生成，结束采集后会显示');
}

// ─────────────────────────────────────────────────────────────
function initThree() {
  // 不抢 mockup #cloudCanvas（已被 mockup 2D 占用）；新建覆盖 canvas
  const overlay = document.createElement('canvas');
  overlay.id = 'sx-cloud-overlay';
  Object.assign(overlay.style, {
    position: 'absolute', inset: '0', width: '100%', height: '100%',
    zIndex: '2', background: '#0F100D',
  });
  ui.stage.appendChild(overlay);
  ui.cloudOverlay = overlay;
  ui.stage.querySelectorAll('.cam-hud, .see-overlay').forEach((el) => {
    el.style.zIndex = '3';
  });

  const r = ui.stage.getBoundingClientRect();
  three.renderer = new THREE.WebGLRenderer({ canvas: overlay, antialias: false });
  three.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  three.renderer.setSize(r.width, r.height, false);
  three.renderer.setClearColor(0x0F100D, 1);

  three.scene = new THREE.Scene();
  three.camera = new THREE.PerspectiveCamera(60, r.width / r.height, 0.05, 200);
  // VGGT/lingbot 输出 OpenCV 风格相机系 (+Y 朝下，+Z 朝前)，
  // THREE 默认 OpenGL 风格 (+Y 朝上)。直接渲染会上下翻转。
  // 跟主项目 spatial.js 一致：把 camera.up 翻成 (0, -1, 0)。
  three.camera.up.set(0, -1, 0);
  three.camera.position.set(0, -0.5, 2.5);
  three.camera.lookAt(0, 0, 0);

  three.controls = new OrbitControls(three.camera, overlay);
  three.controls.enableDamping = true;
  three.controls.dampingFactor = 0.08;
  three.controls.enabled = !session.followEnabled;

  three.sharedMaterial = new THREE.PointsMaterial({
    size: POINT_SIZE,
    vertexColors: true,
    sizeAttenuation: true,
    transparent: false,
    depthTest: true,
    depthWrite: true,
  });

  const axes = new THREE.AxesHelper(0.3);
  axes.material.transparent = true;
  axes.material.opacity = 0.4;
  three.scene.add(axes);

  new ResizeObserver(() => {
    const rr = ui.stage.getBoundingClientRect();
    three.renderer.setSize(rr.width, rr.height, false);
    three.camera.aspect = rr.width / rr.height;
    three.camera.updateProjectionMatrix();
  }).observe(ui.stage);

  (function tick() {
    const now = performance.now();
    if (session.followEnabled && session.followTargetPos && session.followTargetLook) {
      stepFollow(now);
    } else {
      three.controls.update();
    }
    three.renderer.render(three.scene, three.camera);
    requestAnimationFrame(tick);
  })();
}

function stepFollow(nowMs) {
  if (!session.followSmoothPos) {
    session.followSmoothPos = session.followTargetPos.clone();
    session.followSmoothLook = session.followTargetLook.clone();
    session.lastFollowMs = nowMs;
  }
  const dt = Math.max(0, (nowMs - session.lastFollowMs) / 1000);
  session.lastFollowMs = nowMs;
  const k = 1 - Math.exp(-FOLLOW_LERP_RATE * dt);
  session.followSmoothPos.lerp(session.followTargetPos, k);
  session.followSmoothLook.lerp(session.followTargetLook, k);
  three.camera.position.copy(session.followSmoothPos);
  // 强制世界系 up = (0, -1, 0)，避免 lookAt 默认用 +Y 又把画面翻回去
  three.camera.up.set(0, -1, 0);
  three.camera.lookAt(session.followSmoothLook);
}

// ─────────────────────────────────────────────────────────────
async function startCapture() {
  // 清残留 timer（保险）
  if (session.uploadTimer) clearInterval(session.uploadTimer);
  if (session.pollTimer)   clearInterval(session.pollTimer);
  if (session.dgsgTimer)   clearInterval(session.dgsgTimer);
  session.uploadTimer = session.pollTimer = session.dgsgTimer = null;

  // 固定扫描目录：每次 start_inference 后端会清理 data/jszn 并覆盖旧数据
  session.batchId = 'jszn';
  session.fetchAt = 0;
  session.processedFrames = 0;
  session.uploadedFrames = 0;
  session.inflight = 0;
  session.completed = false;
  session.finished = false;
  session.status = 'idle';
  session.pendingCams.clear();
  session.followFrameIdx = -1;
  session.followSmoothPos = null;
  session.followSmoothLook = null;
  three.totalPoints = 0;
  clearAllPoints();
  setFollowEnabled(true);

  setPose('启动中…');
  setMode('点云 POINTS');
  showToast('开始采集');
  showLoading('模型加载中', '等待 lingbot scale 估计完成（前 ~8 帧）…');

  const r = await fetch('/batch/' + session.batchId + '/start_inference', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}',
  });
  if (!r.ok) {
    hideLoading();
    throw new Error('start_inference HTTP ' + r.status);
  }

  session.active = true;
  resetButtons(true);
  setPose('采集中');

  const periodMs = 1000 / FRAME_UPLOAD_FPS;
  session.uploadTimer = setInterval(uploadOneFrame, periodMs);
  session.pollTimer   = setInterval(pollOnce, STATUS_POLL_MS);
  pollOnce();  // 立即拉一次状态
}

async function stopCapture() {
  if (!session.active) return;
  session.active = false;

  if (session.uploadTimer) clearInterval(session.uploadTimer);
  session.uploadTimer = null;

  setPose('finish 触发中…');
  showToast('结束采集，等待剩余帧 + VLM…');
  resetButtons(false);
  ui.btnStart.disabled = true;
  ui.btnStart.style.opacity = '0.5';
  session.finished = true;

  // 3DGS 模式：不等 VLM / DGSG 完整结束，直接用当前 batch 的 frames 触发本地生成。
  // 这走 server.js 里的 /api/gaussian-splats/generate，使用已采集好的 data/<batch>/frames。
  if (shouldGenerateGS()) startLocalGSGeneration();

  // 立即启动 dgsg 轮询，不等 finish_inference 整个返回再开。
  // finish_inference 是 async 长事务（Phase1: 等剩余推理；Phase2: 等 DGSG；
  // Phase3: 卸载模型；Phase4: VLM；Phase5: 保存），整体可达数十秒~几分钟。
  // 期间 pollTimer 会持续把 Phase1 内陆续推理完的剩余帧拉下来 + 跟相机走，
  // dgsgTimer 会持续追 dgsg_status，'done' 后切全局点云。
  startDgsgWait();

  // fire-and-forget：finish 调用在后台跑，不阻塞 stopCapture 流程
  fetch('/batch/' + session.batchId + '/finish_inference', { method: 'POST' })
    .then((r) => {
      if (!r.ok) throw new Error('finish_inference HTTP ' + r.status);
      setPose('finish 完成 · 等 VLM…');
    })
    .catch((e) => {
      console.error('[sentrix-vslam] finish failed', e);
      showToast('finish 异常：' + (e.message || e));
      setPose('finish 失败');
    })
    .finally(() => {
      ui.btnStart.disabled = false;
      ui.btnStart.style.opacity = '1';
    });
}

// ─────────────────────────────────────────────────────────────
// finish 之后：等 dgsg_status='done' → 加载全局点云替换所有累积帧
// ─────────────────────────────────────────────────────────────
function startDgsgWait() {
  if (session.dgsgTimer) return;
  setPose('VLM 处理中…');
  session.dgsgTimer = setInterval(async () => {
    try {
      const r = await fetch('/batch/' + session.batchId + '/dgsg_status');
      if (!r.ok) return;
      const j = await r.json();
      const st = (j && j.status) || 'idle';
      if (st === 'done') {
        clearInterval(session.dgsgTimer);
        session.dgsgTimer = null;
        loadGlobalPointCloud().catch((e) => {
          console.error('[sentrix-vslam] global pc load failed', e);
          showToast('全局点云加载失败');
          setPose('加载失败');
        });
      } else if (st === 'error') {
        clearInterval(session.dgsgTimer);
        session.dgsgTimer = null;
        setPose('VLM 失败');
        showToast('后端 VLM 处理失败');
      } else {
        setPose('VLM ' + st);
      }
    } catch (_) {}
  }, DGSG_POLL_MS);
}

async function loadGlobalPointCloud() {
  setPose('加载全局点云…');
  showToast('加载全局语义点云…', 3000);
  const r = await fetch('/batch/' + session.batchId + '/dgsg_pointcloud');
  if (!r.ok) throw new Error('dgsg_pointcloud HTTP ' + r.status);
  const buf = await r.arrayBuffer();
  // 格式：[uint32 N][N*12B float32 xyz][N*12B float32 rgb][N*2B uint16 obj_idx]
  const dv = new DataView(buf);
  const N = dv.getUint32(0, true);
  if (N === 0) { setPose('全局点云为空'); return; }

  const OFFSET_POS = 4;
  const OFFSET_COL = 4 + N * 12;
  const positions = new Float32Array(buf, OFFSET_POS, N * 3);
  const colors    = new Float32Array(buf, OFFSET_COL, N * 3);

  // 累积帧 → 全局：清掉累积帧
  clearAllPoints();

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('color',    new THREE.BufferAttribute(colors, 3));
  geom.boundingSphere = new THREE.Sphere(new THREE.Vector3(), Infinity);
  const points = new THREE.Points(geom, three.sharedMaterial);
  three.scene.add(points);
  three.framePoints.push({ idx: -1, points });    // -1 标记为全局
  three.totalPoints = N;
  setPointCount(N);

  setPose('全局点云已就位');
  setMode('点云 POINTS · 全局');
  setFollowEnabled(false);
  showToast('共 ' + formatPoints(N) + ' 点（全局语义）', 3000);
  showLoading('渲染完成', '全局语义点云 ' + formatPoints(N) + ' 点 · 可自由查看', 'done');
  hideLoading(3000);
  if (session.pollTimer) { clearInterval(session.pollTimer); session.pollTimer = null; }
  session.completed = true;

  // 如果用户当前选了 3DGS 模式，用本次采集好的 frames 触发本地 gaussian-splats 生成
  if (shouldGenerateGS()) startLocalGSGeneration();
}

// ─────────────────────────────────────────────────────────────
async function uploadOneFrame() {
  if (!session.active || document.hidden) return;
  const v = session.getVideo && session.getVideo();
  if (!v || !v.videoWidth) return;

  const w = v.videoWidth, h = v.videoHeight;
  const off = (typeof OffscreenCanvas !== 'undefined')
    ? new OffscreenCanvas(w, h)
    : Object.assign(document.createElement('canvas'), { width: w, height: h });
  off.getContext('2d').drawImage(v, 0, 0, w, h);
  const blob = off.convertToBlob
    ? await off.convertToBlob({ type: 'image/jpeg', quality: 0.75 })
    : await new Promise((res) => off.toBlob(res, 'image/jpeg', 0.75));

  const fd = new FormData();
  fd.append('files', blob, 'frame.jpg');
  try {
    const r = await fetch('/batch/' + session.batchId + '/frames', { method: 'POST', body: fd });
    if (r.ok) {
      const j = await r.json();
      session.uploadedFrames = j.total_frames || (session.uploadedFrames + 1);
    }
  } catch (e) {
    console.warn('[sentrix-vslam] upload failed', e);
  }
}

// ─────────────────────────────────────────────────────────────
async function pollOnce() {
  // 检查 status，更新计数器，然后驱动 fetch queue
  let st = null;
  try {
    const r = await fetch('/batch/' + session.batchId + '/status');
    if (r.ok) st = await r.json();
  } catch (e) {
    console.warn('[sentrix-vslam] status poll failed', e);
  }

  if (st) {
    const prevProcessed = session.processedFrames;
    session.processedFrames = st.processed_frames || 0;
    session.uploadedFrames  = st.uploaded_frames  || session.uploadedFrames;
    session.status          = st.status || 'unknown';
    setLoop(String(session.processedFrames));
    const cover = Math.min(100,
      Math.round(session.processedFrames /
        Math.max(1, session.uploadedFrames || 1) * 100));
    if (ui.sCover) ui.sCover.firstChild.nodeValue = String(cover);

    // finish 期间剩余帧反馈：让用户看到帧数还在涨 → 视角在跟着走
    if (session.finished && session.processedFrames > prevProcessed) {
      console.log('[sentrix-vslam] finish 期间拉到新帧',
        prevProcessed, '→', session.processedFrames);
      setPose('渲染剩余帧 ' + session.processedFrames + '/' + session.uploadedFrames);
    }
  }

  // 触发 fetch queue（在并发上限内补到上限）
  while (session.inflight < MAX_CONCURRENT &&
         session.fetchAt < session.processedFrames) {
    const idx = session.fetchAt++;
    session.inflight++;
    fetchFrameAndCamera(idx).finally(() => {
      session.inflight--;
    });
  }

  // 剩余帧渲染完毕 → 自动退出相机跟随，让用户能拖着看累积点云。
  // dgsgTimer 仍在后台等，dgsg='done' 时 loadGlobalPointCloud 会无缝替换。
  if (session.finished && session.followEnabled &&
      session.fetchAt >= session.processedFrames &&
      session.inflight === 0 &&
      session.processedFrames > 0) {
    setFollowEnabled(false);
    setPose('剩余帧已渲染 · 等 VLM');
    showToast('点云已就位，可拖动查看', 2000);
  }

  // 注意：不要在 status='completed' 时停 polling。
  // 后端 frame_monitor_thread 退出时就会把 status 设成 'completed'，
  // 但此时 finish_inference 内部的 dgsg+VLM 还没跑完。
  // 真正的"全部完成"由 loadGlobalPointCloud() 在 dgsg_status='done' 后负责
  // 清 pollTimer 并 session.completed=true。
}

async function fetchFrameAndCamera(idx) {
  // 并发拉点云 + 相机
  const pcP  = fetch('/batch/' + session.batchId + '/frame/' + idx + '/point_cloud');
  const camP = fetch('/batch/' + session.batchId + '/frame/' + idx + '/camera');
  let pcResp, camResp;
  try {
    [pcResp, camResp] = await Promise.all([pcP, camP]);
  } catch (e) {
    console.warn('[sentrix-vslam] fetch frame', idx, 'failed', e);
    return;
  }
  if (!pcResp.ok) return;

  const buf = await pcResp.arrayBuffer();
  let cam = null;
  if (camResp.ok) {
    try { const cj = await camResp.json(); cam = cj && cj.camera; } catch (_) {}
  }

  addFrameToScene(idx, buf, cam);
}

// ─────────────────────────────────────────────────────────────
function addFrameToScene(idx, buf, cam) {
  const view = new DataView(buf);
  const N = view.getUint32(0, true);
  if (N === 0) return;
  const positions = new Float32Array(buf, 4, N * 3);
  const colors    = new Float32Array(buf, 4 + N * 12, N * 3);
  const confs     = new Float32Array(buf, 4 + N * 24, N);

  // NaN/Inf/极端坐标 + downsample + conf 过滤（与 spatial.js:1700-1722 一致）
  const stride = Math.max(1, DOWNSAMPLE);
  const pos = new Float32Array(Math.ceil(N / stride) * 3);
  const col = new Float32Array(Math.ceil(N / stride) * 3);
  let count = 0;
  for (let i = 0; i < N; i += stride) {
    const i3 = i * 3;
    const x = positions[i3], y = positions[i3 + 1], z = positions[i3 + 2];
    if (x !== x || (x > 1e10 || x < -1e10)) continue;
    if (y !== y || (y > 1e10 || y < -1e10)) continue;
    if (z !== z || (z > 1e10 || z < -1e10)) continue;
    if (confs[i] <= CONF_FILTER) continue;
    const o = count * 3;
    pos[o] = x; pos[o + 1] = y; pos[o + 2] = z;
    col[o] = colors[i3]; col[o + 1] = colors[i3 + 1]; col[o + 2] = colors[i3 + 2];
    count++;
  }
  if (count === 0) return;

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(pos.subarray(0, count * 3), 3));
  geom.setAttribute('color',    new THREE.BufferAttribute(col.subarray(0, count * 3), 3));
  // 防止 boundingSphere 计算 NaN 顶飞 OrbitControls
  geom.boundingSphere = new THREE.Sphere(new THREE.Vector3(), Infinity);

  const points = new THREE.Points(geom, three.sharedMaterial);
  three.scene.add(points);
  three.framePoints.push({ idx, points });
  three.totalPoints += count;
  setPointCount(three.totalPoints);

  // 首帧到达 → 关闭"模型加载中"提示
  if (three.framePoints.length === 1) {
    hideLoading();
  }

  if (cam) updateFollowTarget(idx, cam);
}

function updateFollowTarget(idx, cam) {
  const t = cam.t_c2w;
  const R = cam.R_c2w;
  if (!t || !R || t.length < 3 || R.length < 3) return;
  // R 是 3x3 嵌套数组，列优先取 forward = R[:,2], up = R[:,1]
  const camPos = new THREE.Vector3(t[0], t[1], t[2]);
  const forward = new THREE.Vector3(R[0][2], R[1][2], R[2][2]).normalize();
  const up      = new THREE.Vector3(R[0][1], R[1][1], R[2][1]).normalize();
  const viewPos = camPos.clone()
    .addScaledVector(forward, -FOLLOW_BACK)
    .addScaledVector(up,      -FOLLOW_DOWN);
  const lookAt  = camPos.clone().addScaledVector(forward, FOLLOW_AHEAD);

  // 总按最新帧更新 target；render loop 内部平滑过渡
  if (idx >= session.followFrameIdx) {
    session.followFrameIdx = idx;
    session.followTargetPos  = viewPos;
    session.followTargetLook = lookAt;
  }
}

function clearAllPoints() {
  for (const e of three.framePoints) {
    three.scene.remove(e.points);
    e.points.geometry.dispose();
    // 共享 material 不 dispose
  }
  three.framePoints.length = 0;
  setPointCount(0);
}

// ─────────────────────────────────────────────────────────────
function setPose(t) { if (ui.ovPose) ui.ovPose.textContent = t; }
function setMode(t) { if (ui.ovMode) ui.ovMode.textContent = t; }
function setLoop(t) { if (ui.ovLoop) ui.ovLoop.textContent = t; }

function formatPoints(n) {
  return n >= 1e6 ? (n / 1e6).toFixed(2) + 'M'
       : n >= 1e3 ? (n / 1e3).toFixed(1) + 'K'
       : String(n);
}

function setPointCount(n) {
  if (!ui.sPoints) return;
  const big = n >= 1e6 ? (n / 1e6).toFixed(2) : n >= 1e3 ? (n / 1e3).toFixed(1) : String(n);
  const unit = n >= 1e6 ? 'M' : n >= 1e3 ? 'K' : '';
  ui.sPoints.innerHTML = big + (unit ? ('<small>' + unit + '</small>') : '');
}

// ─────────────────────────────────────────────────────────────
// 3DGS 模式：旁路上传到远端 FastGS host，结束后训练 + 拉 PLY
// 通过 /fastgs/{id}/{frame|finish|ply} 由 server.js 转发到远端
// ─────────────────────────────────────────────────────────────
// 3DGS 模式（本地生成器）：使用已采集好的 data/<batch>/frames
// server.js 已提供：
//   POST /api/gaussian-splats/generate?batch_id=&frames_dir=
//   GET  /api/gaussian-splats/status?batch_id=
//   GET  /api/gaussian-splats/{batch_id}.ply
//
// 注意：当前 scripts/generate-gs.sh 的 RGB-only 分支会复制
// data/gaussian-splats/lingbot.ply 作为 demo；真实训练需要修 FastGS 环境。
// ─────────────────────────────────────────────────────────────
const GS_POLL_MS = 1500;

const gs = {
  pollTimer: null,
  statusPanel: null,
  batchId: null,
  running: false,
};

function shouldGenerateGS() {
  return session.mode === 'gauss';
}

function startLocalGSGeneration() {
  if (!shouldGenerateGS() || !session.batchId || gs.running) return;
  gs.running = true;
  // 固定 GS 输出：data/gaussian-splats/jszn_gs.ply / jszn_gs_status.json
  gs.batchId = 'jszn_gs';

  const framesDir = '/home/sscy/lingbot-map/stmem-psh/data/jszn/frames';
  const url = '/api/gaussian-splats/generate?batch_id=' + encodeURIComponent(gs.batchId)
    + '&frames_dir=' + encodeURIComponent(framesDir);

  attachGSStatusPanel();
  showLoading('3DGS · 生成中', '使用本地采集帧生成高斯点云…');
  setMode('3DGS · 本地生成');
  setPose('3DGS 生成中');

  fetch(url, { method: 'POST' })
    .then((r) => {
      if (!r.ok) throw new Error('generate HTTP ' + r.status);
      return r.json();
    })
    .then((j) => {
      renderGSStatus({ status: 'starting', progress: 0, message: j.message || '已启动生成任务' });
      gs.pollTimer = setInterval(pollLocalGSStatus, GS_POLL_MS);
      pollLocalGSStatus();
    })
    .catch((e) => {
      gs.running = false;
      showToast('3DGS 启动失败：' + (e.message || e));
      showLoading('3DGS 启动失败', String(e.message || e), 'done');
      hideLoading(3000);
      setPose('3DGS 失败');
    });
}

function attachGSStatusPanel() {
  if (gs.statusPanel) { gs.statusPanel.style.display = 'flex'; return; }
  const p = document.createElement('div');
  p.style.cssText = `
    position:absolute; right:10px; bottom:10px; z-index:5;
    width:340px; max-height:220px; padding:10px 12px;
    background:rgba(15,16,13,0.92); border:1px solid rgba(179,255,56,0.3);
    border-radius:8px; color:#cfd1c8;
    font: 500 11px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    display:flex; flex-direction:column; gap:8px;
    pointer-events:auto;
    backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);
  `;
  p.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;
                font:600 10px/1 -apple-system,sans-serif;letter-spacing:1px;
                color:#B3FF38;text-transform:uppercase;">
      <span>3DGS · 本地生成</span>
      <span class="sx-gs-percent" style="color:#83887C;font-weight:500;">0%</span>
    </div>
    <div style="height:6px;border-radius:999px;background:rgba(255,255,255,.12);overflow:hidden;">
      <i class="sx-gs-bar" style="display:block;height:100%;width:0%;background:#B3FF38;border-radius:999px;"></i>
    </div>
    <div class="sx-gs-msg" style="white-space:pre-wrap;word-break:break-word;">等待生成…</div>
  `;
  ui.stage.appendChild(p);
  gs.statusPanel = p;
}

function renderGSStatus(st) {
  if (!gs.statusPanel) return;
  const pct = Math.max(0, Math.min(100, Number(st.progress || 0)));
  const pctEl = gs.statusPanel.querySelector('.sx-gs-percent');
  const bar = gs.statusPanel.querySelector('.sx-gs-bar');
  const msg = gs.statusPanel.querySelector('.sx-gs-msg');
  if (pctEl) pctEl.textContent = pct + '%';
  if (bar) bar.style.width = pct + '%';
  if (msg) {
    const status = st.status || 'unknown';
    msg.textContent = '[' + status + '] ' + (st.message || '');
    if (status === 'failed') msg.style.color = '#ff6b6b';
    else if (status === 'done') msg.style.color = '#B3FF38';
    else msg.style.color = '#cfd1c8';
  }
}

async function pollLocalGSStatus() {
  if (!gs.running || !gs.batchId) return;
  let st;
  try {
    const r = await fetch('/api/gaussian-splats/status?batch_id=' + encodeURIComponent(gs.batchId));
    if (!r.ok) return;
    st = await r.json();
  } catch (_) { return; }

  renderGSStatus(st);
  if (st.status === 'done') {
    gs.running = false;
    if (gs.pollTimer) { clearInterval(gs.pollTimer); gs.pollTimer = null; }
    setPose('3DGS 生成完成');
    showLoading('渲染完成', '本地高斯点云已生成 · 切到 3DGS 渲染器', 'done');
    hideLoading(2500);
    if (gs.statusPanel) setTimeout(() => { gs.statusPanel.style.display = 'none'; }, 2000);
    openLocalGSPly(st.ply_url || ('/api/gaussian-splats/' + gs.batchId + '.ply'));
  } else if (st.status === 'failed') {
    gs.running = false;
    if (gs.pollTimer) { clearInterval(gs.pollTimer); gs.pollTimer = null; }
    setPose('3DGS 生成失败');
    showLoading('3DGS 生成失败', st.message || '未知错误', 'done');
    hideLoading(4500);
  }
}

function openLocalGSPly(plyUrl) {
  const absUrl = plyUrl.startsWith('http') ? plyUrl : (window.location.origin + plyUrl);
  showEmbeddedGSViewer(absUrl).catch((e) => {
    console.warn('[sentrix-vslam] embedded GS load failed', e);
    showToast('PLY 已就绪：' + plyUrl + '（手动加载失败，可去 GS点云 手动加载）', 8000);
  });
}

async function showEmbeddedGSViewer(absUrl) {
  // gaussian-viewer.js 当前写死 document.querySelector('.gsviewer-canvas-wrap')。
  // 在 VSLAM 左上角 see-stage 内临时创建一个同名容器，让 viewer 直接挂进来。
  let wrap = ui.stage.querySelector('.gsviewer-canvas-wrap');
  if (!wrap) {
    wrap = document.createElement('div');
    wrap.className = 'gsviewer-canvas-wrap';
    Object.assign(wrap.style, {
      position: 'absolute', inset: '0', zIndex: '2',
      background: '#0F100D', overflow: 'hidden', borderRadius: '8px',
      visibility: session.mode === 'gauss' ? 'visible' : 'hidden',
      pointerEvents: session.mode === 'gauss' ? 'auto' : 'none',
    });
    const stats = document.createElement('div');
    stats.id = 'gsStats';
    stats.textContent = 'GS Viewer 待加载';
    stats.style.cssText = `
      position:absolute; right:10px; top:10px; z-index:4;
      padding:5px 9px; border-radius:6px;
      background:rgba(0,0,0,.58); color:#B3FF38;
      font:600 11px/1.2 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
      pointer-events:none;
    `;
    wrap.appendChild(stats);
    ui.stage.appendChild(wrap);
    ui.gsWrap = wrap;
    // 保留 HUD / chips / 按钮在 GS 画面之上
    ui.stage.querySelectorAll('.cam-hud, .see-overlay').forEach((el) => { el.style.zIndex = '3'; });
    if (ui.statusBar) ui.statusBar.style.zIndex = '5';
    if (ui.toast) ui.toast.style.zIndex = '6';
    if (ui.loadingPanel) ui.loadingPanel.style.zIndex = '7';
  } else {
    ui.gsWrap = wrap;
  }

  session.mode = 'gauss';
  if (ui.seg) {
    ui.seg.querySelectorAll('button').forEach((x) => x.classList.toggle('on', x.dataset.mode === 'gauss'));
  }
  showRenderer('gauss');
  setMode('3DGS · 已就绪');
  setPose('3DGS 已加载');
  showToast('3DGS 已在独立渲染器就绪，可用上方按钮切换', 2500);

  // 动态加载 iframe 内自己的 gaussian-viewer.js。它会在当前 window 上注册 loadGSFromUrl。
  if (typeof window.loadGSFromUrl !== 'function') {
    await import('/assets/gaussian-viewer.js');
  }
  if (typeof window.loadGSFromUrl !== 'function') {
    throw new Error('loadGSFromUrl not registered');
  }
  window.loadGSFromUrl(absUrl);
}
