// gaussian-viewer.js — Aholo EGS Viewer with official CameraControl
import {
  createViewer, Scene3D, PerspectiveCamera, Vector3,
  SplatLoader, SplatUtils, setViewerConfig, BackgroundMode, Color
} from './aholo-viewer.js';
import { CameraControl } from './camera-control.js';

const { SplatFileType, parseSplatData } = SplatLoader;
const { createSplat } = SplatUtils;

// ── State ──
let viewer = null, scene = null, camera = null, currentSplat = null;
let animId = null, container = null, resizeObserver = null;
let isLoading = false, isDestroyed = false;
let camCtrl = null;

// ── Helpers ──
function $(id) { return document.getElementById(id); }
function stats(text) { const el = $('gsStats'); if (el) el.textContent = text; }

// ── Config ──
function applyViewerConfig() {
  setViewerConfig(viewer, {
    pixelRatio: 1,
    pipeline: {
      Background: {
        background: { active: BackgroundMode.BasicBackground, basic: { color: new Color(0, 0, 0), alpha: 1 } },
        ground: { enabled: false }
      },
      Splatting: {
        enabled: true,
        pack: { highPrecisionEnabled: false, precalculateEnabled: true, cameraRelativeEnabled: true },
        raster: { normalizedFalloff: false, preBlurAmount: 0.3, blurAmount: 0, focalAdjustment: 2,
                  detailCullingThreshold: 1, maxPixelRadius: 1024, maxStdDev: Math.sqrt(8) },
        composite: { enabled: false }, toneMapping: { enabled: false }, highlightKernel: { enabled: false }
      },
      TAA: { enabled: false }
    }
  });
}

// ── Hint overlay ──
function createHintBox() {
  const el = document.createElement('div');
  el.id = 'gsHintBox';
  el.innerHTML = `<div class="gs-hint-inner">
    <div class="gs-hint-title">🎮 操控提示</div>
    <div class="gs-hint-row"><kbd>左键拖拽</kbd> 旋转</div>
    <div class="gs-hint-row"><kbd>右键拖拽</kbd> 平移</div>
    <div class="gs-hint-row"><kbd>中键拖拽</kbd> 平移</div>
    <div class="gs-hint-row"><kbd>Alt</kbd>+<kbd>左键</kbd> 环绕</div>
    <div class="gs-hint-row"><kbd>滚轮</kbd> 前后移动</div>
    <div class="gs-hint-sep"></div>
    <div class="gs-hint-row"><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> 飞行移动</div>
    <div class="gs-hint-row"><kbd>Q</kbd> 下降 <kbd>E</kbd> 上升</div>
    <div class="gs-hint-row"><kbd>R</kbd><kbd>F</kbd> 左右翻滚</div>
    <div class="gs-hint-sep"></div>
    <div class="gs-hint-row"><kbd>Shift</kbd> ×10 <kbd>Ctrl</kbd> ×2 <kbd>Caps</kbd> ×20</div>
  </div>`;
  el.style.cssText = 'position:absolute;bottom:12px;left:12px;z-index:10;pointer-events:none;'
    + 'background:rgba(0,0,0,0.72);color:#ccc;padding:10px 14px;border-radius:8px;'
    + 'font-size:12px;line-height:1.7;font-family:system-ui,sans-serif;'
    + 'border:1px solid rgba(255,255,255,0.08);backdrop-filter:blur(6px);'
    + 'transition:opacity 0.4s;opacity:1;';
  container.appendChild(el);

  var style = document.createElement("style");
  style.textContent = "#gsHintBox kbd { background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.2);"
    + "border-radius:3px; padding:1px 5px; font-size:10px; font-family:inherit; color:#ddd; margin:0 1px; }"
    + "#gsHintBox .gs-hint-title { color:#fff; font-weight:600; margin-bottom:4px; font-size:12px; }"
    + "#gsHintBox .gs-hint-sep { border-top:1px solid rgba(255,255,255,0.08); margin:4px 0; }"
    + "#gsHintBox .gs-hint-row { white-space:nowrap; }";
  (container || el).appendChild(style);

  // Fade out after 8 seconds of inactivity
  let timer;
  const fade = () => { el.style.opacity = '0.25'; };
  const show = () => { el.style.opacity = '1'; clearTimeout(timer); timer = setTimeout(fade, 8000); };
  const resetTimer = () => { show(); };

  // Show on any interaction with the container
  container.addEventListener('pointerdown', resetTimer);
  container.addEventListener('wheel', resetTimer);
  container.addEventListener('keydown', resetTimer);
  show();
  setTimeout(fade, 8000);

  return el;
}

// ── Init ──
async function initViewer() {
  if (viewer) return true;
  container = document.querySelector('.gsviewer-canvas-wrap');
  if (!container) { console.error('[GS] container not found'); return false; }

  stats('初始化 EGS 引擎...');

  try {
    viewer = createViewer('gs-main-' + Date.now(), container, { antialiasing: false, alpha: false });
    console.log('[GS] createViewer OK');
  } catch (e) {
    console.error('[GS] createViewer failed:', e);
    stats('错误: ' + e.message);
    return false;
  }

  scene = new Scene3D();
  viewer.setScene(scene);

  const w = container.clientWidth || 800, h = container.clientHeight || 600;
  camera = new PerspectiveCamera(60, Math.max(w / Math.max(h, 1), 0.1), 0.1, 2000);
  camera.up.set(0, -1, 0);
  camera.position.set(0, -1.2, 3);
  camera.lookAt(new Vector3(0, 0, 0));
  viewer.setCamera(camera);
  console.log('[GS] camera set');

  applyViewerConfig();

  resizeObserver = new ResizeObserver(() => { if (!viewer || isDestroyed) return; viewer.resize(); });
  resizeObserver.observe(container);

  // CameraControl — official Aholo controls
  const canvas = container.querySelector('canvas');
  if (canvas) {
    camCtrl = new CameraControl(camera, canvas, {
      orbitCenter: { x: 0, y: 0, z: 0 },
      lookSpeed: 0.008,
      moveSpeed: 3.0,
      wheelSpeed: 0.04,
      panSpeed: 0.03,
    });
    console.log('[GS] CameraControl initialized');
  }

  createHintBox();

  stats('EGS 引擎就绪');
  return true;
}

// ── Render Loop ──
function startRenderLoop() {
  if (animId) return;
  function loop() {
    if (isDestroyed) return;
    animId = requestAnimationFrame(loop);
    if (!viewer) return;
    camCtrl?.update();
    viewer.resize();
    viewer.render();
  }
  animId = requestAnimationFrame(loop);
  console.log('[GS] render loop started');
}

function stopRenderLoop() {
  if (animId) { cancelAnimationFrame(animId); animId = null; }
}

// ── Load ──
async function loadGaussianSplat(url) {
  if (isLoading) return;
  if (!viewer) { const ok = await initViewer(); if (!ok) return; startRenderLoop(); bindToolbar(); }

  isLoading = true;
  const t0 = performance.now();
  try {
    console.log('[GS] loading:', url);
    stats('下载解析中...');
    const ext = url.split('.').pop().toLowerCase().split('?')[0];
    const fileType = ext === 'splat' ? SplatFileType.Splat : SplatFileType.PLY;

    const data = await parseSplatData(fileType, url);
    console.log('[GS] parsed, counts:', data.counts, 'shDegree:', data.shDegree);

    stats('构建 splat...');
    const splat = await createSplat(data);
    console.log('[GS] splat created:', splat.constructor.name, 'counts:', splat.counts);

    if (currentSplat) { scene.remove(currentSplat); currentSplat.destroy?.(); currentSplat = null; }
    scene.add(splat);
    scene.notifySceneChange?.();
    currentSplat = splat;

    viewer.resize();
    viewer.render();

    const tm = (performance.now() - t0).toFixed(0);
    const count = (data.counts || splat.counts || '?').toLocaleString?.() || data.counts || splat.counts || '?';
    stats(count + ' 高斯 | ' + tm + 'ms');
    console.log('[GS] load done in', tm, 'ms');
  } catch (e) {
    console.error('[GS] load failed:', e);
    stats('失败: ' + e.message);
  } finally { isLoading = false; }
}

// ── Toolbar ──
function bindToolbar() {
  const demoBtn = $('gsLoadDemoBtn');
  if (demoBtn) demoBtn.onclick = () => loadGaussianSplat('/api/gaussian-splats/demo');

  const urlBtn = $('gsLoadUrlBtn'), urlInput = $('gsUrlInput');
  if (urlBtn && urlInput) {
    urlBtn.onclick = () => {
      if (urlInput.style.display === 'none') { urlInput.style.display = 'inline-block'; urlInput.focus(); }
      else { if (urlInput.value.trim()) loadGaussianSplat(urlInput.value.trim()); urlInput.style.display = 'none'; }
    };
    urlInput.onkeydown = (e) => {
      if (e.key === 'Enter' && urlInput.value.trim()) { loadGaussianSplat(urlInput.value.trim()); urlInput.style.display = 'none'; }
      if (e.key === 'Escape') urlInput.style.display = 'none';
    };
  }

  container?.addEventListener('dragover', (e) => { e.preventDefault(); container.classList.add('drag-over'); });
  container?.addEventListener('dragleave', () => { container.classList.remove('drag-over'); });
  container?.addEventListener('drop', (e) => {
    e.preventDefault(); container.classList.remove('drag-over');
    const f = e.dataTransfer?.files?.[0];
    if (f) loadGaussianSplat(URL.createObjectURL(f));
  });
}

// ── Cleanup ──
function destroy() {
  isDestroyed = true;
  stopRenderLoop();
  camCtrl?.dispose(); camCtrl = null;
  resizeObserver?.disconnect(); resizeObserver = null;
  if (currentSplat) { currentSplat.destroy?.(); currentSplat = null; }
  viewer?.destroy?.(); viewer = null;
  scene = null; camera = null; container = null;
  console.log('[GS] destroyed');
}

// ── Public API ──
window.initGaussianViewer = async function () {
  console.log('[GS] initGaussianViewer called');
  const ok = await initViewer();
  if (!ok) return;
  startRenderLoop();
  bindToolbar();
};

window.loadGSFromUrl = function (url) {
  if (!url) return;
  initViewer().then((ok) => {
    if (!ok) return;
    startRenderLoop();
    bindToolbar();
    loadGaussianSplat(url);
  });
};

window.destroyGaussianViewer = destroy;

