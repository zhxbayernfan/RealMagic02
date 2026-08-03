'use strict';

const fs = require('fs');
const path = require('path');

const WEB_DIR = path.join(__dirname, '../../web');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css':  'text/css; charset=utf-8',
  '.js':   'application/javascript; charset=utf-8',
  '.svg':  'image/svg+xml',
  '.png':  'image/png',
  '.ico':  'image/x-icon',
};

function serveFile(res, filePath, cacheControl) {
  if (!fs.existsSync(filePath)) {
    res.writeHead(404); res.end('Not found'); return;
  }
  const ext = path.extname(filePath).toLowerCase();
  const mime = MIME[ext] || 'application/octet-stream';
  res.writeHead(200, {
    'Content-Type': mime,
    'Cache-Control': cacheControl || 'no-store',
  });
  fs.createReadStream(filePath).pipe(res);
}

function registerWebRoutes(router) {
  router.get('/', (req, res) => {
    serveFile(res, path.join(WEB_DIR, 'index.html'), 'no-store, no-cache, must-revalidate');
  });

  router.get('/index.html', (req, res) => {
    serveFile(res, path.join(WEB_DIR, 'index.html'), 'no-store');
  });

  router.get('/favicon.svg', (req, res) => {
    serveFile(res, path.join(WEB_DIR, 'favicon.svg'), 'public, max-age=3600');
  });

  router.get('/favicon.ico', (req, res) => {
    serveFile(res, path.join(WEB_DIR, 'favicon.ico'), 'public, max-age=3600');
  });

  router.get(
    (p) => p.startsWith('/web/assets/') || p.startsWith('/assets/'),
    (req, res, ctx) => {
      const rel = ctx.url.pathname.replace(/^\/(?:web\/)?assets\//, '');
      const filePath = path.join(WEB_DIR, 'assets', rel);
      serveFile(res, filePath, 'public, max-age=60');
    }
  );

  // 通用静态文件服务：兜底匹配所有未显式路由的路径
  // 用于 index-tab.js, New_Stmem.html, Sentrix Monitor 等新增文件
  router.get(
    (p) => true,
    (req, res, ctx) => {
      const urlPath = ctx.url.pathname;
      // 排除已由其他路由处理的路径和 API 路径
      if (urlPath === '/' || urlPath === '/index.html' ||
          urlPath === '/favicon.svg' || urlPath === '/favicon.ico' ||
          urlPath.startsWith('/assets/') || urlPath.startsWith('/web/assets/') ||
          urlPath.startsWith('/api/') || urlPath.startsWith('/batch/') ||
          urlPath.startsWith('/vlm/')) {
        res.writeHead(404); res.end('Not found'); return;
      }
      const decoded = decodeURIComponent(urlPath);
      const filePath = path.join(WEB_DIR, decoded);
      // 安全检查：防止目录穿越
      if (!filePath.startsWith(WEB_DIR)) {
        res.writeHead(403); res.end('Forbidden'); return;
      }
      serveFile(res, filePath, 'public, max-age=60');
    },
    99  // 最低优先级，最后匹配
  );
}

module.exports = { registerWebRoutes };
