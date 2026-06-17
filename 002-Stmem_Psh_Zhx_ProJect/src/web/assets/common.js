// common.js - shared navigation
function switchTab(name) {
      document.querySelectorAll('.tab-page').forEach(function(p) { p.classList.remove('active'); });
      document.querySelectorAll('.nav-tab').forEach(function(t) { t.classList.remove('active'); });
      var page = document.getElementById('tab-' + name);
      if (page) page.classList.add('active');
      var tabs = document.querySelectorAll('.nav-tab');
      var map = { time: 0, space: 1, fusion: 2, api: 3, about: 4, gsviewer: 5, embodied: 6 };
      if (tabs[map[name]]) tabs[map[name]].classList.add('active');
      if (name === 'time') {
        if (typeof syncHeights === 'function') {
          setTimeout(syncHeights, 0);
          setTimeout(syncHeights, 120);
        }
      }
      if (name === 'space') { setTimeout(function() { if (typeof initPLYViewer === 'function') initPLYViewer(); if (typeof initSceneGraph === 'function') initSceneGraph(); if (typeof onSpatialTabShow === 'function') onSpatialTabShow(); }, 100); }
      if (name === 'gsviewer') { setTimeout(function() { if (typeof initGaussianViewer === 'function') initGaussianViewer(); }, 100); }
      if (name === 'embodied') {
        // 首次进入才加载 standalone，避免主页打开就吃 20MB
        var f = document.getElementById('embodiedFrame');
        if (f && !f.src) f.src = '/Sentrix%20Monitor%20(standalone).html';
      }

    }