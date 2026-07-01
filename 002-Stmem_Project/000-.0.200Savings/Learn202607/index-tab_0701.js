(function(){
  // 等待页面渲染完成后注入我的记忆 tab
  var check = setInterval(function(){
    var navTabs = document.querySelector('.nav-tabs');
    var tabContent = document.querySelector('#tab-embodied');
    if(!navTabs || !tabContent) return;
    clearInterval(check);

    // 添加 nav tab
    var btn = document.createElement('button');
    btn.className = 'nav-tab';
    btn.textContent = '我的记忆';
    btn.setAttribute('onclick', "switchTab('summary')");
    btn.addEventListener('click', function(){ switchTab('summary'); });
    navTabs.appendChild(btn);

    // 添加内容区（iframe 懒加载，首次点击才加载 19MB HTML）
    var summaryDiv = document.createElement('div');
    summaryDiv.id = 'tab-summary';
    summaryDiv.className = 'tab-page';
    summaryDiv.style.cssText = 'display:none;width:100%;height:100%';
    var iframe = document.createElement('iframe');
    iframe.style.cssText = 'display:block;border:0;width:100%;flex:1;min-height:0';
    summaryDiv.appendChild(iframe);
    tabContent.parentNode.appendChild(summaryDiv);

    // 拦截 switchTab 支持 summary + 懒加载
    var origSwitchTab = window.switchTab;
    window.switchTab = function(tabId){
      // 先调用原始 switchTab，让 active class 正常工作
      if (origSwitchTab) origSwitchTab(tabId);
      // 再处理 summary 这个动态 tab
      var allTabs = document.querySelectorAll('[id^="tab-"]');
      var summaryTab = document.getElementById('tab-summary');
      if (tabId === 'summary') {
        allTabs.forEach(function(t){
          if (t !== summaryTab) { t.style.display = 'none'; t.classList.remove('active'); }
        });
        if (summaryTab) {
          summaryTab.style.display = '';
          summaryTab.classList.add('active');
          // 懒加载 iframe
          var f = summaryTab.querySelector('iframe');
          if (f && !f.src) f.src = '/New_Stmem.html';
        }
      } else if (summaryTab) {
        // 切离 summary 时，清掉自己并恢复其他 tab 的 display（让 CSS .active 接管）
        summaryTab.style.display = 'none';
        summaryTab.classList.remove('active');
        allTabs.forEach(function(t){
          if (t !== summaryTab) t.style.display = '';
        });
      }
    };
  }, 500);
})();
