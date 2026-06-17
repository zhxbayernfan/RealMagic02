(function(){
  var tabs=[
    {label:'看见',sel:'.grid'},
    {label:'理解',sel:'[data-screen-label="02 理解"]'},
    {label:'规划',sel:'[data-screen-label="03 规划"]'},
    {label:'记忆',sel:'[data-screen-label="04 记忆"]'},
    {label:'执行',sel:'[data-screen-label="05 执行"]'},
    {label:'数据汇总',url:'/New_Stmem.html'}
  ];
  var stage=document.getElementById('stage');
  if(!stage) return;
  var frame=stage.querySelector('.frame');
  if(!frame) return;
  var grid=frame.querySelector('.grid');
  var tabBar=document.createElement('div');
  tabBar.style.cssText='display:flex;gap:0;padding:0 20px;margin-bottom:10px';
  tabs.forEach(function(t,i){
    var b=document.createElement('button');
    b.textContent=t.label;
    b.style.cssText='padding:8px 16px;border:1px solid #E0E2DD;border-bottom:none;border-radius:8px 8px 0 0;background:#F4F5F2;color:#5E6259;font:600 12px sans-serif;cursor:pointer;margin-right:-1px';
    b.onclick=function(){
      tabBar.querySelectorAll('button').forEach(function(x){x.style.background='#F4F5F2';x.style.color='#5E6259'});
      b.style.background='#FFFFFF';b.style.color='#151613';
      var ifr=frame.querySelector('iframe');
      if(ifr) ifr.remove();
      if(t.url){
        if(grid) grid.style.display='none';
        var n=document.createElement('iframe');
        n.src=t.url;n.style.cssText='flex:1;min-height:0;border:none;width:100%;height:calc(100% - 80px);border-radius:12px';
        frame.appendChild(n);
      }else{
        if(grid) grid.style.display='';
      }
    };
    if(i===0){b.style.background='#FFFFFF';b.style.color='#151613'}
    tabBar.appendChild(b);
  });
  frame.insertBefore(tabBar,frame.firstChild);
})();
