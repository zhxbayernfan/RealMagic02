// camera-control.js — Aholo CameraControl port
// Free-flight (WASD Q/E R/F) + rotate (left drag) + pan (right/middle drag)
// + orbit (alt+left drag) + dolly (wheel) + speed (shift/ctrl/caps)

const EPS = 1e-6, MAX_P = Math.PI/2 - 1e-3, DEF_UP = {x:0,y:1,z:0};
const KEYS = new Set(['KeyW','KeyA','KeyS','KeyD','KeyQ','KeyE','KeyR','KeyF',
  'ShiftLeft','ShiftRight','ControlLeft','ControlRight','AltLeft','AltRight','CapsLock']);

const V = {
  len: a => Math.hypot(a.x,a.y,a.z),
  dot: (a,b) => a.x*b.x + a.y*b.y + a.z*b.z,
  sub: (a,b) => ({x:a.x-b.x,y:a.y-b.y,z:a.z-b.z}),
  add: (a,b) => ({x:a.x+b.x,y:a.y+b.y,z:a.z+b.z}),
  mul: (v,s) => ({x:v.x*s,y:v.y*s,z:v.z*s}),
  cp: v => ({x:v.x,y:v.y,z:v.z}),
  cross: (a,b) => ({x:a.y*b.z-a.z*b.y,y:a.z*b.x-a.x*b.z,z:a.x*b.y-a.y*b.x}),
  norm: v => { const l=Math.hypot(v.x,v.y,v.z); return l<EPS?undefined:{x:v.x/l,y:v.y/l,z:v.z/l}; },
  proj: (v,n) => V.sub(v, V.mul(n, V.dot(v,n))),
  perp: a => { const h=Math.abs(a.y)<0.9?{x:0,y:1,z:0}:{x:1,y:0,z:0}; return V.norm(V.cross(a,h))??{x:1,y:0,z:0}; },
  rot: (v,ax,ang) => { const c=Math.cos(ang),s=Math.sin(ang),cr=V.cross(ax,v),as=V.dot(ax,v)*(1-c);
    return V.add(V.add(V.mul(v,c),V.mul(cr,s)),V.mul(ax,as)); },
  angA: (f,t,ax) => { const pf=V.norm(V.proj(f,ax)),pt=V.norm(V.proj(t,ax));
    if(!pf||!pt)return 0; return Math.atan2(V.dot(V.cross(pf,pt),ax),V.dot(pf,pt)); },
  clamp: (v,l,h) => Math.min(Math.max(v,l),h),
  dist: (x1,y1,x2,y2) => Math.hypot(x2-x1,y2-y1),
};
function nK(ks,c) { return ks.has(c)?1:0; }
function mat4(e) { return {_elements: Float32Array.from(e)}; }

function getUp(c) { return V.norm(c.up??DEF_UP)??DEF_UP; }
function getMat(c) {
  if (c.quaternion) {
    const {x,y,z,w}=c.quaternion,x2=x+x,y2=y+y,z2=z+z;
    const xx=x*x2,xy=x*y2,xz=x*z2,yy=y*y2,yz=y*z2,zz=z*z2,wx=w*x2,wy=w*y2,wz=w*z2;
    return mat4([1-(yy+zz),xy+wz,xz-wy,0, xy-wz,1-(xx+zz),yz+wx,0, xz+wy,yz-wx,1-(xx+yy),0, 0,0,0,1]);
  }
  const r=c.rotation,a=Math.cos(r.x),b=Math.sin(r.x),cc=Math.cos(r.y),d=Math.sin(r.y),e=Math.cos(r.z),f=Math.sin(r.z);
  return mat4([cc*e,a*f+b*e*d,b*f-a*e*d,0, -cc*f,a*e-b*f*d,b*e+a*f*d,0, d,-b*cc,a*cc,0, 0,0,0,1]);
}
function getB(c) {
  const u=getUp(c),m=getMat(c),te=m._elements;
  const f=V.norm({x:-te[8],y:-te[9],z:-te[10]})??{x:0,y:0,z:-1};
  const r=V.norm({x:te[0],y:te[1],z:te[2]})??V.perp(u);
  const vu=V.norm({x:te[4],y:te[5],z:te[6]})??V.norm(V.cross(r,f))??u;
  return {forward:f,right:r,up:u,viewUp:vu};
}
function vuR(f,u,roll) {
  const r=V.norm(V.cross(f,u))??V.perp(f);
  const vu=V.norm(V.cross(r,f))??u;
  return V.norm(V.rot(vu,f,roll))??vu;
}
function fwR(f,u,vu) { const l=vuR(f,u,0); return V.angA(l,vu,f); }
function setB(c,f,u) {
  const r=V.norm(V.cross(f,u))??V.perp(u), vu=V.norm(V.cross(r,f))??u, bk=V.mul(f,-1);
  const m=mat4([r.x,r.y,r.z,0, vu.x,vu.y,vu.z,0, bk.x,bk.y,bk.z,0, 0,0,0,1]);
  const rot=c.rotation;
  if(typeof rot.setFromRotationMatrix==='function') { rot.setFromRotationMatrix(m,rot.order); return; }
  const te=m._elements,m11=te[0],m12=te[4],m13=te[8],m23=te[9],m33=te[10];
  const ey=Math.asin(V.clamp(m13,-1,1));
  let ex,ez;
  if(Math.abs(m13)<0.99999){ex=Math.atan2(-m23,m33);ez=Math.atan2(-m12,m11);}
  else{ex=Math.atan2(te[6],te[5]);ez=0;}
  if(typeof rot.set==='function')rot.set(ex,ey,ez,rot.order);else{rot.x=ex;rot.y=ey;rot.z=ez;}
}
function rotC(c,yd,pd) {
  const b=getB(c),roll=fwR(b.forward,b.up,b.viewUp);
  const cp=Math.asin(V.clamp(V.dot(b.forward,b.up),-1,1));
  const np=V.clamp(cp+pd,-MAX_P,MAX_P);
  const hf=V.norm(V.proj(b.forward,b.up))??V.perp(b.up);
  const yf=V.rot(hf,b.up,yd);
  const f=V.norm(V.add(V.mul(yf,Math.cos(np)),V.mul(b.up,Math.sin(np))));
  if(!f)return; setB(c,f,vuR(f,b.up,roll));
}
function orbC(c,ctr,yd,pd,md) {
  const b=getB(c),roll=fwR(b.forward,b.up,b.viewUp),sm=Math.max(EPS,md);
  let d=V.len(V.sub(c.position,ctr));
  let f=V.norm(V.sub(ctr,c.position))??V.norm(b.forward)??{x:0,y:0,z:-1};
  if(d<sm){d=sm;f=V.norm(b.forward)??f;}
  const cp=Math.asin(V.clamp(V.dot(f,b.up),-1,1));
  const np=V.clamp(cp+pd,-MAX_P,MAX_P);
  const hf=V.norm(V.proj(f,b.up))??V.perp(b.up);
  const yf=V.rot(hf,b.up,yd);
  const nf=V.norm(V.add(V.mul(yf,Math.cos(np)),V.mul(b.up,Math.sin(np))));
  if(!nf)return false;
  c.position.x=ctr.x-nf.x*d; c.position.y=ctr.y-nf.y*d; c.position.z=ctr.z-nf.z*d;
  setB(c,nf,vuR(nf,b.up,roll)); return true;
}
function rollC(c,d) {
  const b=getB(c),vu=V.norm(V.rot(b.viewUp,b.forward,-d));
  if(!vu)return; setB(c,b.forward,vu);
}

export class CameraControl {
  constructor(camera, element, options={}) {
    this.c=camera; this.e=element;
    this.ptrs=new Map(); this.ks=new Set();
    this.oc=V.cp(options.orbitCenter??{x:0,y:0,z:0});
    this._lt=0; this._wd=0; this._cl=false; this._ak=false;
    this._mv=false; this._rt=false; this._pn=false; this._ob=false; this._dp=false;
    const D={enabled:true,keyboardEnabled:true,pointerEnabled:true,orbitEnabled:true,
      useOrbit:false,orbitMinDistance:0.01,moveSpeed:0.4,lookSpeed:0.004,wheelSpeed:0.006,
      panSpeed:0.006,rollSpeed:1,shiftMultiplier:10,ctrlMultiplier:2,capsMultiplier:20};
    const o={...D,...options};
    for(const[k,v]of Object.entries(o)){if(k!=='orbitCenter')this[k]=v;}
    this.omd=Math.max(EPS,o.orbitMinDistance);
    this._ti=element.getAttribute('tabindex'); this._ta=element.style.touchAction;
    if(this._ti===null)element.tabIndex=0;
    element.style.touchAction='none';
    element.addEventListener('pointerdown',this._pd);
    element.addEventListener('pointermove',this._pm);
    element.addEventListener('pointerup',this._pu);
    element.addEventListener('pointercancel',this._pu);
    element.addEventListener('contextmenu',e=>e.preventDefault());
    element.addEventListener('wheel',this._pw,{passive:false});
    element.addEventListener('keydown',this._kd);
    element.addEventListener('keyup',this._ku);
    window.addEventListener('keyup',this._ku);
    window.addEventListener('blur',this._bl);
  }
  setOpts(o) {
    if(o.orbitCenter){this.oc.x=o.orbitCenter.x;this.oc.y=o.orbitCenter.y;this.oc.z=o.orbitCenter.z;}
    for(const[k,v]of Object.entries(o)){if(v!==undefined&&k!=='orbitCenter'){if(k==='orbitMinDistance')this.omd=Math.max(EPS,v);else this[k]=v;}}
  }
  stop() {
    for(const[id]of this.ptrs){if(this.e.hasPointerCapture(id))this.e.releasePointerCapture(id);}
    this.ks.clear();this.ptrs.clear();this._wd=0;this._ak=false;
    this._mv=false;this._rt=false;this._pn=false;this._ob=false;
  }
  update(dt) {
    if(this._dp||!this.enabled)return false;
    const n=performance.now(),d=dt??Math.min((n-(this._lt||n))/1000,0.1);this._lt=n;
    const pc=this.pointerEnabled?this._upP():false;
    const kc=this.keyboardEnabled?this._upK(d):false;
    const wc=this.pointerEnabled?this._upW():false;
    return pc||kc||wc;
  }
  dispose() {
    if(this._dp)return;this.stop();this._dp=true;
    this.e.removeEventListener('pointerdown',this._pd);
    this.e.removeEventListener('pointermove',this._pm);
    this.e.removeEventListener('pointerup',this._pu);
    this.e.removeEventListener('pointercancel',this._pu);
    this.e.removeEventListener('wheel',this._pw);
    this.e.removeEventListener('keydown',this._kd);
    this.e.removeEventListener('keyup',this._ku);
    window.removeEventListener('keyup',this._ku);
    window.removeEventListener('blur',this._bl);
    this.e.style.touchAction=this._ta;
    if(this._ti===null)this.e.removeAttribute('tabindex');
    else this.e.tabIndex=this._ti;
  }
  _pd=e=>{
    if(!this.enabled||!this.pointerEnabled)return;
    this.e.focus({preventScroll:true});this._ak=e.altKey;
    this.ptrs.set(e.pointerId,{pid:e.pointerId,pt:e.pointerType,b:e.button,
      m:this._md(e.pointerType,e.button),lx:e.clientX,ly:e.clientY,x:e.clientX,y:e.clientY});
    this.e.setPointerCapture(e.pointerId);e.preventDefault();
  };
  _pm=e=>{const p=this.ptrs.get(e.pointerId);if(!p)return;this._ak=e.altKey;p.x=e.clientX;p.y=e.clientY;e.preventDefault();};
  _pu=e=>{
    if(this.ptrs.has(e.pointerId)){
      this.ptrs.delete(e.pointerId);
      if(this.e.hasPointerCapture(e.pointerId))this.e.releasePointerCapture(e.pointerId);
      if(this.ptrs.size===0){this._rt=false;this._pn=false;this._ob=false;}
    }
  };
  _pw=e=>{if(this.enabled&&this.pointerEnabled){this._wd+=e.deltaY;e.preventDefault();}};
  _kd=e=>{if(!this.enabled||!this.keyboardEnabled||!KEYS.has(e.code))return;
    this.ks.add(e.code);this._cl=e.getModifierState('CapsLock');
    this._ak=e.altKey||e.code==='AltLeft'||e.code==='AltRight';e.preventDefault();};
  _ku=e=>{this.ks.delete(e.code);this._cl=e.getModifierState('CapsLock');this._ak=e.altKey;};
  _bl=()=>this.stop();
  _md(t,b){if(t==='mouse'){if(b===1||b===2)return'pan';if(b===0&&this.orbitEnabled&&(this.useOrbit||this._om()))return'orbit';}return'rotate';}
  _om(){return this._ak||this.ks.has('AltLeft')||this.ks.has('AltRight');}
  _sp(){let m=1;if(this.ks.has('ShiftLeft')||this.ks.has('ShiftRight'))m*=this.shiftMultiplier;
    if(this.ks.has('ControlLeft')||this.ks.has('ControlRight'))m*=this.ctrlMultiplier;
    if(this._cl||this.ks.has('CapsLock'))m*=this.capsMultiplier;return m;}
  _upP(){
    const ps=Array.from(this.ptrs.values());this._rt=false;this._pn=false;this._ob=false;
    if(ps.length===0)return false;let u=false;
    if(ps.length>=2){
      const[a,b]=ps;
      const lmx=(a.lx+b.lx)*0.5,lmy=(a.ly+b.ly)*0.5,mx=(a.x+b.x)*0.5,my=(a.y+b.y)*0.5;
      const ld=V.dist(a.lx,a.ly,b.lx,b.ly),cd=V.dist(a.x,a.y,b.x,b.y);
      u=this._panP(mx-lmx,my-lmy)||u;
      u=this._mvV((cd-ld)*this.wheelSpeed)||u;
      this._pn=Math.abs(mx-lmx)+Math.abs(my-lmy)+Math.abs(cd-ld)>0.001;
    }else{
      const p=ps[0],m=this._md(p.pt,p.b);
      if(p.m!==m){p.m=m;p.lx=p.x;p.ly=p.y;}
      const dx=p.x-p.lx,dy=p.y-p.ly;
      if(m==='pan'){u=this._panP(dx,dy);this._pn=u;}
      else if(m==='orbit'){u=this._orbP(dx,dy);this._ob=true;}
      else{u=this._rotP(dx,dy);this._rt=u;}
    }
    for(const p of ps){p.lx=p.x;p.ly=p.y;}return u;
  }
  _upK(dt){
    const fw=nK(this.ks,'KeyW')-nK(this.ks,'KeyS');
    const st=nK(this.ks,'KeyD')-nK(this.ks,'KeyA');
    const vt=nK(this.ks,'KeyQ')-nK(this.ks,'KeyE');
    const rl=nK(this.ks,'KeyR')-nK(this.ks,'KeyF');
    const mul=this._sp();let u=false;
    const ml=Math.hypot(fw,st,vt);this._mv=ml>0;
    if(ml>0){
      const s=(this.moveSpeed*mul*dt)/Math.max(1,ml);
      const b=getB(this.c),p=this.c.position;
      p.x+=(b.forward.x*fw+b.right.x*st+b.up.x*vt)*s;
      p.y+=(b.forward.y*fw+b.right.y*st+b.up.y*vt)*s;
      p.z+=(b.forward.z*fw+b.right.z*st+b.up.z*vt)*s;u=true;
    }
    if(rl!==0){rollC(this.c,rl*this.rollSpeed*dt);this._rt=true;u=true;}
    else if(this.ptrs.size===0)this._rt=false;
    return u;
  }
  _upW(){if(Math.abs(this._wd)<0.001)return false;const d=-this._wd*this.wheelSpeed;this._wd=0;return this._mvV(d);}
  _rotP(dx,dy){if(Math.abs(dx)+Math.abs(dy)<0.001)return false;rotC(this.c,-dx*this.lookSpeed,-dy*this.lookSpeed);return true;}
  _orbP(dx,dy){if(Math.abs(dx)+Math.abs(dy)<0.001)return false;return orbC(this.c,this.oc,-dx*this.lookSpeed,-dy*this.lookSpeed,this.omd);}
  _panP(dx,dy){
    if(Math.abs(dx)+Math.abs(dy)<0.001)return false;
    const b=getB(this.c),p=this.c.position,x=dx*this.panSpeed,y=-dy*this.panSpeed;
    p.x+=b.right.x*x+b.viewUp.x*y;p.y+=b.right.y*x+b.viewUp.y*y;p.z+=b.right.z*x+b.viewUp.z*y;return true;
  }
  _mvV(d){if(Math.abs(d)<0.001)return false;const f=getB(this.c).forward,p=this.c.position;p.x+=f.x*d;p.y+=f.y*d;p.z+=f.z*d;return true;}
}
