'use strict';
const activeView=location.pathname==='/source'?'source':location.pathname==='/track'?'track':'overview';document.body.dataset.view=activeView;
for(const a of document.querySelectorAll('.view-nav a'))a.classList.toggle('selected',a.dataset.view===activeView);

const $=id=>document.getElementById(id),names=['下跌 ↓','持平 →','上涨 ↑'];let state=null,lastBars='',lastDirectory=null,polling=false,finishedFit=false;
$('view-title').textContent=activeView==='source'?'让行情按真实节奏到达':activeView==='track'?'一份窗口，一条完整预测链':'实时订单，硬件预测';
const chart=activeView==='overview'?LightweightCharts.createChart($('chart'),{autoSize:true,layout:{background:{type:'solid',color:'#111e30'},textColor:'#93a9c4'},grid:{vertLines:{color:'#1e2c40'},horzLines:{color:'#1e2c40'}},rightPriceScale:{autoScale:true,scaleMargins:{top:.18,bottom:.18}},timeScale:{timeVisible:true,secondsVisible:true,barSpacing:6,minBarSpacing:2,maxBarSpacing:12,rightOffset:6},localization:{timeFormatter:t=>new Date(t*1000).toISOString().slice(11,23),priceFormatter:x=>x.toFixed(4)}}):null;
const candles=chart?chart.addSeries(LightweightCharts.CandlestickSeries,{upColor:'#ff839a',downColor:'#52d9b5',wickUpColor:'#ff839a',wickDownColor:'#52d9b5',borderVisible:false,priceFormat:{type:'price',precision:4,minMove:.00005},autoscaleInfoProvider:original=>{const info=original();if(!info)return null;const r=info.priceRange,mid=(r.minValue+r.maxValue)/2,span=Math.max(.10,Math.abs(mid)*.0002);if(r.maxValue-r.minValue<span)info.priceRange={minValue:mid-span/2,maxValue:mid+span/2};return info;}}):{setData(){}};
function num(v,n=3){return Number.isFinite(v)?v.toFixed(n):'—'}
function pct(v){return Number.isFinite(v)?(v*100).toFixed(2)+'%':'—'}
function fit(){if(!chart)return;const n=state?.telemetry?.bars?.length||0;if(!n)return;chart.priceScale('right').applyOptions({autoScale:true});chart.timeScale().setVisibleLogicalRange({from:Math.min(-5,n-100),to:n+5});}
$('chart-fit').onclick=fit;
// Size only the overview viewport; the market clock and inference stay independent.
function fitOverviewLayout(){
 if(activeView!=='overview')return;
 const layout=document.querySelector('.layout'),desktop=matchMedia('(min-width: 1000px) and (min-height: 560px)').matches;
 if(!desktop){layout.style.removeProperty('--overview-height');return;}
 layout.style.setProperty('--overview-height',Math.max(240,Math.floor(innerHeight-layout.getBoundingClientRect().top-12))+'px');
}
if(activeView==='overview'){
 let resizeFrame=0;
 const scheduleLayout=()=>{cancelAnimationFrame(resizeFrame);resizeFrame=requestAnimationFrame(fitOverviewLayout);};
 addEventListener('resize',scheduleLayout);
 const overviewObserver=new ResizeObserver(scheduleLayout);
 for(const selector of ['header','.view-nav','.heading','#notice','.metrics'])overviewObserver.observe(document.querySelector(selector));
 scheduleLayout();
}
function cell(row,value,cls){const c=document.createElement('td');c.textContent=value;if(cls)c.className=cls;row.append(c);}
function render(s){state=s;const t=s.telemetry||{},pred=t.recent_predictions||[],p=pred.at(-1),b=t.board||{},done=['finished','review'].includes(s.phase);
 $('notice').textContent=s.error||s.message;$('notice').title=s.error||s.message;$('notice').classList.toggle('error',Boolean(s.error));
 $('phase').textContent=({idle:'等待启动',running:'真实板卡运行中',finished:'本轮完成 · 已回收',failed:'失败 · 现场已保存',review:'真实记录复盘'})[s.phase]||s.phase;
 $('start').disabled=s.busy||s.review||s.closing;for(const id of ['slots','duration','dispatch-speed','age-limit'])$(id).disabled=s.busy||s.review;
 $('export').disabled=!done;$('close').textContent=s.busy?'本轮结束后退出':'退出演示服务';$('round').textContent=s.review?'保存记录复盘':'第 '+(s.round||0)+' 轮';
 const dispatchSpeed=s.source?.speed||s.dispatch_speed||1;
 $('run-description').textContent='2025-12-15 · AAPL 原始 MBO · '+dispatchSpeed+'倍实际派发 · 历史市场回放';
 $('source-schedule').textContent='2025-12-15 · AAPL · 原始 MBO事件 · '+dispatchSpeed+'倍实际派发';
 if(s.busy||s.review)$('dispatch-speed').value=String(dispatchSpeed);updateDispatchHint();
 if(s.directory!==lastDirectory){candles.setData([]);lastBars='';finishedFit=false;lastDirectory=s.directory;}
 $('age').textContent=p&&p.age_ms!==null?num(p.age_ms)+' ms':'—';$('npu').textContent=t.NPU?num(t.NPU.mean_us,2)+' μs':'—';$('npu-title').textContent=done?'NPU启动等待 · 全窗均值':'NPU启动等待 · 采样均值';$('npu-note').textContent=done?'完整逐窗原生计时均值':'按窗口身份匹配的最近原生计时样本';
 $('coverage').textContent=pct(t.five_ms_coverage);$('coverage-count').textContent=(t.five_ms_completed||0).toLocaleString()+' / '+(t.opportunities||0).toLocaleString()+' 次机会在5ms内完成';
 $('pipeline-label').textContent=s.pipeline?.name||'整数预处理 · 整窗FIFO · 共享输入';
 $('h50').textContent=pct(t.H50_coverage);$('h50-count').textContent=(t.H50_timely||0)+' / '+(t.opportunities||0)+' 次机会及时';
 $('expired').textContent=(b.expired??t.expired??0).toLocaleString();$('raw').textContent=(t.raw_events||0).toLocaleString()+' / '+(t.raw_target||0).toLocaleString();
 if(p){$('identity').textContent='机会 #'+p.opportunity;$('direction').textContent=names[p.class_index];$('direction').className=['down','flat','up'][p.class_index];$('scores').replaceChildren(...p.scores.map((v,i)=>{const d=document.createElement('div'),a=document.createElement('span'),bar=document.createElement('i'),c=document.createElement('b');a.textContent=names[i].slice(0,2);bar.style.width=(v*140)+'px';c.textContent=pct(v);d.append(a,bar,c);return d;}));}
 else{$('identity').textContent='等待首窗';$('direction').textContent='—';$('scores').replaceChildren();}
 if(t.latest){$('mid').textContent='$'+num(t.latest.mid,4);const bk=t.latest.book;$('book').replaceChildren(...Array.from({length:10},(_,i)=>{const tr=document.createElement('tr');cell(tr,i+1);cell(tr,num(bk[i*4]/10000,4)+' / '+bk[i*4+1].toLocaleString(),'up');cell(tr,num(bk[i*4+2]/10000,4)+' / '+bk[i*4+3].toLocaleString(),'down');return tr;}));}
 else{$('mid').textContent='—';$('book').replaceChildren();}
 const bars=t.bars||[],key=JSON.stringify(bars);if(key!==lastBars){const first=lastBars==='';candles.setData(bars);lastBars=key;if(first||$('chart-follow').checked)fit();}if(done&&!finishedFit){fit();finishedFit=true;}
 const e=t.evaluation||{};$('hit-rate').textContent=pct(e.hit_rate);$('hit-count').textContent=(e.hits||0)+' / '+(e.resolved||0)+' 次已兑现预测';
 $('timely-return-rate').textContent=pct(t.predictions?t.H50_timely/t.predictions:null);$('timely-return-count').textContent=(t.H50_timely||0).toLocaleString()+' / '+(t.predictions||0).toLocaleString()+' 份已完成预测';
 $('settled-count').textContent=(e.resolved||0)+' / '+(e.pending||0);
 $('rows').replaceChildren(...pred.slice().reverse().map(v=>{const tr=document.createElement('tr');cell(tr,'#'+v.opportunity);cell(tr,names[v.class_index],v.class_index===2?'up':v.class_index===0?'down':'');cell(tr,v.actual_class===null?(v.settlement==='pending'?'待未来盘口':'尾部未决'):names[v.actual_class]);cell(tr,v.hit===null?'—':v.hit?'命中':'未命中',v.hit===null?'':v.hit?'hit':'miss');cell(tr,v.mid_change===null?'—':num(v.mid_change,5));cell(tr,v.H50_timely===null?'待时间参考':v.H50_timely?'及时':'超期',v.H50_timely?'hit':'miss');cell(tr,num(v.age_ms));cell(tr,v.npu_us===null?'待收集':num(v.npu_us,2));return tr;}));
 const summary=[];if(t.latency)summary.push('平均 '+num(t.latency.mean_us/1000)+' ms','P99 '+num(t.latency.p99_us/1000)+' ms','最大 '+num(t.latency.max_us/1000)+' ms');summary.push('过期取消 '+(t.expired||0));if(t.unaccepted!==null&&t.unaccepted!==undefined)summary.push('未接收 '+t.unaccepted);$('summary').replaceChildren(...summary.map(text=>{const x=document.createElement('span');x.textContent=text;return x;}));
 $('path').textContent=s.directory?'日志目录：'+s.directory:'';renderEvidence(s);
}
async function action(name,payload={}){const r=await fetch('/api/'+name,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const s=await r.json();if(!r.ok)throw Error(s.error||'操作失败');render(s);}
function updateDispatchHint(){const minutes=Number($('duration').value),speed=Number($('dispatch-speed').value);$('dispatch-hint').textContent=minutes+'分钟历史行情 · '+speed+'倍派发约'+(minutes*60/speed)+'秒，准备和保存另计';}
$('duration').onchange=updateDispatchHint;$('dispatch-speed').onchange=updateDispatchHint;updateDispatchHint();
$('start').onclick=()=>action('start',{minutes:Number($('duration').value),speed:Number($('dispatch-speed').value),slots:Number($('slots').value),max_age_us:Number($('age-limit').value)}).catch(e=>$('notice').textContent=e.message);
$('close').onclick=()=>action('close').catch(e=>$('notice').textContent=e.message);$('export').onclick=()=>window.location.assign('/api/export');
async function poll(){if(polling)return;polling=true;try{const r=await fetch('/api/state',{cache:'no-store'});if(!r.ok)throw Error('状态请求失败');render(await r.json());}catch(e){$('notice').textContent='窗口服务连接中断；已产生的结果保留在日志目录。';}finally{polling=false;}}

