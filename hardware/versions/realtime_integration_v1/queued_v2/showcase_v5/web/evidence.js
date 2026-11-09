function burstChart(bins){
 const svg=$('source-bursts'),NS='http://www.w3.org/2000/svg',W=svg.clientWidth||1200,H=svg.clientHeight||220,L=45,R=15,T=12,B=30;
 svg.setAttribute('viewBox','0 0 '+W+' '+H);svg.replaceChildren();if(!bins.length)return;
 const max=Math.max(10,...bins.map(b=>b.events)),xspan=W-L-R,yspan=H-T-B;
 function shape(name,attrs,text){const x=document.createElementNS(NS,name);for(const [k,v] of Object.entries(attrs))x.setAttribute(k,String(v));if(text!==undefined)x.textContent=text;svg.append(x);return x;}
 for(let i=0;i<=4;i++){const y=T+yspan*(1-i/4);shape('line',{x1:L,y1:y,x2:W-R,y2:y,stroke:'#263a50'});shape('text',{x:L-8,y:y+4,fill:'#91aac5','font-size':11,'text-anchor':'end'},Math.round(max*i/4));}
 const bw=xspan/bins.length;
 bins.forEach((b,i)=>{const h=b.events/max*yspan,r=shape('rect',{x:L+i*bw,y:T+yspan-h,width:Math.max(.7,bw-1),height:h,fill:i===bins.length-1?'#83e1c6':'#61a3db',rx:1});const title=document.createElementNS(NS,'title');title.textContent=b.seconds.toFixed(1)+'s：'+b.events+'条原始事件';r.append(title);});
 for(let i=0;i<=4;i++){const index=Math.min(bins.length-1,Math.floor((bins.length-1)*i/4));shape('text',{x:L+xspan*i/4,y:H-5,fill:'#91aac5','font-size':11,'text-anchor':'middle'},bins[index].seconds.toFixed(1)+'s');}
}
let evidenceChartKey='',seekDragging=false;
const trackStatus={waiting:'等待真实回执',awaiting_future:'预测已返回 · 等待兑现',resolved:'预测与实际已对照',cancelled:'过期或取消 · 未进行预测',not_accepted:'本机会未接收'};
function renderEvidence(s){
 const src=s.source,tr=s.tracking,rv=s.presentation||{};
 $('evidence-mode').textContent=rv.kind||'等待运行记录';
 $('replay-explainer').textContent=s.busy?'真实行情运行中；本轮完成后可回放和逐步观察。':'真实记录复盘只改变观看节奏；延时始终来自原始测量。';
 for(const id of ['replay-zero','replay-play','replay-pause','replay-speed','replay-end','replay-seek','jump-source','jump-prediction','jump-outcome'])$(id).disabled=!rv.available;
 $('replay-play').textContent=rv.playing?'正在播放':'播放记录';$('replay-position').textContent=num((s.busy&&src?src.elapsed_seconds:rv.seconds)||0,3)+' / '+num(rv.total_seconds||0,3)+' s';
 $('replay-speed').value=String(rv.speed||1);$('replay-seek').max=String(rv.total_seconds||1);if(!seekDragging)$('replay-seek').value=String((s.busy&&src?src.elapsed_seconds:rv.seconds)||0);
 if(src){
  if(!paintMarketClock())$('source-clock').textContent=(src.market_clock||'—').split('.')[0];$('source-clock-note').textContent=s.busy?'计划时间轴 · 现实1秒对应历史'+(src.speed||1)+'秒':rv.enabled?(rv.playing?'记录观看中 · 原始派发'+(src.speed||1)+'倍':'记录已暂停'):'已保存记录 · 原始派发'+(src.speed||1)+'倍';$('source-sent').textContent=src.sent.toLocaleString()+' / '+src.total.toLocaleString();$('source-progress').style.width=(src.progress*100)+'%';
  $('source-committed').textContent=src.committed.toLocaleString()+' 条';$('source-pending').textContent='待确认 '+src.awaiting_confirmation.toLocaleString()+' 条 · 含网络与回报间隔';
  $('source-rate').textContent=src.current_rate.toLocaleString()+' 条/s';$('source-peak').textContent='峰值桶 '+src.peak_bucket_events.toLocaleString()+' 条 / 100ms';$('source-packets').textContent=src.packets.toLocaleString()+' 个实际UDP包';
  $('source-last').textContent='原始记录 #'+(src.last_ordinal??'—');
  $('source-status').textContent=rv.enabled?(rv.playing?'真实记录 · 回放中':'真实记录 · 已暂停'):src.status==='completed'?'本片段派发完成':src.status==='dispatching'?'真实UDP派发中':src.status==='holding'?'事件已派发 · 等待时长结束':src.status==='failed'?'派发失败 · 记录保留':'等待测量片段';if(Number.isFinite(src.remaining_seconds)&&src.selected_minutes){const rem=Math.ceil(src.remaining_seconds);$('source-status').textContent+=' · '+src.selected_minutes+'分钟历史 · '+(src.speed||1)+'倍 / 剩余 '+Math.floor(rem/60)+':'+String(rem%60).padStart(2,'0');if(s.busy||s.review)$('duration').value=String(src.selected_minutes)};
  document.querySelector('.source-flow').classList.toggle('dispatching',Boolean(rv.playing||(!rv.enabled&&src.status==='dispatching')));
  const key=JSON.stringify(src.bins);if(activeView==='source'&&key!==evidenceChartKey){burstChart(src.bins);evidenceChartKey=key;}
  $('source-events').replaceChildren(...src.recent_events.slice().reverse().map(e=>{const row=document.createElement('tr');for(const value of ['#'+e.ordinal,e.sequence,e.market_time,e.action+' · '+e.action_name,e.side,e.order_id,e.price===null?'—':num(e.price,4),e.size.toLocaleString()])cell(row,value);return row;}));
 }
 if(tr){
  $('track-input').max=String(s.telemetry?.expected_opportunities||500000);
  if(document.activeElement!==$('track-input'))$('track-input').value=String(tr.opportunity);
  $('track-state').textContent=trackStatus[tr.status]||tr.status;
  $('track-identity').textContent='机会 #'+tr.opportunity+' · 硬件窗口 '+(tr.window??'待返回')+' · 原始记录 '+(tr.ordinal??'尚未派发')+' · 会话 '+tr.session;
  $('track-history').textContent=tr.history+' / 100';$('track-range').textContent=tr.history_start_ordinal===null?'100条有效盘口，40维价量特征':'原始记录 '+tr.history_start_ordinal+'～'+tr.history_end_ordinal+' 范围内的100条有效盘口';
  $('track-slot').textContent=tr.slot===null?'待同窗trace':tr.slot+' / 32 · 代次 '+tr.generation;
  $('track-steps').replaceChildren(...tr.steps.map((step,i)=>{const card=document.createElement('div'),n=document.createElement('span'),title=document.createElement('b'),detail=document.createElement('small');card.className='trace-step '+step.state;n.textContent=step.state==='done'?'✓':String(i+1);title.textContent=step.name;detail.textContent=step.detail;card.append(n,title,detail);return card;}));
  const p=tr.prediction,o=tr.outcome;
  $('track-prediction').textContent=p?names[p.class_index]:tr.cancel_reason?'已取消':'等待返回';$('track-prediction').className=p?['down','flat','up'][p.class_index]:'';
  $('track-logits').textContent=p?'logits ['+p.logits.map(v=>num(v,5)).join(', ')+']':'logits —';
  $('track-raw-event').textContent=tr.record?tr.record.market_time+' · '+tr.record.action_name+' · '+tr.record.side+' · '+tr.record.size+'股':'原始订单尚未派发';
  $('track-future-progress').style.width=(tr.future_progress*2)+'%';$('track-future-count').textContent=tr.future_progress+' / 50 已确认';
  $('track-actual').textContent=o?names[o.class_index]:tr.outcome_status==='no_prediction'?'无预测可对照':tr.outcome_status==='not_dispatched'?'尾部未决':'等待兑现';$('track-actual').className=o?['down','flat','up'][o.class_index]:'';
  $('track-prices').textContent=o?'$'+num(o.before_mid,5)+' → $'+num(o.after_mid,5)+' · 变化 '+num(o.change_usd,5):'未来价格未到达前保持隐藏';
  $('track-hit').textContent=o?(o.hit?'方向命中':'方向未命中'):'—';$('track-hit').className=o?(o.hit?'hit':'miss'):'';
  $('track-future-event').textContent=o?o.market_time+' · 原始记录 #'+o.ordinal:'实际派发并被FPGA提交后揭示';
  const tm=tr.timing||{};$('track-total-time').textContent=num(tm.end_to_end_us,3)+' μs';$('track-board-time').textContent=num(tm.board_us,3)+' μs';$('track-npu-time').textContent=num(tm.NPU_us,2)+' μs';
 }
}
const evidenceError=e=>{$('notice').textContent=e.message;$('notice').classList.add('error');};
$('track-pin').onclick=()=>action('track',{opportunity:Number($('track-input').value)}).catch(evidenceError);
$('track-input').onkeydown=e=>{if(e.key==='Enter')$('track-pin').click();};
$('track-first').onclick=()=>action('track',{}).catch(evidenceError);
for(const stage of ['source','prediction','outcome'])$('jump-'+stage).onclick=()=>action('jump',{stage}).catch(evidenceError);
for(const [id,operation] of [['replay-play','play'],['replay-pause','pause'],['replay-end','end']])$(id).onclick=()=>action('replay',{operation}).catch(evidenceError);
$('replay-zero').onclick=()=>action('replay',{operation:'seek',seconds:0}).catch(evidenceError);
$('replay-speed').onchange=()=>action('replay',{operation:'speed',speed:Number($('replay-speed').value)}).catch(evidenceError);
$('replay-seek').onpointerdown=()=>{seekDragging=true;};
$('replay-seek').onchange=()=>{const seconds=Number($('replay-seek').value);seekDragging=false;action('replay',{operation:'seek',seconds}).catch(evidenceError);};

let focused=new URLSearchParams(location.search).get('focus')==='1';
function recordingFocus(){document.body.classList.toggle('recording-focus',focused);evidenceChartKey='';$('focus-mode').textContent=focused?'展开运行设置':'录屏聚焦';}
$('focus-mode').onclick=()=>{focused=!focused;recordingFocus();};recordingFocus();
poll();setInterval(poll,400);

// Small independent clock requests avoid the full state scan and chart payload.
const marketDisplayClock=new FinEdgeMarketClock();let clockPolling=false;
function paintMarketClock(){
 if(!state?.busy||state.review)return false;
 const text=marketDisplayClock.value(performance.now(),state.directory,state.round);
 if(text===null)return false;
 if($('source-clock').textContent!==text)$('source-clock').textContent=text;
 return true;
}
async function pollMarketClock(){
 if(clockPolling)return;clockPolling=true;
 const begin=performance.now();
 try{const response=await fetch('/api/clock',{cache:'no-store'});if(response.ok){const value=await response.json(),end=performance.now();marketDisplayClock.update(value,end,end-begin);paintMarketClock();}}
 catch{}finally{clockPolling=false;}
}
pollMarketClock();setInterval(pollMarketClock,300);setInterval(paintMarketClock,100);
