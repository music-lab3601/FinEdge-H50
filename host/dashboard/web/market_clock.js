'use strict';
// Smooth only the scheduled market clock. No event or inference counts enter
// this class; those remain driven by actual source publications and ACKs.
globalThis.FinEdgeMarketClock=class {
 constructor(){this.sample=null;this.lastMarketMs=-Infinity;}
 update(sample,receivedAtMs,roundTripMs=0){
  if(sample.mode!=='live'){this.sample=null;this.lastMarketMs=-Infinity;return;}
  if(roundTripMs>300)return; // A delayed browser response is not a fresh clock sync.
  const key=sample.directory+':'+sample.round+':'+sample.session;
  if(this.sample?.key!==key)this.lastMarketMs=-Infinity;
  this.sample={...sample,key,receivedAtMs,roundTripMs};
 }
 value(nowMs,directory,round){
  const s=this.sample;
  if(!s||s.directory!==directory||s.round!==round||nowMs-s.receivedAtMs>1500)return null;
  const elapsed=Math.max(0,Math.min(s.wall_duration_ms,s.elapsed_wall_ms+nowMs-s.receivedAtMs+s.roundTripMs/2));
  const market=Math.max(this.lastMarketMs,s.origin_market_unix_ms+elapsed*s.speed);
  this.lastMarketMs=market;
  return new Date(market+s.utc_offset_minutes*60000).toISOString().slice(11,19);
 }
};
