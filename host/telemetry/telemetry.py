"""Read-only display of FMR2 results. No book reconstruction or model calls.

The book reference is independently frozen display/evaluation material. It is
advanced only by actual FPGA commit ACKs and never supplied to the inference
path. Future price labels stay hidden until their raw events were committed.
"""
from pathlib import Path
import json, math, struct, time
import numpy as np

ROW=np.dtype([(n,'<u8') for n in ('segment','row','ordinal','event_ns')]+[('values','<u8',(40,))])
TRACE=np.dtype([(n,'<u8') for n in ('window','ordinal','slot','generation','request_ns','npu_ns','board_ticks','service_ticks')]+[('logits','<f4',(3,)),('class_index','<u4'),('count_before','<u4'),('count_after','<u4')])
HEADER=struct.Struct('<IIQQQ3fIQQII');OBS=struct.Struct('<9Q')

def read_json(path,default=None):
    # Close the file before parsing. The paired display writer retries transient
    # Windows rename conflicts without ever interrupting the market source.
    try:return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:return default

def distribution(values):
    if not values:return None
    a=np.asarray(values,dtype='f8')
    return dict(count=len(a),mean_us=float(a.mean()),p99_us=float(np.percentile(a,99)),max_us=float(a.max()))

def decode_packet(blob,session):
    if len(blob)!=200:raise ValueError('FMR2 display record must be 200 bytes')
    stamp=struct.unpack_from('<Q',blob)[0];h=HEADER.unpack_from(blob,8)
    if h[:2]!=(0x32524d46,2) or h[2]!=session or h[-1]!=125000000:raise ValueError('FMR2 session/version/clock mismatch')
    code=h[8];reason=code>>24
    if reason>4 or (not reason and (code&255)>2):raise ValueError('FMR2 result class/reason')
    if not reason and not all(math.isfinite(x) for x in h[5:8]):raise ValueError('Non-finite prediction')
    return dict(receive_ns=stamp,window=h[3],ordinal=h[4],logits=list(h[5:8]),class_index=(code&255) if not reason else None,
                reason=reason,board_h50_late=bool(code&256),board_age_late=bool(code&512),board_us=h[9]*.008,service_us=h[10]*.008)

class Telemetry:
    def __init__(self,directory,reference):
        self.d=Path(directory);self.reference=Path(reference)
        self.rows=np.memmap(reference,dtype=ROW,mode='r');self.by_ordinal={int(v):i for i,v in enumerate(self.rows['ordinal'])}
        self.seen=0;self.bars={};self.received=[];self.ids=set();self.position=0;self.committed=0;self.npu={};self.manifest=None;self._trace_key=None
    def _market(self,limit):
        for row in self.rows[self.seen:limit]:
            key=int(row['event_ns'])//100_000_000;mid=(int(row['values'][0])+int(row['values'][2]))/20000
            b=self.bars.setdefault(key,dict(time=key/10-18000,open=mid,high=mid,low=mid,close=mid))
            b.update(high=max(b['high'],mid),low=min(b['low'],mid),close=mid)
        self.seen=max(self.seen,limit)
    def snapshot(self,final=False):
        m=read_json(self.d/'run_manifest.json')
        if not m:return dict(raw_events=0,records=0,opportunities=0,predictions=0,bars=[],recent_predictions=[])
        self.manifest=m;fixture=m['fixture'];first=fixture['first_raw_ordinal'];last=fixture['last_raw_ordinal']
        obs=self.d/'sender/observations_live.bin'
        if not obs.exists():obs=self.d/'sender/observations.bin'
        if obs.exists():
            count=obs.stat().st_size//OBS.size
            if count:
                with obs.open('rb') as f:f.seek((count-1)*OBS.size);r=OBS.unpack(f.read(OBS.size))
                if r[-1]!=m['session']:raise ValueError('ACK session mismatch')
                self.committed=max(self.committed,r[2])
        board=read_json(self.d/'board_result.json')
        if board and board.get('status')=='finished_parked':self.committed=last
        limit=min(int(np.searchsorted(self.rows['ordinal'],min(last,self.committed),side='right')),len(self.rows))
        self._market(limit)
        packets=self.d/'sender/predictions_live.bin'
        if final and (self.d/'sender/predictions.bin').exists():packets=self.d/'sender/predictions.bin'
        if not packets.exists():packets=self.d/'sender/predictions.bin'
        if packets.exists():
            end=packets.stat().st_size//200
            with packets.open('rb') as f:
                f.seek(self.position*200)
                for _ in range(self.position,end):
                    p=decode_packet(f.read(200),m['session']);self.position+=1
                    if not first<=p['ordinal']<=last:continue
                    if p['window'] in self.ids:raise ValueError('Duplicate terminal window in display')
                    if p['ordinal'] not in self.by_ordinal:raise ValueError('Display reference lacks returned window identity')
                    self.ids.add(p['window']);self.received.append(p)
        monitor=read_json(self.d/'ui_telemetry.json',{})
        for p in monitor.get('recent_native',[]):
            if p['reason']==0:self.npu[p['window']]=(p['ordinal'],p['npu_ns']/1000)
        trace_path=self.d/'traces.bin'
        trace_key=(trace_path.stat().st_size,trace_path.stat().st_mtime_ns) if trace_path.exists() else None
        if final and trace_key and trace_key!=self._trace_key:
            for t in np.fromfile(trace_path,dtype=TRACE):
                if int(t['class_index'])>>24==0:self.npu[int(t['window'])]=(int(t['ordinal']),int(t['npu_ns'])/1000)
            self._trace_key=trace_key
        anchor=read_json(self.d/'sender/source/anchor.json')
        due_anchor=anchor['anchor_ns'] if anchor else None;start_event=int(self.rows['event_ns'][0])
        predictions=[];ages=[];timely=within=resolved=hit=timely_hit=expired=other=0
        for p in self.received:
            if p['reason']:
                expired+=p['reason']==1;other+=p['reason']!=1;continue
            i=self.by_ordinal[p['ordinal']];r=self.rows[i];n=self.npu.get(p['window'])
            if n and n[0]!=p['ordinal']:raise ValueError('NPU timing joined to wrong raw ordinal')
            row=dict(p,opportunity=i-98,npu_us=n[1] if n else None,age_ms=None,H50_timely=None,actual_class=None,hit=None,mid_change=None,settlement='pending')
            if due_anchor is not None:
                due=due_anchor+(int(r['event_ns'])-start_event)//m.get('speed',1);age=p['receive_ns']-due
                if age<0:raise ValueError('Negative event-to-result age')
                row['age_ms']=age/1e6;ages.append(age/1000);within+=age<=5_000_000
            if i+50<len(self.rows) and self.rows[i+50]['segment']==r['segment'] and int(self.rows[i+50]['row'])-int(r['row'])==50:
                future=self.rows[i+50]
                if due_anchor is not None:
                    deadline=due_anchor+(int(future['event_ns'])-start_event)//m.get('speed',1)
                    row['H50_timely']=p['receive_ns']<=deadline;timely+=row['H50_timely']
                if i+50<limit:
                    before=int(r['values'][0])+int(r['values'][2]);after=int(future['values'][0])+int(future['values'][2])
                    row['actual_class']=2 if after>before else 0 if after<before else 1
                    row['hit']=row['class_index']==row['actual_class'];row['mid_change']=(after-before)/20000;row['settlement']='resolved'
                    resolved+=1;hit+=row['hit'];timely_hit+=bool(row['hit'] and row['H50_timely'])
                elif final:row['settlement']='not_dispatched'
            elif final:row['settlement']='no_reference'
            predictions.append(row)
        for row in predictions[-60:]:
            scores=np.asarray(row['logits'],dtype='f8');scores=np.exp(scores-scores.max());row['scores']=(scores/scores.sum()).tolist()
        metrics=monitor.get('metrics',{})
        if board and board.get('status')=='finished_parked':
            b=board['metrics'];metrics=dict(records=int(b['1']),opportunities=int(b['2']),admitted=int(b['3']),retired=int(b['5']),snapshot_full=int(b['7']),expired=int(b['29']),occupied=int(b['28']),queue_peak=int(b['27']))
        opportunities=max(0,limit-99);count=len(predictions)
        if final and opportunities!=m['fixture'].get('opportunities',opportunities):raise ValueError('Display opportunity denominator mismatch')
        latest=None
        if limit:
            r=self.rows[limit-1];book=r['values'].tolist();latest=dict(book=book,mid=(book[0]+book[2])/20000,time=int(r['event_ns'])/1e9-18000)
        analysis=read_json(self.d/'analysis.json')
        if final and analysis:
            if analysis['predictions']!=count or analysis['measured_opportunities']!=opportunities:raise ValueError('Saved analysis/display counts disagree')
            if within!=analysis['five_ms_completed'] or timely!=analysis['H50_timely']:raise ValueError('Display clock/deadline scoring differs from saved analysis')
        self.last_predictions=predictions
        return dict(raw_events=max(0,min(last,self.committed)-first+1),raw_target=fixture['measured_events'],bootstrap_committed=min(self.committed,first-1),bootstrap_target=first-1,
                    records=limit,opportunities=opportunities,expected_opportunities=fixture.get('opportunities',10000),predictions=count,expired=expired,other_cancelled=other,
                    unaccepted=opportunities-count-expired-other if final else None,board=metrics,slots=m['slots'],latest=latest,bars=list(self.bars.values()),
                    recent_predictions=predictions[-60:],latency=distribution(ages),NPU=distribution([p['npu_us'] for p in predictions if p['npu_us'] is not None]),
                    five_ms_completed=within,five_ms_coverage=within/opportunities if opportunities else None,H50_timely=timely,H50_coverage=timely/opportunities if opportunities else None,
                    evaluation=dict(resolved=resolved,hits=hit,hit_rate=hit/resolved if resolved else None,timely_hit_rate=timely_hit/resolved if resolved else None,pending=count-resolved,scope='仅已实际派发并由FPGA提交的未来盘口；同源重建参考方向'),
                    chart_source='同源重建盘口参考，按真实FPGA提交进度显示；不进入模型',timing_source='PC QPC计划派发至UDP结果接收',analysis=analysis)
