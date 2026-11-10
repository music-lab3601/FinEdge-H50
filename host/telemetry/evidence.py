"""Read-only source publications and identity-joined single-window evidence.

No source progress is inferred from a scheduled clock. Actual API completion
records drive the counters. Replays filter dispatches, ACKs and terminals by
their original PC QPC timestamps; prices at row+50 require a visible ACK.
"""
from datetime import datetime,timezone,timedelta
from pathlib import Path
import math,time
import numpy as np
from host.telemetry.telemetry import ROW, TRACE, OBS, read_json
from host.telemetry.dispatch_timing import checked_speed, wall_duration_ns

PUB=np.dtype([(n,'<u8') for n in ('packet','ordinal','count','due','ready','submitted','done','overdue')])
RAW=np.dtype([('length','u1'),('rtype','u1'),('publisher','<u2'),('instrument','<u4'),('event_ns','<u8'),('order','<u8'),('price','<i8'),('size','<u4'),('flags','u1'),('channel','u1'),('action','u1'),('side','u1'),('recv_ns','<u8'),('delta','<i4'),('sequence','<u4')])
ACK_DTYPE=np.dtype([(n,'<u8') for n in ('stamp','flags','ordinal','eligible','windows','batches','error','ticks','session')])
NY=timezone(timedelta(hours=-5)) # This fixed source date is December15, standard time.
ACTIONS={'A':'新增','C':'撤单','M':'修改','T':'成交','F':'成交减量','R':'清空','N':'状态'}

def market_time(ns):
    seconds,fraction=divmod(int(ns),1_000_000_000)
    return datetime.fromtimestamp(seconds,NY).strftime('%H:%M:%S')+'.'+str(fraction).zfill(9)

def complete_records(path,dtype):
    path=Path(path)
    if not path.exists():return np.empty(0,dtype=dtype)
    with path.open('rb') as stream:
        count=path.stat().st_size//dtype.itemsize
        return np.fromfile(stream,dtype=dtype,count=count)

class ReplayClock:
    def __init__(self):self.enabled=False;self.playing=False;self.speed=1.;self.position=0;self.updated=time.monotonic_ns()
    def value(self,total):
        now=time.monotonic_ns()
        if self.playing:self.position=min(total,self.position+int((now-self.updated)*self.speed))
        self.updated=now
        if self.position>=total:self.playing=False
        return min(total,max(0,self.position))
    def command(self,operation,total,**payload):
        self.value(total);self.enabled=True
        if operation=='play':
            if self.position>=total:self.position=0
            self.playing=True
        elif operation=='pause':self.playing=False
        elif operation=='seek':
            seconds=payload.get('seconds',0)
            if type(seconds) not in (float,int) or not math.isfinite(seconds) or not 0<=seconds<=total/1e9+.001:raise ValueError('无效回放时间。')
            self.position=min(total,int(seconds*1e9));self.playing=False
        elif operation=='speed':
            speed=payload.get('speed')
            if speed not in (.1,.25,.5,1,2,4):raise ValueError('无效复盘速度。')
            self.speed=float(speed)
        elif operation=='end':self.position=total;self.playing=False
        else:raise ValueError('未知复盘操作。')
        self.updated=time.monotonic_ns()

class Evidence:
    def __init__(self,directory,config,telemetry):
        self.d=Path(directory);self.cfg=config;self.telemetry=telemetry
        self.raw=np.memmap(config['raw_source'],dtype=RAW,mode='r')
        self.publications=np.empty(0,dtype=PUB);self.acks=np.empty(0,dtype=ACK_DTYPE)
        self.native={};self.selected=None;self.explicit=False;self.clock=ReplayClock()
        self.manifest=None;self.start=0;self.end=0;self._trace_key=None
    def update(self):
        m=read_json(self.d/'run_manifest.json')
        if not m:return False
        self.manifest=m;f=m['fixture'];first=f['first_raw_ordinal'];last=f['last_raw_ordinal']
        source=self.d/'sender/source';final=read_json(source/'source.json')
        pub=source/('publication.bin' if final or not (source/'publication_live.bin').exists() else 'publication_live.bin')
        records=complete_records(pub,PUB)
        if len(records):
            if int(records[0]['ordinal'])!=first or np.any(records['count']<1) or np.any(records['count']>24):raise ValueError('Source publication bounds/count mismatch')
            if np.any(records['ordinal'][1:]!=records['ordinal'][:-1]+records['count'][:-1]) or np.any(np.diff(records['packet'])!=1):raise ValueError('Source publication discontinuity')
            if np.any(records['done']<records['submitted']) or np.any(records['submitted']<records['due']):raise ValueError('Source publication clock regression')
            if int(records[-1]['ordinal']+records[-1]['count']-1)>last:raise ValueError('Source publication beyond fixture')
            if final:records=records[np.cumsum(records['count'])<=int(final['events_submitted'])]
        self.publications=records
        ack_path=self.d/'sender/observations_live.bin'
        if not ack_path.exists():ack_path=self.d/'sender/observations.bin'
        self.acks=complete_records(ack_path,ACK_DTYPE)
        if len(self.acks) and np.any(self.acks['session']!=m['session']):raise ValueError('Source/ACK session mismatch')
        monitor=read_json(self.d/'ui_telemetry.json',{})
        for row in monitor.get('initial_native',[])+monitor.get('recent_native',[]):
            if row.get('reason')==0:self.native[(row['window'],row['ordinal'])]=row
        trace_path=self.d/'traces.bin';trace_key=(trace_path.stat().st_size,trace_path.stat().st_mtime_ns) if trace_path.exists() else None
        if trace_key!=self._trace_key:
            for row in complete_records(trace_path,TRACE):
                if int(row['class_index'])>>24==0:
                    self.native[(int(row['window']),int(row['ordinal']))]={k:int(row[k]) for k in ('slot','generation','npu_ns','request_ns')}
            self._trace_key=trace_key
        anchor=read_json(source/'anchor.json')
        self.start=int(anchor['anchor_ns']) if anchor else 0
        if self.start:
            self.end=max([self.start+wall_duration_ns(f.get('duration_ns',f['event_span_ns']),m.get('speed',1))]+([int(records[-1]['done'])] if len(records) else [])+
                         [p['receive_ns'] for p in self.telemetry.received]+([int(self.acks[-1]['stamp'])] if len(self.acks) else []))
        return True
    def bounds(self):return self.start,max(0,self.end-self.start)
    def _published(self,cutoff):
        if not len(self.publications):return self.publications
        return self.publications[:int(np.searchsorted(self.publications['done'],cutoff,side='right'))]
    def _committed(self,cutoff):
        a=self.acks[self.acks['stamp']<=cutoff]
        return int(a[-1]['ordinal']) if len(a) else 0
    def raw_row(self,ordinal):
        r=self.raw[ordinal-1];action=chr(int(r['action']));side=chr(int(r['side']));price=int(r['price'])
        return dict(ordinal=ordinal,sequence=int(r['sequence']),order_id=str(int(r['order'])),market_time=market_time(r['event_ns']),
                    action=action,action_name=ACTIONS.get(action,action),side={'A':'卖','B':'买','N':'—'}.get(side,side),
                    price=price/1e9 if abs(price)<9_000_000_000_000_000_000 else None,size=int(r['size']),flags=int(r['flags']))
    def source_snapshot(self,cutoff,committed):
        f=self.manifest['fixture'];first=f['first_raw_ordinal'];last=f['last_raw_ordinal'];pub=self._published(cutoff)
        sent=int(np.sum(pub['count'])) if len(pub) else 0;last_sent=first+sent-1
        target=f['measured_events'];rows=[self.raw_row(i) for i in range(max(first,last_sent-11),last_sent+1)] if sent else []
        market_duration=f.get('duration_ns',f['event_span_ns']);speed=checked_speed(self.manifest.get('speed',1));duration=wall_duration_ns(market_duration,speed)
        elapsed=max(0,min(duration,cutoff-self.start)) if self.start else 0;market_elapsed=min(market_duration,elapsed*speed)
        counts=np.bincount(((pub['submitted']-self.start)//100_000_000).astype('i8'),weights=pub['count']).astype('i8') if len(pub) else np.zeros(0,dtype='i8')
        bin_last=max(elapsed//100_000_000,len(counts)-1);counts=np.pad(counts,(0,max(0,bin_last+1-len(counts))))
        bars=[dict(seconds=i/10,events=int(n),events_per_second=int(n)*10) for i,n in enumerate(counts)] if self.start else []
        committed_measured=max(0,min(last,committed)-first+1)
        anchor_event=int(self.telemetry.rows[0]['event_ns'])
        status='preparing' if not self.start else 'completed' if sent==target and cutoff>=self.start+duration else 'holding' if sent==target else 'dispatching'
        final=read_json(self.d/'sender/source/source.json')
        if final and final['status']!='passed':status='failed'
        dataset=getattr(self,'cfg',{}).get('dataset',dict(symbol='AAPL',date='2025-12-15',data_kind='historical'))
        return dict(status=status,symbol=dataset['symbol'],date=dataset['date'],data_kind=dataset['data_kind'],schedule=f'原事件间隔{speed}倍派发',speed=speed,destination='192.168.126.50:20001 · PL千兆网',
                    market_clock=market_time(anchor_event+market_elapsed),sent=sent,total=target,packets=len(pub),committed=committed_measured,
                    awaiting_confirmation=max(0,sent-committed_measured),last_ordinal=last_sent if sent else None,progress=sent/target,
                    current_bucket_events=int(counts[bin_last]),current_rate=int(counts[bin_last])*10,peak_bucket_events=int(counts.max()) if len(counts) else 0,
                    selected_minutes=f.get('minutes'),duration_seconds=duration/1e9,elapsed_seconds=elapsed/1e9,remaining_seconds=max(0,duration-elapsed)/1e9,
                    market_duration_seconds=market_duration/1e9,market_elapsed_seconds=market_elapsed/1e9,
                    bins=bars,recent_events=rows,session=str(self.manifest['session']),
                    bootstrap_committed=min(committed,first-1),bootstrap_total=first-1,
                    submission_scope='实际UDP send完成记录；计数不由计划时间推算；100ms桶，异步发布约20ms',
                    confirmation_scope='待确认包含网络和ACK回报间隔，不等同于FPGA内部FIFO积压',
                    live_publication_available=(self.d/'sender/source/publication_live.bin').exists())
    def choose(self,opportunity=None):
        if opportunity is None:self.selected=None;self.explicit=False;return
        if type(opportunity)!=int or not 1<=opportunity<=self.manifest['fixture']['opportunities']:raise ValueError('机会编号超出本片段。')
        self.selected=opportunity;self.explicit=True
    def _publication_for(self,ordinal):
        if not len(self.publications):return None
        index=int(np.searchsorted(self.publications['ordinal'],ordinal,side='right'))-1
        if index<0:return None
        p=self.publications[index]
        return p if ordinal<int(p['ordinal']+p['count']) else None
    def _terminal_for(self,ordinal):return next((p for p in self.telemetry.received if p['ordinal']==ordinal),None)
    def tracking_snapshot(self,cutoff,committed,final=False):
        rows=self.telemetry.rows
        if self.selected is None:
            p=next((p for p in self.telemetry.received if not p['reason'] and p['receive_ns']<=cutoff),None)
            if p:self.selected=self.telemetry.by_ordinal[p['ordinal']]-98
        opportunity=self.selected or 1;i=opportunity+98;r=rows[i];ordinal=int(r['ordinal']);pub=self._publication_for(ordinal)
        sent=pub is not None and int(pub['done'])<=cutoff
        limit=int(np.searchsorted(rows['ordinal'],min(committed,self.manifest['fixture']['last_raw_ordinal']),side='right'))
        history=max(0,min(100,limit-(i-99)));terminal=self._terminal_for(ordinal)
        visible=terminal is not None and terminal['receive_ns']<=cutoff
        predicted=visible and terminal['reason']==0;cancelled=visible and terminal['reason']!=0
        ack_history=history
        # A window-specific successful inference proves its full100-row input
        # existed, even when a slower cumulative commit ACK is still in flight.
        # Preserve that ACK count separately rather than inventing a new ACK.
        if predicted:history=100
        native=self.native.get((terminal['window'],ordinal)) if visible else None
        future=None;resolved=False;future_i=i+50;future_progress=max(0,min(50,limit-i-1))
        if future_i<len(rows) and rows[future_i]['segment']==r['segment'] and int(rows[future_i]['row'])-int(r['row'])==50:
            resolved=predicted and future_i<limit
            if resolved:
                before=int(r['values'][0])+int(r['values'][2]);after=int(rows[future_i]['values'][0])+int(rows[future_i]['values'][2]);actual=2 if after>before else 0 if after<before else 1
                future=dict(class_index=actual,hit=actual==terminal['class_index'],before_mid=before/20000,after_mid=after/20000,change_usd=(after-before)/20000,
                            market_time=market_time(rows[future_i]['event_ns']),ordinal=int(rows[future_i]['ordinal']))
        due=int(pub['due']) if pub is not None and sent else None
        timing=dict(end_to_end_us=(terminal['receive_ns']-due)/1000 if predicted and due is not None else None,
                    NPU_us=native['npu_ns']/1000 if predicted and native else None,
                    board_us=terminal['board_us'] if predicted else None,service_us=terminal['service_us'] if predicted else None,
                    source_late_us=(int(pub['submitted'])-int(pub['due']))/1000 if sent else None)
        status='cancelled' if cancelled else 'resolved' if resolved else 'awaiting_future' if predicted else 'not_accepted' if final and limit>i else 'waiting'
        steps=[dict(name='原始订单派发',state='done' if sent else 'pending',detail='真实UDP提交记录' if sent else '等待实际派发'),
               dict(name='100条盘口历史',state='done' if history==100 else 'active' if history else 'pending',detail=str(history)+' / 100 · '+('本窗口推理回执确认' if predicted and ack_history<100 else 'FPGA提交确认')),
               dict(name='整数预处理 · 整窗FIFO',state='done' if predicted else 'cancelled' if cancelled or status=='not_accepted' else 'active' if history==100 else 'pending',detail='4800B原生INT8输入；完成由推理回执确认'),
               dict(name='共享DDR → NPU',state='done' if predicted else 'cancelled' if cancelled or status=='not_accepted' else 'pending',detail=('槽 '+str(native['slot'])+' · 代次 '+str(native['generation'])) if native and 'slot' in native else '真实槽身份待板端记录收集' if predicted else '无推理：本机会未接收' if status=='not_accepted' else '等待真实终态'),
               dict(name='预测返回',state='done' if predicted else 'cancelled' if cancelled or status=='not_accepted' else 'pending',detail='窗口 '+str(terminal['window']) if visible else '本机会未接收，无预测' if status=='not_accepted' else '保持同一窗口身份'),
               dict(name='后续50条兑现',state='done' if resolved else 'pending',detail=str(future_progress)+' / 50 · 实际盘口提交后揭示')]
        return dict(opportunity=opportunity,selection='manual' if self.explicit else 'first_prediction',status=status,session=str(self.manifest['session']),
                    window=str(terminal['window']) if visible else None,ordinal=ordinal if sent else None,record=self.raw_row(ordinal) if sent else None,
                    history=history,ack_history=ack_history,history_start_ordinal=int(rows[i-99]['ordinal']) if sent else None,history_end_ordinal=ordinal if sent else None,
                    prediction=dict(class_index=terminal['class_index'],logits=terminal['logits'],receive_ns=str(terminal['receive_ns'])) if predicted else None,
                    cancel_reason=terminal['reason'] if cancelled else None,slot=native.get('slot') if native and predicted else None,
                    generation=str(native['generation']) if native and predicted and 'generation' in native else None,
                    future_progress=future_progress,outcome=future,timing=timing,steps=steps,
                    outcome_status='no_prediction' if final and not predicted else 'resolved' if resolved else 'not_dispatched' if final and future_progress<50 else 'pending',
                    stage_scope='节点依据真实发送/ACK/回执确认；没有独立阶段时间戳的节点不标耗时')
    def jump(self,stage):
        rows=self.telemetry.rows;opportunity=self.selected
        if opportunity is None:
            p=next((p for p in self.telemetry.received if not p['reason']),None)
            if not p:raise ValueError('没有可追踪的真实预测。')
            opportunity=self.telemetry.by_ordinal[p['ordinal']]-98;self.choose(opportunity)
        i=opportunity+98;ordinal=int(rows[i]['ordinal']);pub=self._publication_for(ordinal);terminal=self._terminal_for(ordinal)
        if stage=='source':
            if pub is None:raise ValueError('该事件没有实际派发记录。')
            stamp=int(pub['done'])
        elif stage=='prediction':
            if not terminal or terminal['reason']:raise ValueError('该机会没有实际预测。')
            stamp=terminal['receive_ns']
        elif stage=='outcome':
            if not terminal or terminal['reason'] or i+50>=len(rows):raise ValueError('该窗口没有可兑现的预测。')
            future=int(rows[i+50]['ordinal']);a=self.acks[self.acks['ordinal']>=future]
            if not len(a):raise ValueError('尾部未来盘口未实际派发，保持未决。')
            stamp=max(int(a[0]['stamp']),terminal['receive_ns'])
        else:raise ValueError('未知追踪节点。')
        self.clock.enabled=True;self.clock.playing=False;self.clock.position=max(0,min(self.bounds()[1],stamp-self.start));self.clock.updated=time.monotonic_ns()
    def snapshot(self,final=False):
        if not self.update():return dict(source=None,tracking=None,presentation=None)
        start,total=self.bounds();position=self.clock.value(total)
        cutoff=start+position if self.clock.enabled and start else max(self.end,time.perf_counter_ns()) if final else time.perf_counter_ns()
        committed=self._committed(cutoff)
        is_end=final and (not self.clock.enabled or position>=total)
        return dict(source=self.source_snapshot(cutoff,committed),tracking=self.tracking_snapshot(cutoff,committed,is_end),
                    presentation=dict(enabled=self.clock.enabled,playing=self.clock.playing,speed=self.clock.speed,seconds=position/1e9 if self.clock.enabled else total/1e9 if final else max(0,min(total,cutoff-start))/1e9 if start else 0,total_seconds=total/1e9,
                                      available=bool(final and start and len(self.publications)),kind='真实板测记录复盘' if self.clock.enabled else '已保存真实记录' if final else '实时观察',
                                      timing_rescaled=False,clock_scope='原始PC QPC；回放速度只改变观看节奏'))
