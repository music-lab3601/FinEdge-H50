"""Pre-RTL 32-bit streaming contract model. No floating-point arithmetic.

step() observes signals BEFORE the edge and applies handshakes AT that edge.
The ring is frozen during output. This is a specification model, not RTL.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RowIdentity:
    epoch: int
    ordinal: int
    source: int
    timestamp: int

    def __post_init__(self):
        if any(type(v) is not int or not 0 <= v < 2**64 for v in
               (self.epoch,self.ordinal,self.source,self.timestamp)):
            raise ValueError('Identity fields must be uint64')


@dataclass(frozen=True)
class Beat:
    data: int
    sop: bool
    last: bool
    identity: RowIdentity
    emit: bool = True

    def __post_init__(self):
        if type(self.data) is not int or not 0 <= self.data < 2**32:
            raise ValueError('Data must be uint32 bits')


class WindowCoreModel:
    def __init__(self):
        self.clear()

    def clear(self):
        self.ring=[[0]*40 for _ in range(100)]
        self.ids=[None]*100
        self.pointer=self.count=0
        self.previous=None
        self.stage=[]
        self.pending=None
        self.pending_emit=False
        self.phase='input'
        self.cursor=0
        self.errors=0

    def _fault(self, last):
        self.count=self.pointer=0
        self.previous=None
        self.stage=[]
        self.pending=None
        self.pending_emit=False
        self.phase='input' if last else 'drain'
        self.errors=min(2**32-1,self.errors+1)

    def signals(self):
        valid=self.phase=='output'
        result={'s_ready':self.phase in ('input','drain'),'m_valid':valid,
                'm_data':None,'m_sop':False,'m_last':False,'m_first':None,'m_last_identity':None,
                'errors':self.errors,'phase':self.phase}
        if valid:
            channel,t=divmod(self.cursor,100)
            result.update(m_data=self.ring[(self.pointer+t)%100][channel],
                          m_sop=self.cursor==0,m_last=self.cursor==3999,
                          m_first=self.ids[self.pointer],m_last_identity=self.previous)
        return result

    def step(self, beat=None, m_ready=True, reset=False):
        observed=self.signals()
        if reset:
            # Reset/clear cancels the edge's transfers and aborts a partial output.
            self.clear()
            return dict(observed,s_ready=False,m_valid=False)
        phase=self.phase
        if phase=='output':
            if m_ready:
                self.cursor+=1
                if self.cursor==4000:self.phase='input';self.cursor=0
        elif phase=='latency':
            self.cursor-=1
            if self.cursor==0:self.phase='output'
        elif phase=='commit':
            self.ring[self.pointer][self.cursor]=self.stage[self.cursor]
            self.cursor+=1
            if self.cursor==40:
                self.ids[self.pointer]=self.pending
                self.pointer=(self.pointer+1)%100
                self.count=min(100,self.count+1)
                self.previous=self.pending
                emit=self.pending_emit
                self.stage=[];self.pending=None;self.pending_emit=False;self.cursor=0
                if self.count==100 and emit:self.phase='latency';self.cursor=2
                else:self.phase='input'
        elif beat is not None:
            if phase=='drain':
                if beat.last:self.phase='input'
            else:
                index=len(self.stage)
                if (beat.sop != (index==0) or beat.last != (index==39)
                    or (index>0 and (beat.identity!=self.pending or beat.emit!=self.pending_emit))):
                    self._fault(beat.last)
                else:
                    if index==0:self.pending=beat.identity;self.pending_emit=beat.emit
                    self.stage.append(beat.data)
                    if index==39:
                        prev=self.previous;new=self.pending
                        if prev is not None and prev.epoch==new.epoch and not (
                            new.ordinal==prev.ordinal+1 and new.source>prev.source and new.timestamp>=prev.timestamp):
                            self._fault(True)
                        else:
                            if prev is not None and prev.epoch!=new.epoch:self.count=self.pointer=0
                            self.phase='commit';self.cursor=0
        return observed
