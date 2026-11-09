"""Cooperative probe stop: handlers only latch, SDK returns are retained first."""


class ProbeCancelled(RuntimeError):
    pass


class ProbeStop:
    def __init__(self):
        self.cancelled=False
        self.stage='created'
        self.requests=[]
        self.observed_at=None
        self.owners=[]
        self.ledgers=[]
        self.retained_returns=[]
        self.report_errors=[]

    def request(self,signum,frame=None):
        # No SDK calls, cleanup or asynchronous raise from a signal callback.
        self.cancelled=True
        self.requests.append(dict(signal=signum,stage=self.stage))

    def hold(self,value):
        self.retained_returns.append(value)
        return value

    def bind_owner(self,owner):
        if not any(o is owner for o in self.owners):self.owners.append(owner)

    def bind_ledger(self,ledger):
        if not any(o is ledger for o in self.ledgers):self.ledgers.append(ledger)

    def isolate(self,reason):
        for owner in self.owners:
            owner.quarantine(reason)
            for ledger in getattr(owner,'known_probe_ledgers',[]):
                if not ledger.blocked:ledger.stop(reason)
        for ledger in self.ledgers:
            if not ledger.blocked:ledger.stop(reason)

    def checkpoint(self,stage):
        self.stage=stage
        blocked=any(getattr(o,'blocked',False) or getattr(o,'failed',False) for o in self.owners)
        if blocked and not self.cancelled:
            self.cancelled=True;self.requests.append(dict(signal=None,stage=stage,reason='owner isolated'))
        if self.cancelled:
            if self.observed_at is None:self.observed_at=stage
            self.isolate('Probe stopped at '+self.observed_at)
            raise ProbeCancelled('Probe stopped at '+self.observed_at)

    def call(self,stage,operation,retain=None):
        self.checkpoint(stage+':before')
        result=operation()
        self.hold(result) # unconditional strong ref before cancellation/getters
        if retain is not None:retain(result)
        self.checkpoint(stage+':after')
        return result

    def snapshot(self):
        return dict(cancelled=self.cancelled,requests=list(self.requests),observed_at=self.observed_at,
            current_stage=self.stage,owners_retained=len(self.owners),SDK_returns_retained=len(self.retained_returns),
            report_errors=list(self.report_errors))

    def park(self,sleep,save_stop):
        saved=False
        while True:
            try:self.checkpoint('parked')
            except ProbeCancelled:
                if not saved:
                    try:save_stop(self.snapshot())
                    except Exception as error:self.report_errors.append(type(error).__name__+': '+str(error))
                    saved=True # a reporting error must not release the owner
            sleep(30) # keep all owners alive; no SDK monitoring/reconnect
