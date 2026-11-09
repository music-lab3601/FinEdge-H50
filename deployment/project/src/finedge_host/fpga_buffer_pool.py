"""Fixed-size lease bookkeeping; no currently released hardware adapter.

Actual device completion/visibility evidence must come from a separately
validated transport. ISSUED, readback equality, or an elapsed delay cannot
construct a producer completion. Local simulations exercise this protocol.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Lease:
    boot_id: str
    allocation_id: str
    slot: int
    generation: int
    request_id: int


@dataclass(frozen=True)
class ProducerCompletion:
    lease: Lease
    issued_writes: int
    completed_writes: int
    outstanding_reads: int
    visibility_fence: str
    contract_id: str


@dataclass(frozen=True)
class ConsumerCompletion:
    lease: Lease
    input_released: bool
    output_copied: bool
    contract_id: str


class FixedPool:
    """Bounded metadata and strong refs: one lease and owned objects per slot.

    Matching contract IDs identify the transport evidence, not a proof by
    themselves. This class does not discover, validate or enable hardware.
    """
    def __init__(self,boot_id,allocation_id,slot_count,slot_bytes,*,producer_contract,consumer_contract):
        if (not boot_id or not allocation_id or type(slot_count) is not int or not 1<=slot_count<=4
            or type(slot_bytes) is not int or slot_bytes<=0 or slot_bytes%64
            or not producer_contract or not consumer_contract):
            raise ValueError('Explicit bounded pool and completion contracts required')
        self.boot_id,self.allocation_id=boot_id,allocation_id
        self.slot_bytes=slot_bytes
        self.producer_contract,self.consumer_contract=producer_contract,consumer_contract
        self.slots=[dict(state='FREE',generation=0,lease=None,held=[]) for _ in range(slot_count)]
        self.blocked=False
        self.reason=None
        self.request_id=self.reuses=0

    def quarantine(self,reason):
        self.blocked=True
        self.reason=str(reason)
        for slot in self.slots:
            if slot['state']!='FREE':slot['state']='QUARANTINED'

    def view(self,lease,state):
        if self.blocked:
            raise RuntimeError('Pool quarantined: '+str(self.reason))
        if (type(lease) is not Lease or type(lease.slot) is not int or not 0<=lease.slot<len(self.slots)
            or self.slots[lease.slot]['lease']!=lease or self.slots[lease.slot]['state']!=state):
            self.quarantine('Stale lease or invalid lifecycle transition')
            raise ValueError(self.reason)
        return self.slots[lease.slot]

    def acquire(self):
        if self.blocked:raise RuntimeError('Pool quarantined')
        for index,slot in enumerate(self.slots):
            if slot['state']=='FREE':
                if slot['generation']==2**64-1 or self.request_id==2**64-1:
                    self.quarantine('Generation exhausted');raise OverflowError(self.reason)
                self.request_id+=1
                slot['generation']+=1
                lease=Lease(self.boot_id,self.allocation_id,index,slot['generation'],self.request_id)
                slot.update(state='CPU_READY',lease=lease)
                return lease
        raise BufferError('No free slot; apply backpressure')

    def hold(self,lease,obj,state):
        self.view(lease,state)['held'].append(obj)

    def submitted(self,lease):
        self.view(lease,'CPU_READY')['state']='PL_INFLIGHT'

    def visible(self,receipt):
        if type(receipt) is not ProducerCompletion:
            self.quarantine('Producer completion receipt required');raise ValueError(self.reason)
        slot=self.view(receipt.lease,'PL_INFLIGHT')
        values=(receipt.issued_writes,receipt.completed_writes,receipt.outstanding_reads)
        if (any(type(v) is not int for v in values) or receipt.issued_writes<=0
            or receipt.issued_writes!=receipt.completed_writes or receipt.outstanding_reads!=0
            or not receipt.visibility_fence or receipt.contract_id!=self.producer_contract):
            self.quarantine('Writes not drained/visible');raise ValueError(self.reason)
        slot['state']='PL_COMPLETE_VISIBLE'

    def host_copied(self,lease,owned_input):
        slot=self.view(lease,'PL_COMPLETE_VISIBLE')
        slot['held'].append(owned_input)
        slot['state']='ARM_COPIED'

    def npu_submitted(self,lease):
        self.view(lease,'ARM_COPIED')['state']='NPU_INFLIGHT'

    def consumed(self,receipt):
        if type(receipt) is not ConsumerCompletion:
            self.quarantine('Consumer completion receipt required');raise ValueError(self.reason)
        slot=self.view(receipt.lease,'NPU_INFLIGHT')
        if (receipt.input_released is not True or receipt.output_copied is not True
            or receipt.contract_id!=self.consumer_contract):
            self.quarantine('NPU input still owned or output incomplete');raise ValueError(self.reason)
        slot['state']='NPU_COMPLETE'

    def release(self,lease):
        slot=self.view(lease,'NPU_COMPLETE')
        slot.update(state='FREE',lease=None,held=[])
        self.reuses+=1

    def summary(self):
        return dict(slots=len(self.slots),reserved_bytes=len(self.slots)*self.slot_bytes,
            states=[s['state'] for s in self.slots],held_objects=sum(len(s['held']) for s in self.slots),
            issued=self.request_id,released=self.reuses,blocked=self.blocked,reason=self.reason)
