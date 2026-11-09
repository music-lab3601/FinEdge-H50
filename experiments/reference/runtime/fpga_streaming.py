"""Bounded, stride-one WDM batching for a future full-day hardware transport.

No hardware fallback or device allocation lives here. Display coalescing is
separate from the lossless record/window stream.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from threading import Lock

from experiments.reference.runtime.udw_e_frontend import CandidateFrontend


@dataclass(frozen=True)
class WdmBatch:
    sequence: int
    continuation: bool
    payload: bytes
    records: tuple
    endpoints: tuple
    first_window_id: int

    @property
    def window_count(self):
        return len(self.endpoints)


class WdmBatcher:
    """Consume a fresh raw source, normalize once, emit every eligible window.

    The caller must drain/ack each yielded batch before requesting the next.
    A segment change flushes the prior batch, and marks the new batch as a
    fresh window history. At most one record is read ahead at this boundary.
    """
    def __init__(self, normalization, batch_records=32):
        if type(batch_records) is not int or not 1 <= batch_records <= 32:
            raise ValueError('Batch must contain 1..32 records')
        self.front = CandidateFrontend(normalization,'H')
        self.capacity = batch_records
        self.started = self.exhausted = False
        self.record_count = self.window_count = self.batch_count = 0
        self.max_batch_records = self.max_payload_bytes = 0

    def batches(self, records):
        if self.started:
            raise ValueError('A stream cannot be reset or replayed implicitly')
        self.started = True
        payload, saved, endpoints = [], [], []
        continuation = False
        first_window_id = 1

        def flush():
            self.batch_count += 1
            raw = b''.join(payload)
            self.max_batch_records=max(self.max_batch_records,len(saved))
            self.max_payload_bytes=max(self.max_payload_bytes,len(raw))
            return WdmBatch(self.batch_count,continuation,raw,tuple(saved),
                            tuple(endpoints),first_window_id)

        for record in records:
            item = self.front.push(record)
            # CandidateFrontend historically selected stride16. Full-day mode
            # explicitly selects EVERY opportunity; packet batching is not sampling.
            item['selected'] = item['opportunity']
            if item['segment_changed']:
                if payload:
                    yield flush()
                    payload,saved,endpoints=[],[],[]
                continuation=False
                first_window_id=self.window_count+1
            if not payload:
                first_window_id=self.window_count+1
            payload.append(self.front.pack(item))
            saved.append(record)
            self.record_count += 1
            if item['selected']:
                endpoints.append((tuple(item['first']),tuple(item['last'])))
                self.window_count += 1
            if len(saved)==self.capacity:
                yield flush()
                payload,saved,endpoints=[],[],[]
                continuation=True
        if payload:
            yield flush()
        self.exhausted=True

    def summary(self):
        return dict(records=self.record_count,windows=self.window_count,batches=self.batch_count,
            exhausted=self.exhausted,segments=self.front.continuity.epoch,
            max_batch_records=self.max_batch_records,max_payload_bytes=self.max_payload_bytes,
            stride=1,model_batch=1)


class BatchAccounting:
    """One in-flight batch; ambiguous/failed acknowledgements stop the stream."""
    def __init__(self):
        self.next_sequence=1
        self.submitted=self.completed=self.records=0
        self.pending=None
        self.failed=False

    def submit(self,batch):
        if self.failed or self.pending is not None or batch.sequence!=self.next_sequence:
            self.failed=True
            raise ValueError('Unexpected batch or previous batch still in flight')
        if len(batch.payload)!=len(batch.records)*256:
            self.failed=True
            raise ValueError('Record byte count mismatch')
        self.pending=batch
        self.submitted+=batch.window_count
        self.records+=len(batch.records)

    def acknowledge(self,sequence,predictions):
        try:
            batch=self.pending
            if self.failed or batch is None or sequence!=batch.sequence or len(predictions)!=batch.window_count:
                raise ValueError('Batch acknowledgement/count mismatch')
            for index,(prediction,endpoints) in enumerate(zip(predictions,batch.endpoints)):
                if (prediction.get('window_id')!=batch.first_window_id+index
                    or tuple(prediction.get('first',()))!=endpoints[0]
                    or tuple(prediction.get('last',()))!=endpoints[1]):
                    raise ValueError('Window identity/order mismatch')
            self.completed+=batch.window_count
            self.next_sequence+=1
            self.pending=None
        except Exception:
            self.failed=True
            raise

    def finish(self):
        if self.failed or self.pending is not None or self.submitted!=self.completed:
            raise ValueError('Incomplete or failed stream; no silent drops')
        return dict(records=self.records,submitted=self.submitted,completed=self.completed,
                    batches=self.next_sequence-1)


class LatestDisplay:
    """Single owned snapshot, never an inference/log queue.

    Counters and every result must already have been recorded before publish.
    GUI polls at its own rate; later producer mutations cannot change its view.
    """
    def __init__(self):
        self.lock=Lock()
        self.latest=None
        self.published=self.coalesced=0

    def publish(self,snapshot):
        owned=deepcopy(snapshot)
        self.publish_owned(owned)

    def publish_owned(self,snapshot):
        """Transfer a fresh snapshot with no mutable aliases to its producer.

        Caller relinquishes mutation rights to the snapshot and all mutable
        descendants. Use publish() when exclusive ownership is not proven.
        The consumer receives the same owned object through take().
        """
        with self.lock:
            self.coalesced+=int(self.latest is not None)
            self.latest=snapshot
            self.published+=1

    def take(self):
        with self.lock:
            value=self.latest
            self.latest=None
            return value
