"""Shared E candidate continuity/normalization; no device or owner access."""
from collections import deque
from copy import deepcopy
import numpy as np
from .stream import InputBuilder
from .ddr_transport_contract import pack_record


SEGMENT_FIELDS=('date','symbol','snippet_id','research_segment_id','native_segment_id')
IDENTITY_FIELDS=('source_record_row_0based','source_message_position_1based',
                 'timestamp_ns_since_exchange_midnight')


class Continuity:
    """InputBuilder-equivalent eligible metadata rules, without a tensor ring."""
    def __init__(self, *, allow_segments=True):
        self.allow_segments=allow_segments
        self.previous=None;self.key=None;self.epoch=0;self.count=0
        self.endpoints=deque(maxlen=100)
        self.failed=False

    def accept(self, meta, identity=None):
        if self.failed:raise ValueError('Continuity already failed')
        try:
            key=tuple(meta[k] for k in SEGMENT_FIELDS)
            for k in (*IDENTITY_FIELDS,'sample_row_0based','research_reject_mask','bid_levels','ask_levels'):
                if type(meta[k]) is not int or not 0<=meta[k]<2**64:
                    raise ValueError('Invalid metadata integer: '+k)
            if meta['research_reject_mask'] or min(meta['bid_levels'],meta['ask_levels'])<10:
                raise ValueError('Record is not eligible with complete top ten')
            changed=self.previous is None or key!=self.key
            if changed:
                if self.previous is not None and not self.allow_segments:
                    raise ValueError('Initial E plan covers one fixed continuous segment')
                self.epoch+=1;self.count=0;self.endpoints.clear()
            else:
                old=self.previous
                if (meta['source_record_row_0based']!=old['source_record_row_0based']+1
                    or meta['sample_row_0based']!=old['sample_row_0based']+1):
                    raise ValueError('Discontinuous record rows; window cleared')
                if (meta['source_message_position_1based']<=old['source_message_position_1based']
                    or meta['timestamp_ns_since_exchange_midnight']<old['timestamp_ns_since_exchange_midnight']):
                    raise ValueError('Source message or timestamp regressed')
            ident=(self.epoch,*(meta[k] for k in IDENTITY_FIELDS))
            if identity is not None and (len(identity)!=4 or any(type(x) is not int for x in identity)
                                         or tuple(identity)!=ident):
                raise ValueError('Metadata and WDM identity differ')
            self.count+=1;self.endpoints.append(ident)
            self.previous=deepcopy(meta);self.key=key
            return dict(identity=ident,segment_changed=changed,opportunity=self.count>=100,
                        selected=self.count>=100 and (self.count-100)%16==0,
                        first=self.endpoints[0],last=ident)
        except (ValueError,KeyError,TypeError,OverflowError):
            self.count=0;self.endpoints.clear();self.failed=True
            raise


class FrozenNormalizer:
    # Call the exact frozen method, without constructing InputBuilder's ring.
    normalize=InputBuilder.normalize

    def __init__(self, normalization):
        self.mean=np.asarray(normalization['mean'],dtype=np.float64)
        self.std=np.asarray(normalization['std'],dtype=np.float64)
        if (self.mean.shape!=(40,) or self.std.shape!=(40,) or not np.isfinite(self.mean).all()
            or not np.isfinite(self.std).all() or np.any(self.std<=0)):
            raise ValueError('Invalid forty-channel normalization')


class CandidateFrontend:
    def __init__(self, normalization, arm):
        if arm not in ('S','H','C'):raise ValueError('Unknown E arm')
        self.arm=arm;self.normalizer=FrozenNormalizer(normalization);self.continuity=Continuity()
        self.ring=None if arm=='H' else np.empty((100,40),dtype='<f4')
        self.pointer=0

    def push(self, record, identity=None):
        try:
            info=self.continuity.accept(record.meta,identity)
            if info['segment_changed']:self.pointer=0
            normalized=self.normalizer.normalize(record.values)
            if self.ring is not None:
                self.ring[self.pointer]=normalized;self.pointer=(self.pointer+1)%100
            info['normalized']=normalized
            return info
        except (ValueError,KeyError,TypeError,OverflowError):
            self.continuity.count=0;self.continuity.endpoints.clear();self.continuity.failed=True
            self.pointer=0
            raise

    def pack(self, info):
        """Separate packing interval; H creates WDM, S/C only selected tensors."""
        if self.arm=='H':
            return pack_record(tuple(int(v) for v in info['normalized'].view('<u4')),
                               info['identity'],info['selected'])
        if not info['selected']:return None
        out=np.empty((1,40,100),dtype='<f4');tail=100-self.pointer
        out[0,:,:tail]=self.ring[self.pointer:].T
        if self.pointer:out[0,:,tail:]=self.ring[:self.pointer].T
        return out.tobytes()
