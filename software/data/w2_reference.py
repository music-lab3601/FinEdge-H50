"""Independent order-map replay: manual byte offsets, no production decoder/book.

Reads selected raw messages captured during the original scan. This cross-check
does not independently prove that the original feed had no missing messages.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import struct

def replay(folder):
    folder=Path(folder)
    checks={x['sequence']:x for x in map(json.loads,(folder/'checkpoints.jsonl').read_text(encoding='utf-8').splitlines())}
    orders={};mutations=0;checked=0;last=0
    def number(payload,start,size):return int.from_bytes(payload[start:start+size],'big')
    with (folder/'selected_messages.bin').open('rb') as f:
        while header:=f.read(10):
            if len(header)!=10:raise ValueError('Reference trace truncated')
            seq,length=struct.unpack('<QH',header);p=f.read(length)
            if len(p)!=length or seq<=last:raise ValueError('Reference trace invalid')
            last=seq;t=chr(p[0])
            if t in 'AF':
                oid=number(p,11,8);side=chr(p[19]);qty=number(p,20,4);price=number(p,32,4)
                if oid in orders or side not in 'BS' or qty<=0 or price<=0:raise ValueError('Reference add error')
                orders[oid]=[side,price,qty];mutations+=1
            elif t in 'ECXDU':
                oid=number(p,11,8)
                if oid not in orders:raise ValueError('Reference missing order')
                side,price,qty=orders[oid]
                if t=='U':
                    new=number(p,19,8);new_qty=number(p,27,4);new_price=number(p,31,4)
                    if new in orders or new_qty<=0 or new_price<=0:raise ValueError('Reference replace error')
                    orders.pop(oid);orders[new]=[side,new_price,new_qty]
                elif t=='D':orders.pop(oid)
                else:
                    reduction=number(p,19,4)
                    if reduction<=0 or reduction>qty:raise ValueError('Reference reduction error')
                    if reduction==qty:orders.pop(oid)
                    else:orders[oid][2]-=reduction
                mutations+=1
            # P/Q/B and administrative messages never alter visible orders.
            if seq in checks:
                levels={'B':defaultdict(int),'S':defaultdict(int)}
                for side,price,qty in orders.values():levels[side][price]+=qty
                top={'B':[[p,levels['B'][p]] for p in sorted(levels['B'],reverse=True)[:10]],
                     'S':[[p,levels['S'][p]] for p in sorted(levels['S'])[:10]]}
                digest=hashlib.sha256()
                for oid,(side,price,qty) in sorted(orders.items()):digest.update(struct.pack('>QcQQ',oid,side.encode(),price,qty))
                expected=checks[seq]
                if top!=expected['top'] or len(orders)!=expected['active_orders'] or digest.hexdigest()!=expected['order_sha256']:
                    raise ValueError(f'Independent reference differs at source sequence {seq}')
                checked+=1
    if checked!=len(checks):raise ValueError('Missing reference checkpoint')
    return {'status':'passed','mutations_replayed':mutations,'checkpoints_compared':checked,
            'scope':'Independent byte decoding/order updates for all selected messages; full order-state hashes and top ten at recorded checkpoints'}
