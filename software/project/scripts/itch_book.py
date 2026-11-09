"""ITCH5 (2019 profile), deterministic visible book. No labels or training.

Prices remain unsigned Price(4) integers; quantities are shares.
Only A/F/E/C/X/D/U mutate visible orders. Unknown semantics fail closed.
"""
from bisect import bisect_left, insort
from collections import defaultdict
import hashlib
import struct

VERSION='W2-audit-1.0'
LENGTHS=dict(S=12,R=39,H=25,Y=20,L=26,V=35,W=12,K=28,J=35,h=21,A=36,F=40,E=31,C=36,X=23,D=19,U=35,P=44,Q=40,B=19,I=50,N=20)
LENGTH_BYTES={ord(k):v for k,v in LENGTHS.items()}
MUTATIONS=set('AFECXDU')
# Field definitions are offset, length, kind; header fields are shared.
FIELDS={
 'S':[('event',11,1,'a')],
 'R':[('stock',11,8,'a'),('market_category',19,1,'a'),('financial_status',20,1,'a'),('round_lot_size',21,4,'i'),('round_lots_only',25,1,'a'),('issue_classification',26,1,'a'),('issue_subtype',27,2,'a'),('authenticity',29,1,'a'),('short_sale_threshold',30,1,'a'),('ipo_flag',31,1,'a'),('luld_reference_tier',32,1,'a'),('etp_flag',33,1,'a'),('etp_leverage_factor',34,4,'i'),('inverse_indicator',38,1,'a')],
 'H':[('stock',11,8,'a'),('trading_state',19,1,'a'),('reserved',20,1,'a'),('reason',21,4,'a')],
 'Y':[('stock',11,8,'a'),('reg_sho_action',19,1,'a')],
 'L':[('mpid',11,4,'a'),('stock',15,8,'a'),('primary_market_maker',23,1,'a'),('market_maker_mode',24,1,'a'),('market_participant_state',25,1,'a')],
 'V':[('level_1',11,8,'i'),('level_2',19,8,'i'),('level_3',27,8,'i')],
 'W':[('breached_level',11,1,'a')],
 'K':[('stock',11,8,'a'),('release_time',19,4,'i'),('release_qualifier',23,1,'a'),('ipo_price',24,4,'i')],
 'J':[('stock',11,8,'a'),('reference_price',19,4,'i'),('upper_price',23,4,'i'),('lower_price',27,4,'i'),('extensions',31,4,'i')],
 'h':[('stock',11,8,'a'),('market_code',19,1,'a'),('halt_action',20,1,'a')],
 'A':[('order_id',11,8,'i'),('side',19,1,'a'),('shares',20,4,'i'),('stock',24,8,'a'),('price',32,4,'i')],
 'E':[('order_id',11,8,'i'),('shares',19,4,'i'),('match_number',23,8,'i')],
 'X':[('order_id',11,8,'i'),('shares',19,4,'i')],
 'D':[('order_id',11,8,'i')],
 'U':[('order_id',11,8,'i'),('new_order_id',19,8,'i'),('shares',27,4,'i'),('price',31,4,'i')],
 'Q':[('shares',11,8,'i'),('stock',19,8,'a'),('price',27,4,'i'),('match_number',31,8,'i'),('cross_type',39,1,'a')],
 'B':[('match_number',11,8,'i')],
 'I':[('paired_shares',11,8,'i'),('imbalance_shares',19,8,'i'),('imbalance_direction',27,1,'a'),('stock',28,8,'a'),('far_price',36,4,'i'),('near_price',40,4,'i'),('reference_price',44,4,'i'),('cross_type',48,1,'a'),('price_variation',49,1,'a')],
 'N':[('stock',11,8,'a'),('interest',19,1,'a')]
}
FIELDS['F']=FIELDS['A']+[('mpid',36,4,'a')]
FIELDS['C']=FIELDS['E']+[('printable',31,1,'a'),('execution_price',32,4,'i')]
FIELDS['P']=FIELDS['A']+[('match_number',36,8,'i')]

def decode(payload):
    if not payload:raise ValueError('Empty message')
    typ=chr(payload[0])
    if LENGTHS.get(typ)!=len(payload):raise ValueError(f'Unknown message/length {typ}/{len(payload)}')
    locate,tracking=struct.unpack_from('>HH',payload,1)
    timestamp=int.from_bytes(payload[5:11],'big')
    if timestamp>=86400*10**9:raise ValueError('Timestamp outside day')
    event={'type':typ,'stock_locate':locate,'tracking_number':tracking,'timestamp_ns':timestamp}
    for name,offset,length,kind in FIELDS[typ]:
        value=payload[offset:offset+length]
        event[name]=int.from_bytes(value,'big') if kind=='i' else value.decode('ascii').rstrip(' ')
    return event

def iter_frames(stream, chunk_size=4*1024**2):
    """2-byte big-endian length-prefixed file framing, not a network sequence."""
    tail=b'';base=0;sequence=0
    while True:
        chunk=stream.read(chunk_size)
        if not chunk:
            if tail:raise ValueError(f'Truncated frame at uncompressed offset {base}')
            return
        data=tail+chunk;position=0;total=len(data)
        while position+2<=total:
            length=(data[position]<<8)|data[position+1]
            if length<11 or length>50:raise ValueError(f'Invalid frame length {length} at {base+position}')
            end=position+2+length
            if end>total:break
            payload=data[position+2:end]
            if LENGTH_BYTES.get(payload[0])!=length:raise ValueError(f'Unknown type/length at {base+position}')
            sequence+=1
            yield sequence,base+position,payload
            position=end
        tail=data[position:];base+=position

class Book:
    def __init__(self):
        self.orders={};self.levels={'B':{},'S':{}};self.prices={'B':[],'S':[]}
    def _level(self,side,price,delta):
        levels=self.levels[side];old=levels.get(price,0);new=old+delta
        if new<0:raise ValueError('Negative aggregated shares')
        if new:
            if not old:insort(self.prices[side],price)
            levels[price]=new
        elif old:
            del levels[price];self.prices[side].pop(bisect_left(self.prices[side],price))
    def add(self,oid,side,price,shares):
        if oid in self.orders:raise ValueError(f'Duplicate active order {oid}')
        if side not in ('B','S') or shares<=0 or price<=0:raise ValueError('Invalid order side/price/shares')
        self.orders[oid]=(side,price,shares);self._level(side,price,shares)
    def apply(self,event):
        typ=event['type']
        if typ not in MUTATIONS:return False
        oid=event['order_id']
        if typ in 'AF':
            self.add(oid,event['side'],event['price'],event['shares']);return True
        if oid not in self.orders:raise ValueError(f'Unknown order reference {typ}/{oid}')
        side,price,remaining=self.orders[oid]
        if typ=='U':
            new_id=event['new_order_id']
            if new_id in self.orders or event['shares']<=0 or event['price']<=0:
                raise ValueError('Invalid replacement ID/price/shares')
            del self.orders[oid];self._level(side,price,-remaining)
            self.add(new_id,side,event['price'],event['shares'])
        else:
            amount=remaining if typ=='D' else event['shares']
            if amount<=0 or amount>remaining:raise ValueError(f'Invalid reduction {typ}/{oid}/{amount}/{remaining}')
            self._level(side,price,-amount)
            if amount==remaining:del self.orders[oid]
            else:self.orders[oid]=(side,price,remaining-amount)
        return True
    def top(self):
        return {'B':[[p,self.levels['B'][p]] for p in reversed(self.prices['B'][-10:])],
                'S':[[p,self.levels['S'][p]] for p in self.prices['S'][:10]]}
    def validate(self):
        reference={'B':defaultdict(int),'S':defaultdict(int)}
        for side,price,shares in self.orders.values():reference[side][price]+=shares
        if any(dict(reference[s])!=self.levels[s] or sorted(reference[s])!=self.prices[s] for s in 'BS'):
            raise ValueError('Order-to-price-level invariant violated')

def order_digest(orders):
    h=hashlib.sha256()
    for oid,(side,price,shares) in sorted(orders.items()):
        h.update(struct.pack('>QcQQ',oid,side.encode(),price,shares))
    return h.hexdigest()

# Diagnostic flags, not a frozen training filter. Invalid rows are kept, not dropped.
FLAGS={1:'outside_regular_system_phase',2:'fewer_than_ten_levels',4:'locked_book',8:'crossed_book',16:'stock_not_trading_or_operational_halt',32:'timestamp_regression'}
ROW=struct.Struct('<QQIIII40Q')
def snapshot(book,regular,tradable,time_regression=False):
    top=book.top();flags=0
    if not regular:flags|=1
    if len(top['B'])<10 or len(top['S'])<10:flags|=2
    if not tradable:flags|=16
    if time_regression:flags|=32
    if top['B'] and top['S']:
        if top['B'][0][0]==top['S'][0][0]:flags|=4
        if top['B'][0][0]>top['S'][0][0]:flags|=8
    features=[]
    for level in range(10):
        features+=top['S'][level] if level<len(top['S']) else [0,0]
        features+=top['B'][level] if level<len(top['B']) else [0,0]
    return flags,features
