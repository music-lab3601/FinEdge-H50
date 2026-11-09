"""W3 ITCH5 profile adds the documented 48-byte O administrative message.
W2 source stays unchanged. Unknown messages still stop the audit.
"""
from software.data import itch_book as old
from software.data.itch_book import Book, FLAGS, ROW, MUTATIONS, snapshot, order_digest
VERSION = 'W3-audit-1.0'
LENGTHS = {**old.LENGTHS, 'O':48}
FIELDS_O = [('stock',11,8,'a'), ('open_eligibility',19,1,'a'),
            ('min_price',20,4,'i'), ('max_price',24,4,'i'), ('near_execution_price',28,4,'i'),
            ('near_execution_time',32,8,'i'), ('lower_collar',40,4,'i'), ('upper_collar',44,4,'i')]


def decode(payload):
    if not payload or payload[0] != ord('O'): return old.decode(payload)
    if len(payload) != 48: raise ValueError('O length must be 48')
    ts = int.from_bytes(payload[5:11], 'big')
    if ts >= 86400*10**9: raise ValueError('Timestamp outside day')
    result = dict(type='O', stock_locate=int.from_bytes(payload[1:3], 'big'),
                  tracking_number=int.from_bytes(payload[3:5], 'big'), timestamp_ns=ts)
    for name, start, size, kind in FIELDS_O:
        value = payload[start:start+size]
        result[name] = int.from_bytes(value, 'big') if kind == 'i' else value.decode('ascii').rstrip(' ')
    return result


def iter_frames(stream, chunk_size=4*1024**2):
    tail=b''; base=0; sequence=0
    lengths={ord(k):v for k,v in LENGTHS.items()}
    while True:
        block=stream.read(chunk_size)
        if not block:
            if tail: raise ValueError(f'Truncated frame at {base}')
            return
        data=tail+block; pos=0
        while pos+2<=len(data):
            size=(data[pos]<<8)|data[pos+1]
            if not 11<=size<=50: raise ValueError(f'Invalid length at {base+pos}')
            end=pos+2+size
            if end>len(data): break
            payload=data[pos+2:end]
            if lengths.get(payload[0]) != size: raise ValueError(f'Unknown type/length at {base+pos}')
            sequence+=1
            yield sequence,base+pos,payload
            pos=end
        tail=data[pos:]; base+=pos
