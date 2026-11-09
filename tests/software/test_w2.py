"""Hand-calculable synthetic fixtures. No financial prediction claims."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

from software.data.itch_book import Book, decode, iter_frames, snapshot
from software.data import w2

def raw(t,body,ts=123456789,locate=14):
    return t.encode()+struct.pack('>HH',locate,17)+ts.to_bytes(6,'big')+body
def add(oid,side,price,qty=100,t='A',ts=123456789):
    return raw(t,struct.pack('>QcI8sI',oid,side.encode(),qty,b'AAPL    ',price)+(b'TEST' if t=='F' else b''),ts)
def frame(payload):return len(payload).to_bytes(2,'big')+payload
def event(t,**fields):return {'type':t,**fields}

class DecodeTests(unittest.TestCase):
    def test_literal_big_endian_and_large_id(self):
        p=bytes.fromhex('41 000e 0011 010203040506 0020000000000001 42 00000064 4141504c20202020 0017a120')
        d=decode(p)
        self.assertEqual(d['order_id'],9007199254740993)
        self.assertEqual(d['timestamp_ns'],0x010203040506)
        self.assertEqual((d['side'],d['stock'],d['shares'],d['price']),('B','AAPL',100,1548576))
    def test_f_and_c_fields(self):
        self.assertEqual(decode(add(1,'S',100,123,'F'))['mpid'],'TEST')
        d=decode(raw('C',struct.pack('>QIQcI',1,7,99,b'N',200)))
        self.assertEqual((d['shares'],d['match_number'],d['printable'],d['execution_price']),(7,99,'N',200))
    def test_u(self):
        d=decode(raw('U',struct.pack('>QQII',1,2,35,120)))
        self.assertEqual((d['order_id'],d['new_order_id'],d['shares'],d['price']),(1,2,35,120))
    def test_l_y(self):
        d=decode(raw('L',b'ABCD'+b'AAPL    '+b'YNA'))
        self.assertEqual((d['mpid'],d['stock'],d['primary_market_maker'],d['market_participant_state']),('ABCD','AAPL','Y','A'))
        self.assertEqual(decode(raw('Y',b'AAPL    0'))['reg_sho_action'],'0')
    def test_timestamp_outside_day(self):
        with self.assertRaises(ValueError):decode(raw('S',b'O',86400*10**9,0))
    def test_unknown_and_bad_length(self):
        for p in [b'',raw('z',b'O'),raw('A',b'O')]:
            with self.assertRaises(ValueError):decode(p)
    def test_frames_split_across_chunks(self):
        encoded=frame(add(1,'B',100))+frame(raw('D',struct.pack('>Q',1)))
        for chunk in [1,2,3,7,41,1000]:
            frames=list(iter_frames(io.BytesIO(encoded),chunk))
            self.assertEqual([x[0] for x in frames],[1,2]);self.assertEqual(frames[1][1],38)
    def test_truncated_frames(self):
        for encoded in [b'\x00',b'\x00\x24A',frame(add(1,'B',100))[:-1],b'\x00\x01z']:
            with self.assertRaises(ValueError):list(iter_frames(io.BytesIO(encoded),3))

class BookTests(unittest.TestCase):
    def setUp(self):self.book=Book()
    def apply(self,p):self.book.apply(decode(p));self.book.validate()
    def test_hand_computed_all_mutations(self):
        self.apply(add(1,'B',100,100));self.apply(add(2,'B',100,50,'F'))
        self.apply(raw('E',struct.pack('>QIQ',1,20,555)))
        self.apply(raw('C',struct.pack('>QIQcI',1,10,556,b'N',999)))
        self.assertEqual(self.book.levels['B'],{100:120})
        self.apply(raw('X',struct.pack('>QI',2,5)))
        self.apply(raw('U',struct.pack('>QQII',1,3,30,101)))
        self.assertEqual(self.book.levels['B'],{100:45,101:30})
        self.apply(raw('D',struct.pack('>Q',2)))
        self.apply(raw('E',struct.pack('>QIQ',3,30,557)))
        self.assertEqual(self.book.orders,{})
    def test_deeper_orders_survive_and_fill_top(self):
        for i in range(12):self.apply(add(i+1,'B',100+i))
        self.assertEqual(self.book.top()['B'][-1],[102,100])
        self.apply(raw('D',struct.pack('>Q',12)))
        self.assertEqual(self.book.top()['B'][-1],[101,100])
    def test_trade_reports_never_change_book(self):
        self.apply(add(1,'B',100));before=dict(self.book.orders)
        for t in 'PQB':self.assertFalse(self.book.apply(event(t,order_id=1,shares=999)))
        self.assertEqual(self.book.orders,before)
    def test_bad_events_preserve_book(self):
        self.apply(add(1,'B',100));before=dict(self.book.orders)
        for e in [event('A',order_id=1,side='S',shares=1,price=101),event('D',order_id=999),
                  event('X',order_id=1,shares=101),event('X',order_id=1,shares=0),
                  event('U',order_id=1,new_order_id=1,shares=10,price=101)]:
            with self.assertRaises(ValueError):self.book.apply(e)
            self.assertEqual(self.book.orders,before)
    def test_missing_depth_is_flagged(self):
        self.apply(add(1,'B',100))
        flags,features=snapshot(self.book,True,True)
        self.assertTrue(flags&2);self.assertEqual(len(features),40)
    def test_sorted_40_features_and_state_flags(self):
        for i in range(10):
            self.apply(add(i+1,'B',100-i,10+i));self.apply(add(i+101,'S',101+i,20+i))
        flags,f=snapshot(self.book,True,True)
        self.assertEqual(flags,0);self.assertEqual(f[:4],[101,20,100,10]);self.assertEqual(f[-4:],[110,29,91,19])
        flags,_=snapshot(self.book,False,False,True);self.assertEqual(flags,1|16|32)
    def test_locked_crossed_flags(self):
        self.apply(add(1,'B',100));self.apply(add(2,'S',100))
        self.assertTrue(snapshot(self.book,True,True)[0]&4)
        self.apply(add(3,'B',101));self.assertTrue(snapshot(self.book,True,True)[0]&8)

class PipelineTests(unittest.TestCase):
    def test_full_synthetic_day_and_output_check(self):
        messages=[raw('S',b'O',1,0)]
        r=bytearray(39);r[0]=ord('R');r[1:3]=(14).to_bytes(2,'big');r[5:11]=(2).to_bytes(6,'big')
        r[11:19]=b'AAPL    ';r[19:21]=b'QN';r[21:25]=(100).to_bytes(4,'big');r[25:34]=b'NCZ  PNN1';r[34:38]=b'\0'*4;r[38]=ord('N')
        messages += [bytes(r),raw('H',b'AAPL    T     ',3),raw('S',b'S',4,0)]
        for i in range(10):messages.extend([add(i+1,'B',100-i,1000000,ts=5),add(i+101,'S',101+i,100,ts=5)])
        messages.append(raw('S',b'Q',6,0))
        for i in range(1010):messages.append(raw('X',struct.pack('>QI',1,1),7+i))
        # A stock-state transition forces a new segment without clearing the book.
        messages.extend([raw('H',b'AAPL    H     ',2000),raw('H',b'AAPL    T     ',2001),raw('X',struct.pack('>QI',1,1),2002)])
        messages.extend([raw('S',b'M',3000,0),raw('S',b'E',4000,0),raw('S',b'C',5000,0)])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'synthetic.gz'
            with gzip.open(source,'wb') as f:f.write(b''.join(map(frame,messages)))
            cfg=json.loads((Path(__file__).resolve().parents[2]/'config/research/w2_audit.json').read_text(encoding='utf-8'))
            cfg.update(expected_sample_messages=len(messages),expected_sample_target_messages=len(messages)-6)
            data={'sample':{'reference_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'reference_sha256_source':'SYNTHETIC TEST'}}
            with patch.object(w2,'preflight',return_value=(source,data,cfg)),patch.object(w2,'REPORTS',root/'reports'):
                folder=w2.execute('full',root/'outputs');result=w2.check(folder)
                self.assertEqual(result['valid_rows'],1011);self.assertEqual(result['segments'],2)
                report=json.loads((folder/'report.json').read_text())
                self.assertEqual(report['reference']['mutations_replayed'],1031)
                self.assertFalse(report['training_ready'])
                with (folder/'records.bin').open('ab') as f:f.write(b'x')
                with self.assertRaises(ValueError):w2.check(folder)

if __name__=='__main__':unittest.main(verbosity=2)
