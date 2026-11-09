"""Synthetic W3 checks. Passing these never certifies a real market day."""
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np

from software.data import itch_book_w3 as parser
from software.data import w3_download as download
from software.data import w3_samples as samples
from software.data import w3_engine
from software.data import w3
from tests.software import test_w2


class ParserTests(unittest.TestCase):
    def test_o_administrative_fields_and_chunk_boundary(self):
        payload=test_w2.raw('O', b'AAPL    Y'+(100).to_bytes(4,'big')+(200).to_bytes(4,'big')+
            (150).to_bytes(4,'big')+(1234567890123).to_bytes(8,'big')+(110).to_bytes(4,'big')+(190).to_bytes(4,'big'))
        decoded=parser.decode(payload)
        self.assertEqual(decoded['near_execution_time'],1234567890123)
        self.assertEqual(decoded['upper_collar'],190)
        self.assertFalse(parser.Book().apply(decoded))
        for chunk in [1,7,4096]:self.assertEqual(list(parser.iter_frames(io.BytesIO(test_w2.frame(payload)),chunk))[0][2],payload)
    def test_unknown_and_truncated(self):
        for body in [test_w2.frame(test_w2.raw('z',b'x')),test_w2.frame(test_w2.raw('S',b'O'))[:-1]]:
            with self.assertRaises(ValueError):list(parser.iter_frames(io.BytesIO(body),3))
    def test_full_engine_backward_compatible_and_output_tamper(self):
        # Run the existing independent synthetic lifecycle fixture through the new engine.
        with patch.object(test_w2,'w2',w3_engine):
            test_w2.PipelineTests('test_full_synthetic_day_and_output_check').test_full_synthetic_day_and_output_check()


class WindowTests(unittest.TestCase):
    def test_99_100_101_and_future_trim(self):
        self.assertEqual([len(samples.endpoints(0,n,1)) for n in [99,100,101,102]],[0,0,1,2])
        self.assertEqual(len(samples.endpoints(0,200,50)),51)
    def test_segments_and_roles_never_join(self):
        self.assertEqual(len(samples.endpoints(0,80,1))+len(samples.endpoints(80,160,1)),0)
        for end in samples.endpoints(50,250,20):
            self.assertGreaterEqual(end-99,50);self.assertLess(end+20,250)
    def test_invalid_horizon(self):
        for h in [None,0,-1,1.5,True]:
            with self.assertRaises(ValueError):samples.endpoints(0,200,h)
    def test_labels_and_threshold_ties(self):
        a=np.zeros(4,dtype=samples.DTYPE)
        a['features'][:,0]=[100,101,100,100];a['features'][:,2]=[100,101,100,100]
        np.testing.assert_equal(samples.labels(a,[0,1,2],1,0),[2,0,1])
        self.assertEqual(samples.labels(a,[0],1,.01)[0],1)
    def test_ring_order_reset_and_normalization(self):
        ring=samples.RingWindow();rows=np.arange(101*40).reshape(101,40)
        for i,row in enumerate(rows):
            actual=ring.push(row)
            if i<99:self.assertIsNone(actual)
            else:np.testing.assert_equal(actual,rows[i-99:i+1])
        ring.reset();self.assertIsNone(ring.push(rows[0]))
    def test_stats_units_zero_variance_chunk_equivalence(self):
        rows=np.tile(np.arange(40),(200,1)).astype(float);rows[:,0]=np.arange(200)+10000
        expected=samples.physical(rows)
        a=samples.fit_stats([rows[:71],rows[71:]])
        np.testing.assert_allclose(a['mean'],expected.mean(axis=0))
        self.assertAlmostEqual(a['std'][0],expected[:,0].std())
        self.assertEqual(a['std'][1],1)
    def test_lazy_loader_matches_cpu_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=np.zeros(150,dtype=samples.DTYPE);a['features']=np.arange(150*40).reshape(150,40)+10000
            path=Path(tmp)/'records.bin';a.tofile(path)
            norm=samples.fit_stats([a['features']])
            manifest={'ranges':[{'split':'train','end_start':99,'end_stop':145,'records_path':str(path)}],
                      'normalization':norm,'horizon':5,'threshold':0.0}
            ds=samples.Windows(manifest,'train');self.assertEqual(len(ds),46)
            for idx in [0,1,45]:
                x,y=ds[idx]
                expected=((samples.physical(a['features'][idx:idx+100])-norm['mean'])/norm['std']).T.astype(np.float32)
                np.testing.assert_array_equal(x,expected);self.assertEqual(x.shape,(40,100));self.assertEqual(y,2)
            with self.assertRaises(IndexError):ds[46]
            self.assertEqual(len(samples.Windows(manifest,'final_test')),0)
            ds.close()
    def test_unconfirmed_protocol_and_download_refused(self):
        with self.assertRaises(ValueError):samples.validate_protocol({**w3.draft(),'formal_training_protocol_frozen':False})
        with patch.object(w3,'draft',return_value={'approved_acquisition_dates':[]}):
            with self.assertRaises(ValueError):w3.require_acquisition('2025-12-08')
    def test_prepare_train_only_stats_and_heldout_labels_sealed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d=w3.draft()
            d.update(formal_training_protocol_frozen=True,shared_comparison_confirmed=True,
                     approved_acquisition_dates=d['dates'],label_horizon=1,label_threshold=0.0)
            runs={};train=[]
            for i,date in enumerate(d['dates']):
                folder=root/date;folder.mkdir()
                a=np.zeros(400,dtype=samples.DTYPE);a['features']=100
                base=np.repeat(np.tile([1000,1100,1100,1000],10),10)+(i*100000 if i>=2 else 0)
                a['features'][:,0::4]=(base+1)[:,None];a['features'][:,2::4]=(base-1)[:,None]
                a.tofile(folder/'records.bin')
                report={'source_sha256':'synthetic-'+date}
                download.save(folder/'report.json',report)
                download.save(folder/'output_hashes.json',{'records.bin':download.sha256(folder/'records.bin')})
                download.save(folder/'segments.json',[{'start_row':0,'end_row':399,'segment_id':1}])
                runs[date]=(folder,report,a,[{'start_row':0,'end_row':399,'segment_id':1}])
                if i<2:train.append(a['features'])
            with patch.object(samples,'ROOT',root),patch.object(samples,'draft',return_value=d),\
                 patch.object(samples,'config',return_value={'data_root':str(root)}),\
                 patch.object(samples,'get_run',side_effect=lambda date,recheck:runs[date]),\
                 patch.object(samples,'count_labels',wraps=samples.count_labels) as count:
                samples.prepare()
                self.assertEqual(count.call_count,3)  # two training dates plus tuning; no comparison/final labels.
                samples.check_prepared()
                checked=json.loads((root/'artifacts/w3_2026-10-03/prepared_check.json').read_text(encoding='utf-8'))
                self.assertEqual(checked['training_window_label_checks'],6)
                self.assertFalse(checked['comparison_and_final_labels_inspected'])
            meta=json.loads((root/'artifacts/w3_2026-10-03/prepared_latest.json').read_text(encoding='utf-8'))
            m=json.loads(Path(meta['manifest']).read_text(encoding='utf-8'))
            np.testing.assert_allclose(m['normalization']['mean'],samples.physical(np.concatenate(train)).mean(axis=0))
            self.assertEqual(m['sample_counts']['final_test'],300)
            self.assertFalse(m['comparison_and_final_labels_inspected'])
            # A re-signed malformed boundary must still fail the structural check.
            m['ranges'][0]['end_start']+=1
            download.save(Path(meta['manifest']),m)
            download.save(root/'artifacts/w3_2026-10-03/prepared_latest.json',{'manifest':meta['manifest'],'sha256':download.sha256(meta['manifest'])})
            with patch.object(samples,'ROOT',root),patch.object(samples,'draft',return_value=d):
                with self.assertRaisesRegex(ValueError,'boundary'):samples.check_prepared()

    def test_source_version_guard_rejects_changed_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp);ptr=f/'pointer.json';download.save(ptr,{'folder':str(f)})
            download.save(f/'report.json',{'status':'passed_full_day_engineering_audit','full_day':True,
                'trading_date':'2025-12-08','symbol':'AAPL','source_sha256':'raw'})
            download.save(f/'output_hashes.json',{'records.bin':'changed'})
            d={'symbol':'AAPL','source_versions':{'2025-12-08':{'raw_sha256':'raw','records_sha256':'approved'}}}
            with patch.object(samples,'pointer',return_value=ptr),patch.object(samples,'draft',return_value=d):
                with self.assertRaisesRegex(ValueError,'Data version'):samples.get_run('2025-12-08')


class DownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blob=gzip.compress(bytes(range(256))*30);cls.mode='ok';cls.requests=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def headers(self,code,size,cr=None):
                self.send_response(code);self.send_header('Content-Length',str(size));self.send_header('ETag','"fixture"')
                self.send_header('Last-Modified','Wed, 01 Jan 2025 00:00:00 GMT')
                if cr:self.send_header('Content-Range',cr)
                self.end_headers()
            def do_HEAD(self):self.headers(200,len(cls.blob))
            def do_GET(self):
                start,end=map(int,self.headers.get('Range').split('=')[1].split('-'));cls.requests.append(start)
                body=cls.blob[start:end+1]
                if cls.mode=='ignored':self.headers_response(200,len(cls.blob));self.wfile.write(cls.blob);return
                self.headers_response(206,len(body),f'bytes {start}-{end}/{len(cls.blob)}')
                self.wfile.write(body)
        # BaseHTTPRequestHandler reserves instance self.headers for request headers.
        Handler.headers_response=Handler.headers
        Handler.do_HEAD=lambda self:self.headers_response(200,len(cls.blob))
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.url=f'http://127.0.0.1:{cls.server.server_port}/synthetic.gz'
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def setUp(self):type(self).mode='ok';type(self).requests=[]
    def test_download_integrity_reuse_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(download,'BLOCK',97):
            path=Path(tmp)/'x.gz'
            a=download.fetch(self.url,path,len(self.blob),reserve=0,allow_local_test=True)
            self.assertFalse(a['publisher_checksum_verified']);self.assertEqual(path.read_bytes(),self.blob)
            self.assertEqual(download.fetch(self.url,path,len(self.blob),reserve=0,allow_local_test=True)['sha256'],a['sha256'])
            path.write_bytes(b'broken')
            with self.assertRaises(ValueError):download.fetch(self.url,path,len(self.blob),reserve=0,allow_local_test=True)
    def test_resume_known_identity(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(download,'BLOCK',97):
            path=Path(tmp)/'x.gz';meta=download.head(self.url);identity={k:meta[k] for k in ['url','bytes','etag','last_modified']}
            path.with_name('x.gz.part').write_bytes(self.blob[:97]);download.save(path.with_name('x.gz.download.json'),{'identity':identity})
            download.fetch(self.url,path,len(self.blob),reserve=0,allow_local_test=True)
            self.assertEqual(type(self).requests[0],97)
    def test_ignored_range_and_unknown_part(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'x.gz';type(self).mode='ignored'
            with self.assertRaises(ValueError):download.fetch(self.url,path,len(self.blob),reserve=0,allow_local_test=True)
            self.assertFalse(path.exists())
            path.with_name('x.gz.part').write_bytes(b'bad');path.with_name('x.gz.download.json').unlink()
            with self.assertRaises(ValueError):download.fetch(self.url,path,len(self.blob),reserve=0,allow_local_test=True)
    def test_bad_crc_never_published(self):
        original=type(self).blob
        try:
            type(self).blob=original[:-8]+b'\0'*8
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'x.gz'
                with self.assertRaises((OSError,EOFError)):download.fetch(self.url,path,len(self.blob),reserve=0,allow_local_test=True)
                self.assertFalse(path.exists())
        finally:type(self).blob=original


if __name__=='__main__':unittest.main(verbosity=2)
