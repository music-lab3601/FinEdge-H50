import sys,unittest
from pathlib import Path
import numpy as np
from deployment.icraft.native.common3 import ROOT

from deployment.icraft.native.common3 import BASE, V2, PROJECT, read, sha
from deployment.icraft.native.codec3 import SCALE, encode, decode, encode_record, pack, unpack
from deployment.icraft.native.prepare_holdout3 import frozen_check

class NativeEntryTests(unittest.TestCase):
    @unittest.skipUnless((ROOT/'release_manifest.json').is_file(), 'External historical SDK/data receipts required')
    def test_frozen_identity_and_actual_abi(self):
        f=frozen_check();c=read(ROOT/'input_contract.json')
        self.assertEqual(c['base_contract_sha256'],'81ffd0d31b348281160ca2d8f0d94ec2f9b5cc511cfb19425727b9584928a73c')
        for m in f['models'].values():
            n=read(ROOT/m['json']);t=n['ops'][0]['outputs'][0]['dtype']
            self.assertEqual(t['shape'],[1,3,1,100,16]);self.assertEqual(t['layout'],'@layout(NCHWc16)')
            self.assertEqual(t['element_dtype']['storage_dtype'],'@sint(8)')
            self.assertEqual(sum(o.get('layer_count',0) for o in n['ops']),m['expected_layers_per_forward'])
            self.assertTrue(all(o['_type_key'].endswith(('Input','Output','HardOp')) for o in n['ops']))
            self.assertEqual([o for o in n['ops'] if o['_type_key'].endswith('HardOp') and not o['compile_target'].startswith('@zhuget')],[])

    @unittest.skipUnless((ROOT/'release_manifest.json').is_file(), 'External historical SDK/data receipts required')
    def test_64_goldens_match_actual_sdk(self):
        g=ROOT/'golden_v2';x=np.load(g/'normalized.npy')
        expected=(g/'sdk/sdk_encoded.bin').read_bytes()
        self.assertEqual(len(x),64);self.assertEqual(len(expected),64*4800)
        self.assertEqual(b''.join(encode(v) for v in x),expected)
        self.assertEqual((g/'encoded.bin').read_bytes(),expected)

    def test_signed_ties_and_saturation(self):
        # Use exact scale=1 coordinates through SDK source; chosen feature0
        # products below have exact half-integer quotients in float32.
        v=np.zeros((1,40,100),dtype='<f4')
        coords=np.array([-1000,-128,-127.5,-2.5,-1.5,-.5,.5,1.5,2.5,127.5,1000],dtype='<f4')
        v[0,0,:len(coords)]=coords*SCALE[0]
        self.assertEqual(unpack(encode(v))[0,0,:len(coords)].tolist(),[-128,-128,-127,-2,-1,0,1,2,3,127,127])

    def test_physical_offsets_and_800_zero_padding_bytes(self):
        q=((np.arange(4000).reshape(1,40,100)%255)-128).astype('i1');b=pack(q)
        self.assertEqual(len(b),4800)
        for f in range(40):
            for t in range(100):self.assertEqual(b[(f//16)*1600+t*16+f%16],int(q[0,f,t])%256)
        self.assertEqual(sum(1 for f in range(40,48) for t in range(100) if b[(f//16)*1600+t*16+f%16]==0),800)
        self.assertTrue(np.array_equal(unpack(b),q))

    def test_record_once_and_ring_wrap(self):
        rows=np.random.default_rng(90210).normal(size=(207,40)).astype('<f4')
        ring=np.zeros((100,40),dtype='i1')
        for i,row in enumerate(rows):
            ring[i%100]=np.frombuffer(encode_record(row),'i1')
            if i>=99:
                ordered=ring[(np.arange(i-99,i+1)%100)].T[None].copy()
                self.assertEqual(pack(ordered),encode(rows[i-99:i+1].T[None].copy()))

    def test_invalid_input_and_padding_rejected(self):
        good=np.zeros((1,40,100),dtype='<f4')
        for bad in [good.astype('f8'),good[0],np.full_like(good,np.nan),np.full_like(good,np.inf)]:
            with self.assertRaises(ValueError):encode(bad)
        for bad in [b'',bytes(4000),bytes(4801)]:
            with self.assertRaises(ValueError):decode(bad)
        bad=bytearray(4800);bad[2*1600+8]=1
        with self.assertRaises(ValueError):decode(bytes(bad))

    @unittest.skipUnless((ROOT/'release_manifest.json').is_file(), 'External historical SDK/data receipts required')
    def test_scales_fit_only_observed_data_and_invariant(self):
        x=np.load(V2/'data/inputs.npy');s=np.maximum(abs(x.min((0,2))),abs(x.max((0,2))))/np.float32(127)
        self.assertTrue(np.array_equal(s,SCALE));self.assertEqual(SCALE.shape,(40,));self.assertTrue((SCALE>0).all())
        self.assertEqual(read(ROOT/'input_contract.json')['numerical_quantization']['calibration_inputs_sha256'],sha(V2/'data/inputs.npy'))

    @unittest.skipUnless((ROOT/'release_manifest.json').is_file(), 'External historical SDK/data receipts required')
    def test_new_full_footprints_isolated_and_nonoverlapping(self):
        e=np.load(ROOT/'data/endpoints.npy');old=np.load(V2/'data/endpoints.npy')
        self.assertEqual(e.tolist(),np.linspace(250099,450210,1024,dtype=np.int64).tolist())
        self.assertGreater(int(e.min())-99,int(old.max())+50);self.assertGreaterEqual(int(np.diff(e).min()),150)
        ids=np.load(ROOT/'data/identities.npy')
        self.assertEqual(ids.shape,(1024,150,4))
        self.assertTrue((ids[:,:,0]==ids[:,0:1,0]).all());self.assertTrue((np.diff(ids[:,:,1].astype('i8'),axis=1)==1).all())

    @unittest.skipUnless((ROOT/'release_manifest.json').is_file(), 'External historical SDK/data receipts required')
    def test_fresh_normalization_independently_rebuilt(self):
        raw=np.load(ROOT/'data/raw_footprints.npy')[:,:100].astype('f8');norm=read(PROJECT/'vendor/ed4_model/normalization.json')
        mid=(raw[:,:,0]+raw[:,:,2])*.5
        raw[:,:,::2]=(raw[:,:,::2]-mid[:,:,None])/mid[:,:,None]
        expected=((raw-np.array(norm['mean']))/np.array(norm['std'])).astype('<f4').transpose(0,2,1)
        self.assertTrue(np.array_equal(expected,np.load(ROOT/'data/inputs.npy')))

    @unittest.skipUnless((ROOT/'release_manifest.json').is_file(), 'External historical SDK/data receipts required')
    def test_h50_label_uses_only_evaluation_future(self):
        raw=np.load(ROOT/'data/raw_footprints.npy');a=raw[:,99,0]+raw[:,99,2];b=raw[:,149,0]+raw[:,149,2]
        expected=np.where(b>a,2,np.where(b<a,0,1));self.assertTrue(np.array_equal(expected,np.load(ROOT/'data/labels.npy')))

    @unittest.skipUnless((ROOT/'release_manifest.json').is_file(), 'External historical SDK/data receipts required')
    def test_quality_counts_budget_and_no_fresh_tuning(self):
        q=read(ROOT/'quality_report.json');y=np.load(ROOT/'data/labels.npy');a=np.load(ROOT/'artifacts/fresh/original_fp32.npy')
        self.assertEqual(int((a.argmax(1)==y).sum()),470)
        b=np.load(ROOT/'artifacts/fresh/mixed_int8_entry/logits.npy');pred=b.argmax(1)
        self.assertEqual(int((pred==y).sum()),461)
        f1=[]
        for c in range(3):
            tp=np.count_nonzero((y==c)&(pred==c));den=np.count_nonzero(y==c)+np.count_nonzero(pred==c);f1.append(2*tp/den)
        self.assertAlmostEqual(float(np.mean(f1)),q['rows']['mixed_int8_entry']['macro_f1'],places=14)
        self.assertTrue(q['rows']['mixed_int8_entry']['quality_budget_passed']);frozen_check()

    @unittest.skipUnless((ROOT/'release_manifest.json').is_file(), 'External historical SDK/data receipts required')
    def test_parameter_and_bias_audit(self):
        for kind in ['mixed','tf32']:
            a=read(ROOT/f'artifacts/weight_audit/{kind}.json');self.assertEqual(a['status'],'passed');self.assertEqual(a['changed_tensors'],2)
            self.assertEqual(sorted(p['id'] for p in a['parameters'] if not p['bit_equal']),[6,20])
            self.assertTrue(all(p['bit_equal'] or p['formula_checked'] for p in a['parameters']))

    @unittest.skipUnless((ROOT/'release_manifest.json').is_file(), 'External historical SDK/data receipts required')
    def test_board_request_bound_to_bytes_outputs_and_layer_counts(self):
        r=read(ROOT/'board_handoff/request.json');self.assertEqual(r['total_forwards'],86)
        self.assertEqual(len((ROOT/r['inputs']).read_bytes()),32*4800)
        self.assertEqual(sha(ROOT/r['inputs']),r['inputs_sha256']);self.assertFalse(r['board_access_by_D'])
        for m in r['models']:
            self.assertEqual(np.fromfile(ROOT/m['expected'],'<f4').shape,(96,))
            self.assertEqual(sha(ROOT/m['expected']),m['expected_sha256'])
            a=np.load(ROOT/f"artifacts/fresh/{m['name']}/logits.npy")[:32]
            self.assertEqual(a.tobytes(),(ROOT/m['expected']).read_bytes())
            summary=read(ROOT/f"artifacts/fresh/{m['name']}/summary.json")
            self.assertEqual(summary['input_bytes'],4800);self.assertTrue(summary['direct_integer_write']);self.assertFalse(summary['runtime_fp32_input_adapter'])

    @unittest.skipUnless((ROOT/'release_manifest.json').is_file(), 'External historical SDK/data receipts required')
    def test_prior_v1_v2_releases_unchanged(self):
        for root,name in [(BASE,'candidate.json'),(V2,'release_manifest.json')]:
            manifest=read(root/name)
            for row in manifest['files']:
                p=(root/row['path']).resolve();self.assertTrue(p.is_relative_to(root));self.assertEqual(sha(p),row['sha256'],str(p))

if __name__=='__main__':unittest.main()
