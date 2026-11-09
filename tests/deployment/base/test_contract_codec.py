from pathlib import Path
import sys
import unittest
import numpy as np

from deployment.icraft.base.common import ROOT

sys.dont_write_bytecode=True
from deployment.icraft.base.codec import encode_external_fp32, decode_external_fp32, encode_internal_int8, decode_internal_int8, encode_for_full_graph, tf32_rne, STEP
from deployment.icraft.base.prepare import validate_footprints, endpoints
from deployment.icraft.base.common import read, sha


class CodecContractTests(unittest.TestCase):
    def test_external_bytes_roundtrip(self):
        x=np.arange(4000,dtype='<f4').reshape(1,40,100);x.flat[0]=-0.
        b=encode_external_fp32(x)
        self.assertEqual(len(b),16000)
        self.assertEqual(decode_external_fp32(b).tobytes(),x.tobytes())

    def test_full_graph_refuses_int8(self):
        with self.assertRaises(ValueError):encode_for_full_graph(np.zeros((1,40,100),'<f4'),'int8')

    def test_wrong_shape_and_precision(self):
        for x in (np.zeros((40,100),'<f4'),np.zeros((1,40,100),'<f8'),np.zeros((1,100,40),'<f4')):
            with self.assertRaises(ValueError):encode_external_fp32(x)

    def test_nonfinite(self):
        for value in (np.inf,-np.inf,np.nan):
            x=np.zeros((1,40,100),'<f4');x.flat[0]=value
            with self.assertRaises(ValueError):encode_internal_int8(x)

    def test_wrong_byte_lengths(self):
        with self.assertRaises(ValueError):decode_external_fp32(bytes(4000))
        with self.assertRaises(ValueError):decode_internal_int8(bytes(16000))

    def test_scale_guard(self):
        for scale in (0,-1,np.inf,np.nan):
            with self.assertRaises(ValueError):encode_internal_int8(np.zeros((1,40,100),'<f4'),scale)

    def test_signed_saturation(self):
        x=np.full((1,40,100),100,dtype='<f4');x[0,:,::2]=-100
        q=np.frombuffer(encode_internal_int8(x),dtype=np.int8).reshape(1,40,100)
        self.assertTrue(np.all(q[:,:,::2]==-128));self.assertTrue(np.all(q[:,:,1::2]==127))

    def test_tf32_halfway_even(self):
        bits=np.array([0x3f801000,0x3f803000,0xbf801000,0xbf803000],dtype='<u4')
        expected=np.array([0x3f800000,0x3f804000,0xbf800000,0xbf804000],dtype='<u4')
        np.testing.assert_array_equal(tf32_rne(bits.view('<f4')).view('<u4'),expected)

    @unittest.skipUnless((ROOT/'golden/inputs.npy').is_file(), 'External historical SDK/data receipts required')
    def test_sdk_goldens_exact(self):
        x=np.load(ROOT/'golden/inputs.npy')
        actual=b''.join(encode_internal_int8(v[None]) for v in x)
        self.assertEqual(actual,(ROOT/'golden/encoded.sqb').read_bytes())

    @unittest.skipUnless((ROOT/'golden/encoded.sqb').is_file(), 'External historical SDK/data receipts required')
    def test_sdk_dequantization(self):
        raw=(ROOT/'golden/encoded.sqb').read_bytes()
        count=len(raw)//4000
        x=np.concatenate([decode_internal_int8(raw[i*4000:(i+1)*4000]) for i in range(count)])
        sdk=np.fromfile(ROOT/'golden/sdk_dequantized.f32',dtype='<f4').reshape(count,40,100)
        np.testing.assert_array_equal(x,sdk)


class IsolationTests(unittest.TestCase):
    @unittest.skipUnless((ROOT/'data/identities.u64.npy').is_file(), 'External historical SDK/data receipts required')
    def test_real_full_footprints(self):
        ids=np.load(ROOT/'data/identities.u64.npy');a,b=endpoints()
        validate_footprints(ids,np.concatenate((a,b)))
        self.assertGreater(int(b.min())-99,int(a.max())+50)
        self.assertGreaterEqual(int(b.min()-a.max()-1),149)

    def test_segment_crossing_rejected(self):
        ids=np.zeros((151,4),dtype='<u8');ids[:,0]=1;ids[:,1]=np.arange(151);ids[:,2]=np.arange(1,152)
        ids[100:,0]=2
        with self.assertRaises(ValueError):validate_footprints(ids,[99])

    def test_profile_gap_rejected(self):
        ids=np.zeros((151,4),dtype='<u8');ids[:,0]=1;ids[:,1]=np.arange(151);ids[:,2]=np.arange(1,152)
        ids[30:,1]+=1
        with self.assertRaises(ValueError):validate_footprints(ids,[99])

    @unittest.skipUnless((ROOT/'contracts/evaluation_release.json').is_file(), 'External historical SDK/data receipts required')
    def test_frozen_evaluation_identity(self):
        release=read(ROOT/'contracts/evaluation_release.json')
        self.assertEqual(release['evaluation_inputs_sha256'],sha(ROOT/'data/evaluation.inputs.npy'))
        self.assertEqual(release['compiled_json_sha256'],sha(ROOT/'artifacts/compile/int8/tcn_d_ZG.json'))

    @unittest.skipUnless((ROOT/'input_contract.json').is_file(), 'External historical SDK/data receipts required')
    def test_input_contract_matches_compiler(self):
        c=read(ROOT/'input_contract.json');n=read(ROOT/'artifacts/compile/int8/tcn_d_ZG.json')
        self.assertEqual(c['input']['compiled_dtype'],n['ops'][0]['outputs'][0]['dtype'])
        self.assertEqual(c['input']['storage_bytes'],16000)
        self.assertFalse(c['direct_int8_input_supported_by_current_full_graph'])


if __name__=='__main__':unittest.main(verbosity=2)
