from pathlib import Path
import hashlib,json,sys,unittest,importlib.util
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;PROJECT=BASE.parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from codec_external import encode,decode,encode_as_int8

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

class PrecisionImprovementTests(unittest.TestCase):
    def test_v1_payloads_unchanged(self):
        for item in read(BASE/'candidate.json')['files']:
            self.assertEqual(sha(BASE/item['path']),item['sha256'],item['path'])

    def test_frozen_graph_identity(self):
        frozen=read(ROOT/'contracts/frozen_candidate.json')
        self.assertEqual(sha(ROOT/frozen['model_path']),frozen['model_sha256'])
        self.assertEqual(sha(ROOT/frozen['model_path'].replace('.json','.raw')),frozen['params_sha256'])
        self.assertEqual(read(ROOT/'data/manifest.json')['candidate_frozen_sha256'],sha(ROOT/'contracts/frozen_candidate.json'))

    def test_unused_nonoverlapping_full_footprints(self):
        ends=np.load(ROOT/'data/endpoints.npy');ids=np.load(ROOT/'data/identities.npy')
        self.assertEqual(len(ends),512);self.assertGreaterEqual(int(ends.min())-99,50000)
        self.assertTrue(np.all(ends[1:]-99>ends[:-1]+50))
        self.assertTrue(np.all(ids[:,:,0]==ids[:,0:1,0]))
        self.assertTrue(np.all(np.diff(ids[:,:,1].astype(np.int64),axis=1)==1))

    def test_exact_integer_h50_labels(self):
        raw=np.load(ROOT/'data/raw_footprints.npy');labels=np.load(ROOT/'data/labels.npy')
        expected=[]
        for row in raw:
            a=int(row[99,0])+int(row[99,2]);b=int(row[149,0])+int(row[149,2])
            expected.append(2 if b>a else 0 if b<a else 1)
        np.testing.assert_array_equal(labels,np.array(expected))

    def test_frozen_original_preprocess(self):
        package=PROJECT/'vendor/ed4_model'
        spec=importlib.util.spec_from_file_location('D_v2_delivered_runtime',package/'runtime.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        raw=np.load(ROOT/'data/raw_footprints.npy')[:32,:100]
        actual=module.preprocess_raw(raw,read(package/'normalization.json'))
        np.testing.assert_array_equal(actual,np.load(ROOT/'data/inputs.npy')[:32])

    def test_actual_mixed_types(self):
        c=read(ROOT/'input_contract.json')
        self.assertEqual(c['arithmetic_precision_counts'],{'@tf(32)':6,'@sint(8)':9})
        self.assertEqual(c['expected_layers_per_forward'],33)
        self.assertFalse(c['old_internal_value246_codec_applicable'])

    def test_external_type_matches_generated_model(self):
        c=read(ROOT/'input_contract.json');net=read(ROOT/c['model_json'])
        self.assertEqual(c['input']['compiled_dtype'],net['ops'][0]['outputs'][0]['dtype'])
        self.assertEqual(c['input']['storage_bytes'],16000)
        self.assertFalse(c['full_window_direct_int8_input_supported'])

    def test_fp32_codec_and_rejection(self):
        x=np.load(ROOT/'data/inputs.npy')[0:1]
        self.assertEqual(decode(encode(x)).tobytes(),x.tobytes())
        with self.assertRaises(ValueError):encode_as_int8(x)
        with self.assertRaises(ValueError):encode(x.astype(np.float64))
        with self.assertRaises(ValueError):decode(bytes(4000))

    def test_saved_counts_and_budget(self):
        q=read(ROOT/'quality_report.json');labels=np.load(ROOT/'data/labels.npy')
        for name in ('ordinary_int8','mixed','tf32'):
            p=np.load(ROOT/f'artifacts/fresh/{name}/logits.npy').argmax(1)
            self.assertEqual(int((p==labels).sum()),q['rows'][name]['correct'])
        self.assertTrue(q['rows']['mixed']['quality_budget_passed'])
        self.assertFalse(q['rows']['ordinary_int8']['quality_budget_passed'])
        self.assertIsNone(q['npu_speedup']);self.assertFalse(q['board_tested'])

    def test_failed_fp16_attempt_preserved(self):
        for p in ('artifacts/mixed/stage_results.json','artifacts/codegen_conservative/result.json'):
            self.assertEqual(read(ROOT/p)['status'],'failed')
        self.assertEqual(read(ROOT/'artifacts/mixed/stage_results.json')['stages'][-1]['exit_code'],3221226505)
        self.assertEqual(read(ROOT/'contracts/implementation_amendment.json')['precision_configurations_attempted'],2)

if __name__=='__main__':unittest.main(verbosity=2)
