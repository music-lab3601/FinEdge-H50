"""Read-only, stdlib-only G intake verifier; never opens device or SDK."""
import hashlib,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    release=read(ROOT/'release_manifest.json')
    for r in release['files']:
        p=(ROOT/r['path']).resolve()
        if not p.is_relative_to(ROOT) or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('Payload changed: '+r['path'])
    c=read(ROOT/'input_contract.json');f=read(ROOT/'contracts/frozen_candidate.json');q=read(ROOT/'quality_report.json')
    assert sha(ROOT/'input_contract.json')==f['input_contract_sha256']
    assert sha(ROOT/'scripts/codec3.py')==f['codec_sha256']
    import struct
    public=c['numerical_quantization']['scale']
    carrier=read(ROOT/'contracts/quantization_int8_folded.json')['scale']
    assert public==carrier and struct.pack('<40f',*public)==(ROOT/'artifacts/int8_folded/scales.f32').read_bytes()
    assert c['storage_bytes']==4800 and c['external_dtype']=='signed_int8'
    counts={}
    for name,m in f['models'].items():
        assert sha(ROOT/m['json'])==m['json_sha256'] and sha(ROOT/m['raw'])==m['raw_sha256']
        n=read(ROOT/m['json']);t=n['ops'][0]['outputs'][0]['dtype']
        assert t['shape']==[1,3,1,100,16] and t['element_dtype']['storage_dtype']=='@sint(8)'
        counts[name]=sum(o.get('layer_count',0) for o in n['ops']);assert counts[name]==m['expected_layers_per_forward']
    print(json.dumps(dict(status='passed',files_verified=len(release['files']),external_bytes=4800,logical_bytes=4000,
        input_bytes_reduction_percent=70.0,completion_counts=counts,offline_quality_passed=q['rows']['mixed_int8_entry']['quality_budget_passed'],
        board_tested=False,npu_speedup=None,production_default_changed=False),indent=2))
if __name__=='__main__':main()
