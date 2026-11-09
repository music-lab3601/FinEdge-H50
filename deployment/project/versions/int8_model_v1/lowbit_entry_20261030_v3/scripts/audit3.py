import subprocess
from common3 import ROOT,BASE,V2,read,save,sha,env

def main():
    out=ROOT/'artifacts/weight_audit';out.mkdir(exist_ok=False)
    pairs=[('mixed',V2/'artifacts/mixed_tf32_compat/tcn_d_quantized.json',ROOT/'artifacts/int8_folded/tcn_d_integer_folded_quantized.json'),
           ('tf32',BASE/'artifacts/compile/tf32/tcn_d_quantized.json',ROOT/'artifacts/int8_entry_tf32_body/d3t_quantized.json')]
    for name,a,b in pairs:
        cmd=[str(ROOT/'build/Release/d_audit_weights.exe'),str(a),str(a.with_suffix('.raw')),str(b),str(b.with_suffix('.raw')),str(ROOT/'artifacts/int8_folded/scales.f32')]
        r=subprocess.run(cmd,cwd=out,env=env(),capture_output=True,text=True,encoding='utf-8')
        save(out/(name+'.command.json'),dict(command=cmd,exit_code=r.returncode,stderr=r.stderr))
        if r.returncode:raise RuntimeError('Weight audit failed '+r.stderr)
        import json;save(out/(name+'.json'),json.loads(r.stdout))
    # Correct a wording error (loss vs remaining budget margin); no numeric change.
    q=read(ROOT/'quality_report.json');q['limitations'][1]='9/1024 fewer correct than original; narrow remaining margin against 1pp budget';save(ROOT/'quality_report.json',q)
    print('Both parameter streams independently audited')

if __name__=='__main__':main()
