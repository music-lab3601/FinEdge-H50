"""Use native channel-blocked input to avoid unsupported integer IO transforms."""
import copy,subprocess,time
from deployment.icraft.native.common3 import ROOT, SDK, read, save, env, sha

def main():
 out=ROOT/'artifacts/int8_physical';out.mkdir(exist_ok=False)
 source=ROOT/'artifacts/int8_folded/tcn_d_integer_folded_adapted.json';n=read(source)
 first=next(i for i,o in enumerate(n['ops']) if o['_type_key'].endswith('Conv2d'))
 decoded=copy.deepcopy(n['ops'][first]['inputs'][0]);entry=copy.deepcopy(decoded);entry['v_id']=0
 entry['dtype']['element_dtype']=dict(_type_key='icraft::xir::QuantizedType',expressed_dtype='@fp(32)',storage_dtype='@sint(8)',scale='@scales([axis=-1]48*1)',zero_points=[])
 remaining=n['ops'][first:]
 produced={v['v_id'] for o in remaining for v in o['outputs']}
 missing=sorted({v['v_id'] for o in remaining for v in o['inputs'] if v['_type_key'].endswith('Value')} - produced)
 assert missing==[decoded['v_id']],missing
 inp=copy.deepcopy(n['ops'][0]);inp['outputs']=[entry]
 cast=dict(_type_key='icraft::xir::Cast',compile_target='@zhuget(330)',inputs=[entry],outputs=[decoded],op_id=9000,name='decode',tags={},target_dtype=copy.deepcopy(decoded['dtype']))
 n['ops']=[inp,cast]+remaining;n['name']='d3p';model=out/'d3p_adapted.json';save(model,n)
 import shutil;shutil.copy2(source.with_suffix('.raw'),model.with_suffix('.raw'))
 cmd=[str(SDK/'bin/icraft.exe'),'generate','--json',str(model),'--raw',str(model.with_suffix('.raw')),'--jr_path',str(out)]
 t=time.perf_counter()
 with (out/'generate.log').open('wb') as f:r=subprocess.run(cmd,cwd=out,env=env(),stdout=f,stderr=subprocess.STDOUT)
 save(out/'result.json',dict(exit_code=r.returncode,seconds=time.perf_counter()-t,command=cmd,model_sha256=sha(model),raw_sha256=sha(model.with_suffix('.raw'))))
 print(r.returncode)
if __name__=='__main__':main()
