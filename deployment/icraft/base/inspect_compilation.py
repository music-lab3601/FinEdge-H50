from collections import Counter
import json
from deployment.icraft.base.common import ROOT, read

for kind in ('tf32','int8'):
    for stage in ('quantized','adapted','ZG'):
        net=read(ROOT/f'artifacts/compile/{kind}/tcn_d_{stage}.json')
        print(kind,stage,'ops',len(net['ops']),'params',net.get('params_bytes'),
              'names',dict(Counter(o['_type_key'].split('::')[-1] for o in net['ops'])))
        for op in net['ops'][:3]:
            print(json.dumps({k:op[k] for k in ('_type_key','op_id','name','inputs','outputs','target_dtype','layer_count') if k in op},ensure_ascii=False))
