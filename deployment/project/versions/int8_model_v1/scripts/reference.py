"""Original FP32 task-quality reference; no training and no board access."""
import time
import importlib.util
import numpy as np
import torch
import onnxruntime as ort
from common import ROOT, PROJECT, PACKAGE, read, save, sha


def summary(predictions, labels):
    predictions=np.asarray(predictions,dtype=np.int64);labels=np.asarray(labels,dtype=np.int64)
    matrix=np.zeros((3,3),np.int64)
    np.add.at(matrix,(labels,predictions),1)
    tp=np.diag(matrix).astype(float); actual=matrix.sum(1); selected=matrix.sum(0)
    f1=np.divide(2*tp,actual+selected,out=np.zeros(3),where=(actual+selected)>0)
    recall=np.divide(tp,actual,out=np.zeros(3),where=actual>0)
    return dict(samples=len(labels),accuracy=float(np.mean(predictions==labels)),macro_f1=float(f1.mean()),
        class_recall=recall.tolist(),class_f1=f1.tolist(),label_counts=actual.tolist(),
        confusion_actual_rows_predicted_columns=matrix.tolist())


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    model=torch.jit.load(str(PACKAGE/'model_cpu.ts'),map_location='cpu').eval()
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    sessions={name:ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider']) for name,path in (
        ('original_onnx',PROJECT/'artifacts/export/tcn_ed4.onnx'),
        ('candidate_onnx',ROOT/'model/tcn_polyphase.onnx'))}
    out=ROOT/'artifacts/reference';out.mkdir(exist_ok=True)
    if (out/'report.json').exists():raise ValueError('Reference already completed; do not replace evidence')
    golden=np.fromfile(PACKAGE/'golden/inputs.f32',dtype='<f4').reshape(32,40,100)
    expected=np.fromfile(PACKAGE/'golden/script_batch32.f32',dtype='<f4').reshape(32,3)
    report=dict(board_accessed=False,training=False,onnx_rtol=1e-4,onnx_atol=1e-5,
                model_sha256=sha(PACKAGE/'model_cpu.ts'),graph_sha256=sha(ROOT/'model/tcn_polyphase.onnx'),groups={})
    spec=importlib.util.spec_from_file_location('D_delivered_runtime',PACKAGE/'runtime.py')
    runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
    raw=np.load(ROOT/'data/raw.u64.npy');rows=np.load(ROOT/'data/rows.f32.npy')
    ends=np.load(ROOT/'data/calibration.endpoints.npy')[:32]
    original_pre=runtime.preprocess_raw(np.stack([raw[e-99:e+1] for e in ends]),read(PACKAGE/'normalization.json'))
    if not np.array_equal(original_pre,np.stack([rows[e-99:e+1].T for e in ends])):
        raise ValueError('Delivered preprocess_raw mismatch')
    report['delivered_preprocess_raw_32_windows_bit_equal']=True
    with torch.inference_mode():
        for group,inputs in [('golden',golden),('calibration',np.load(ROOT/'data/calibration.inputs.npy')),
                             ('evaluation',np.load(ROOT/'data/evaluation.inputs.npy'))]:
            predictions={key:[] for key in ('torch','original_onnx','candidate_onnx')}
            for x in inputs:
                predictions['torch'].append(model(torch.from_numpy(x[None])).numpy()[0])
                for name,session in sessions.items():
                    predictions[name].append(session.run(None,{session.get_inputs()[0].name:x[None]})[0][0])
            predictions={key:np.asarray(value,dtype='<f4') for key,value in predictions.items()}
            row=dict(samples=len(inputs),comparisons={})
            for name,values in predictions.items():
                np.save(out/f'{group}.{name}.npy',values)
                if name!='torch':
                    ref=predictions['torch']
                    passed=bool(np.allclose(values,ref,rtol=1e-4,atol=1e-5))
                    same=bool(np.array_equal(values.argmax(1),ref.argmax(1)))
                    row['comparisons'][name]=dict(numerical_passed=passed,classes_all_equal=same,
                                                   max_abs=float(abs(values-ref).max()))
                    if not passed or not same:raise ValueError(group+' '+name+' FP32 equivalence failed')
            if group=='golden':
                row['torch_vs_delivered_golden_max_abs']=float(abs(predictions['torch']-expected).max())
                if not np.allclose(predictions['torch'],expected,rtol=1e-4,atol=1e-5):raise ValueError('Golden mismatch')
            else:
                labels=np.load(ROOT/f'data/{group}.labels.npy')
                row['task_quality']=summary(predictions['torch'].argmax(1),labels)
            report['groups'][group]=row
            print(group,row,flush=True)
    report.update(status='fp32_equivalence_passed',scope='same-day separated engineering screening, not original validation reproduction')
    save(out/'report.json',report)


if __name__=='__main__':main()
